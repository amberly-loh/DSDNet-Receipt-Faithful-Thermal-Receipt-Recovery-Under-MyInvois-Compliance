from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.artifact_mask import ReceiptArtifactMaskNet  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from train_dsdnet_receipt import add_natural_receipt_degradation  # noqa: E402


def crop(image: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    import cv2

    height, width = image.shape
    if height < size or width < size:
        scale = max(size / height, size / width)
        image = cv2.resize(
            image,
            (int(np.ceil(width * scale)), int(np.ceil(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = image.shape
    top = rng.randrange(height - size + 1)
    left = rng.randrange(width - size + 1)
    return image[top : top + size, left : left + size]


def make_batch(
    records: list[dict], settings: dict, size: int, batch_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = [], []
    for offset in range(batch_size):
        item_seed = seed * 1009 + offset
        rng = random.Random(item_seed)
        record = records[item_seed % len(records)]
        clean = crop(read_gray(record["clean_reference"]), size, rng)
        observed, _, masks = add_natural_receipt_degradation(
            clean, settings, item_seed + 7_000_003, return_masks=True
        )
        blob_mask = np.zeros_like(clean, dtype=np.uint8)
        blob_probability = float(settings.get("blob_probability", 0.0))
        if rng.random() < blob_probability:
            import cv2

            low, high = settings.get("blob_count", [2, 12])
            count = rng.randint(int(low), int(high))
            h, w = observed.shape
            overlay = np.zeros_like(observed, dtype=np.uint8)
            for _ in range(count):
                cx, cy = rng.randrange(w), rng.randrange(h)
                rmin, rmax = settings.get("blob_radius", [2, 12])
                rx = rng.randint(int(rmin), int(rmax))
                ry = rng.randint(int(rmin), int(rmax))
                angle = rng.uniform(0.0, 180.0)
                cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, 255, -1, cv2.LINE_AA)
            # Real crumpled receipts contain irregular toner-like islands and
            # branching smears rather than clean ellipses. Add correlated
            # multi-scale fields and broken strokes so training covers the
            # compact grey blobs seen in natural scans.
            noise = np.random.default_rng(item_seed + 91).random((h, w)).astype(np.float32)
            coarse = cv2.GaussianBlur(noise, (0, 0), rng.uniform(4.0, 13.0))
            cutoff = float(np.percentile(coarse, rng.uniform(86.0, 94.0)))
            irregular = (coarse >= cutoff).astype(np.uint8) * 255
            irregular = cv2.morphologyEx(
                irregular,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            for _ in range(rng.randint(1, 5)):
                start = (rng.randrange(w), rng.randrange(h))
                end = (
                    int(np.clip(start[0] + rng.randint(-w // 3, w // 3), 0, w - 1)),
                    int(np.clip(start[1] + rng.randint(-h // 8, h // 8), 0, h - 1)),
                )
                cv2.line(irregular, start, end, 255, rng.randint(1, 4), cv2.LINE_AA)
            overlay = np.maximum(overlay, irregular)
            sigma = rng.uniform(*settings.get("blob_blur_sigma", [0.4, 2.2]))
            soft = cv2.GaussianBlur(overlay, (0, 0), sigma).astype(np.float32) / 255.0
            strength = rng.uniform(*settings.get("blob_strength", [0.08, 0.42]))
            polarity = -1.0 if rng.random() < 0.85 else 1.0
            observed = np.clip(observed.astype(np.float32) + polarity * 255.0 * strength * soft, 0, 255).astype(np.uint8)
            blob_mask = soft >= 0.10
        images.append(observed.astype(np.float32) / 255.0)
        # Broad texture covers much of genuine thermal paper and caused the
        # first v2 experiment to label almost every pixel as artifact. Train
        # only on explicit fold geometry and compact synthetic blobs. Smooth
        # paper texture remains the responsibility of background estimation.
        target = masks["crease"] | blob_mask.astype(bool)
        targets.append(target.astype(np.float32))
    return (
        torch.from_numpy(np.stack(images)).unsqueeze(1),
        torch.from_numpy(np.stack(targets)).unsqueeze(1),
    )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


@torch.inference_mode()
def validate(
    model: ReceiptArtifactMaskNet,
    records: list[dict],
    settings: dict,
    size: int,
    device: torch.device,
    seed: int,
    batches: int = 24,
    threshold: float = 0.95,
) -> dict[str, float]:
    model.eval()
    intersections = unions = false_positive = background = 0
    for index in range(batches):
        observed, target = make_batch(records, settings, size, 2, seed + 50_000 + index)
        prediction = torch.sigmoid(model(observed.to(device))) >= float(threshold)
        truth = target.to(device) >= 0.5
        intersections += int((prediction & truth).sum())
        unions += int((prediction | truth).sum())
        false_positive += int((prediction & ~truth).sum())
        background += int((~truth).sum())
    model.train()
    return {
        "crease_iou": intersections / max(unions, 1),
        "false_positive_rate": false_positive / max(background, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a validation-gated receipt crease mask")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "artifact_mask_validation.json")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output-name", default="training_artifact_mask_v1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest = (args.config.parent / config["generation_manifest"]).resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    train_records = [row for row in records if row["project_partition"] == "train"]
    validation_records = [row for row in records if row["project_partition"] == "validation"]
    if not train_records or not validation_records:
        raise SystemExit("Both training and validation records are required")

    settings = config["natural_receipt_augmentation"]
    training = config["training"]
    size = int(training["crop_size"])
    batch_size = int(training["batch_size"])
    total = args.iterations or int(training["iterations"])
    interval = int(training["checkpoint_interval"])
    model = ReceiptArtifactMaskNet(int(training["width"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    positive_weight = torch.tensor(
        [float(training.get("positive_weight", 3.0))], device=device
    )
    output = REPO / "artifacts" / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.jsonl"
    model.train()
    best: dict | None = None
    for iteration in range(1, total + 1):
        observed, target = make_batch(train_records, settings, size, batch_size, seed + iteration)
        observed, target = observed.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(observed)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
        dice = dice_loss(logits, target)
        loss = bce + dice
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"iteration": iteration, "loss": float(loss.detach()), "bce": float(bce.detach()), "dice_loss": float(dice.detach())}
        if iteration % interval == 0 or iteration == total:
            metrics = validate(
                model,
                validation_records,
                settings,
                size,
                device,
                seed + iteration,
                threshold=float(config["selection"].get("inference_threshold", 0.95)),
            )
            row.update(metrics)
            checkpoint = output / f"checkpoint_{iteration:07d}.pt"
            torch.save({"iteration": iteration, "model": model.state_dict(), "config": config}, checkpoint)
            if best is None or (metrics["crease_iou"], -metrics["false_positive_rate"]) > (best["crease_iou"], -best["false_positive_rate"]):
                best = {**metrics, "checkpoint": str(checkpoint.resolve()), "iteration": iteration}
                torch.save({"iteration": iteration, "model": model.state_dict(), "config": config}, output / "best.pt")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if iteration == 1 or iteration % 25 == 0:
            print(json.dumps(row), flush=True)
    summary = {
        "partition": "validation only",
        "formal_test_evidence": False,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "best": best,
        "deployment_enabled": bool(
            best
            and best["crease_iou"] >= float(config["selection"]["minimum_crease_iou"])
            and best["false_positive_rate"] <= float(config["selection"]["maximum_false_positive_rate"])
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
