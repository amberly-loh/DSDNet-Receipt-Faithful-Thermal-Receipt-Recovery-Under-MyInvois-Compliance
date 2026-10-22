from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssim_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    mu_l = F.avg_pool2d(left, 11, stride=1, padding=5)
    mu_r = F.avg_pool2d(right, 11, stride=1, padding=5)
    var_l = F.avg_pool2d(left * left, 11, stride=1, padding=5) - mu_l * mu_l
    var_r = F.avg_pool2d(right * right, 11, stride=1, padding=5) - mu_r * mu_r
    cov = F.avg_pool2d(left * right, 11, stride=1, padding=5) - mu_l * mu_r
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_l * mu_r + c1) * (2 * cov + c2)) / ((mu_l**2 + mu_r**2 + c1) * (var_l + var_r + c2) + 1e-12)
    return 1.0 - score.mean()


def stroke_gradient_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Penalise broken character edges without requiring OCR in the train loop."""
    sobel_x = left.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)
    left_x = F.conv2d(left, sobel_x, padding=1); right_x = F.conv2d(right, sobel_x, padding=1)
    left_y = F.conv2d(left, sobel_y, padding=1); right_y = F.conv2d(right, sobel_y, padding=1)
    return F.l1_loss(left_x, right_x) + F.l1_loss(left_y, right_y)


def foreground_text_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Weight errors on dark printed strokes more than white paper."""
    foreground_weight = (1.0 - right).pow(1.5)
    return (torch.abs(left - right) * foreground_weight).sum() / foreground_weight.sum().clamp_min(1e-6)


def crop_and_augment(clean: np.ndarray, degraded: np.ndarray, size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height, width = clean.shape
    if height < size or width < size:
        scale = max(size / height, size / width)
        import cv2
        clean = cv2.resize(clean, (int(np.ceil(width * scale)), int(np.ceil(height * scale))), interpolation=cv2.INTER_CUBIC)
        degraded = cv2.resize(degraded, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC)
        height, width = clean.shape
    top = rng.randrange(height - size + 1)
    left = rng.randrange(width - size + 1)
    clean = clean[top : top + size, left : left + size]
    degraded = degraded[top : top + size, left : left + size]
    if rng.random() < 0.5:
        clean, degraded = np.fliplr(clean), np.fliplr(degraded)
    rotation = rng.randrange(4)
    return np.ascontiguousarray(np.rot90(clean, rotation)), np.ascontiguousarray(np.rot90(degraded, rotation))


def add_natural_receipt_degradation(
    degraded: np.ndarray, settings: dict, seed: int
) -> tuple[np.ndarray, dict]:
    """Add deterministic paper texture and folds while retaining a clean target."""
    import cv2

    rng = np.random.default_rng(seed)
    if not settings.get("enabled", False) or rng.random() > float(settings["probability"]):
        return degraded, {"applied": False}
    image = degraded.astype(np.float32) / 255.0
    amplitude = float(rng.uniform(*settings["texture_amplitude"]))
    scale = float(rng.uniform(*settings["texture_scale_pixels"]))
    noise = rng.normal(0.0, 1.0, image.shape).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), scale)
    texture /= max(float(texture.std()), 1e-6)
    image = np.clip(image + amplitude * texture, 0.0, 1.0)
    crease_count = 0
    if rng.random() < float(settings["crease_probability"]):
        low, high = settings["crease_count"]
        crease_count = int(rng.integers(int(low), int(high) + 1))
        h, w = image.shape
        for _ in range(crease_count):
            y = int(rng.integers(max(2, h // 12), max(3, h - h // 12)))
            slope = float(rng.uniform(-0.08, 0.08))
            strength = float(rng.uniform(*settings["crease_strength"]))
            thickness = int(rng.integers(int(settings["crease_thickness"][0]), int(settings["crease_thickness"][1]) + 1))
            overlay = np.zeros_like(image)
            cv2.line(overlay, (0, y), (w - 1, int(np.clip(y + slope * w, 0, h - 1))), 1.0, thickness)
            image = np.clip(image - strength * cv2.GaussianBlur(overlay, (0, 0), 1.2), 0.0, 1.0)
    return (image * 255.0 + 0.5).astype(np.uint8), {
        "applied": True, "texture_amplitude": amplitude, "texture_scale_pixels": scale,
        "crease_count": crease_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Chapter 4 DSDNet receipt trainer")
    parser.add_argument("--config", default=str(REPO / "configs" / "dsdnet_receipt_train.json"))
    parser.add_argument("--feasibility", action="store_true", help="Run one non-formal optimisation step using the configured crop")
    parser.add_argument("--development-iterations", type=int, help="Run a non-formal development budget without changing the locked formal total")
    parser.add_argument("--development-checkpoint-interval", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-weights", type=Path, help="Start a new development run from model weights only")
    parser.add_argument("--resume-learning-rate", type=float)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    train_config_path = Path(args.config).resolve()
    locked = json.loads(train_config_path.read_text(encoding="utf-8"))
    protocol, _ = load_config(REPO / "configs" / "default.json")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this eight-stage training configuration")
    seed = int(locked["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    network = locked["network"]
    model = DSDNetReceiptAdapter(
        protocol["dsdnet"]["source_path"],
        stages=int(network["stages"]),
        branches=int(network["maxout_branches"]),
        prior_channels=int(network["prior_channels"]),
        data_channels=int(network["data_channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=locked["optimizer"]["learning_rate"],
        betas=(locked["optimizer"]["beta1"], locked["optimizer"]["beta2"]),
        eps=locked["optimizer"]["epsilon"], weight_decay=locked["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=locked["schedule"]["t_max_iterations"], eta_min=locked["schedule"]["eta_min"])
    use_amp = bool(locked["training"].get("mixed_precision", False))
    use_checkpointing = bool(locked["training"].get("gradient_checkpointing", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start = 0
    if args.initial_weights and args.resume:
        raise SystemExit("Use either --initial-weights or --resume, not both")
    if args.initial_weights:
        state = torch.load(args.initial_weights, map_location=device)
        model.load_state_dict(state.get("model", state.get("model_state_dict", state)))
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["iteration"])
        if args.resume_learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_learning_rate
            scheduler.base_lrs = [args.resume_learning_rate for _ in optimizer.param_groups]
    generation_manifest = (train_config_path.parent / locked["generation_manifest"]).resolve()
    records = [json.loads(line) for line in generation_manifest.read_text(encoding="utf-8").splitlines() if line]
    records = [record for record in records if record["project_partition"] == "train"]
    total = 1 if args.feasibility else (args.development_iterations or int(locked["training"]["total_iterations"]))
    run_kind = "training_feasibility" if args.feasibility else ("training_development" if args.development_iterations else "training")
    output = REPO / "artifacts" / (args.output_name or run_kind)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.jsonl"
    crop = int(locked["training"]["crop_height"])
    model.train()
    for iteration in range(start, total):
        record = records[iteration % len(records)]
        clean = read_gray(record["clean_reference"])
        degraded = read_gray(record["degraded_path"])
        clean, degraded = crop_and_augment(clean, degraded, crop, random.Random(seed + iteration))
        degraded, natural_augmentation = add_natural_receipt_degradation(
            degraded, locked.get("natural_receipt_augmentation", {}), seed + 1000003 + iteration
        )
        target = torch.from_numpy(clean).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        kernel = gaussian_kernel(float(record["parameters"]["blur_sigma"])).to(device)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp):
            restored = model(
                observed,
                kernel,
                retain_diagnostics=False,
                gradient_checkpointing=use_checkpointing,
            )["restored"]
            l1 = F.l1_loss(restored, target)
            structural = ssim_loss(restored, target)
            stroke = stroke_gradient_loss(restored, target)
            foreground = foreground_text_loss(restored, target)
            loss = l1 + float(locked["objective"]["ssim_coefficient"]) * structural + float(locked["objective"].get("stroke_gradient_coefficient", 0.0)) * stroke + float(locked["objective"].get("foreground_text_coefficient", 0.0)) * foreground
        scaler.scale(loss).backward()
        if locked["training"].get("gradient_clipping", False):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(locked["training"]["gradient_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        row = {
            "iteration": iteration + 1,
            "source_id": record["source_id"],
            "variant_id": record["variant_id"],
            "loss": float(loss.detach()),
            "l1": float(l1.detach()),
            "ssim_loss": float(structural.detach()),
            "stroke_gradient_loss": float(stroke.detach()),
            "foreground_text_loss": float(foreground.detach()),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "mixed_precision": use_amp,
            "gradient_checkpointing": use_checkpointing,
            "natural_receipt_augmentation": natural_augmentation,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        checkpoint_interval = args.development_checkpoint_interval if args.development_iterations else int(locked["training"]["checkpoint_interval"])
        if args.feasibility or (iteration + 1) % checkpoint_interval == 0 or iteration + 1 == total:
            checkpoint = output / f"checkpoint_{iteration + 1:07d}.pt"
            payload = {
                "iteration": iteration + 1,
                "model": model.state_dict(),
                "config_sha256": sha256(train_config_path),
                "run_kind": "non-formal development" if args.development_iterations or args.feasibility else "formal candidate",
                "locked_total_iterations": int(locked["training"]["total_iterations"]),
            }
            if iteration + 1 == total or not args.development_iterations:
                payload["optimizer"] = optimizer.state_dict()
                payload["scheduler"] = scheduler.state_dict()
            torch.save(payload, checkpoint)
            (output / f"checkpoint_{iteration + 1:07d}.sha256").write_text(sha256(checkpoint), encoding="ascii")


if __name__ == "__main__":
    main()
