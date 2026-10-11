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


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Chapter 4 DSDNet receipt trainer")
    parser.add_argument("--config", default=str(REPO / "configs" / "dsdnet_receipt_train.json"))
    parser.add_argument("--feasibility", action="store_true", help="Run one non-formal 128x128 optimisation step")
    parser.add_argument("--resume", type=Path)
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
    model = DSDNetReceiptAdapter(protocol["dsdnet"]["source_path"], stages=8, branches=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=locked["optimizer"]["learning_rate"],
        betas=(locked["optimizer"]["beta1"], locked["optimizer"]["beta2"]),
        eps=locked["optimizer"]["epsilon"], weight_decay=locked["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=locked["schedule"]["t_max_iterations"], eta_min=locked["schedule"]["eta_min"])
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["iteration"])
    generation_manifest = (train_config_path.parent / locked["generation_manifest"]).resolve()
    records = [json.loads(line) for line in generation_manifest.read_text(encoding="utf-8").splitlines() if line]
    records = [record for record in records if record["project_partition"] == "train"]
    total = 1 if args.feasibility else int(locked["training"]["total_iterations"])
    output = REPO / "artifacts" / ("training_feasibility" if args.feasibility else "training")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.jsonl"
    crop = int(locked["training"]["crop_height"])
    model.train()
    for iteration in range(start, total):
        record = records[iteration % len(records)]
        clean = read_gray(record["clean_reference"])
        degraded = read_gray(record["degraded_path"])
        clean, degraded = crop_and_augment(clean, degraded, crop, random.Random(seed + iteration))
        target = torch.from_numpy(clean).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        kernel = gaussian_kernel(float(record["parameters"]["blur_sigma"])).to(device)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        restored = model(observed, kernel, retain_diagnostics=False)["restored"]
        l1 = F.l1_loss(restored, target)
        structural = ssim_loss(restored, target)
        loss = l1 + float(locked["objective"]["ssim_coefficient"]) * structural
        loss.backward()
        optimizer.step()
        scheduler.step()
        row = {
            "iteration": iteration + 1,
            "source_id": record["source_id"],
            "variant_id": record["variant_id"],
            "loss": float(loss.detach()),
            "l1": float(l1.detach()),
            "ssim_loss": float(structural.detach()),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        checkpoint_interval = int(locked["training"]["checkpoint_interval"])
        if args.feasibility or (iteration + 1) % checkpoint_interval == 0:
            checkpoint = output / f"checkpoint_{iteration + 1:07d}.pt"
            torch.save({"iteration": iteration + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config_sha256": sha256(train_config_path)}, checkpoint)
            (output / f"checkpoint_{iteration + 1:07d}.sha256").write_text(sha256(checkpoint), encoding="ascii")


if __name__ == "__main__":
    main()
