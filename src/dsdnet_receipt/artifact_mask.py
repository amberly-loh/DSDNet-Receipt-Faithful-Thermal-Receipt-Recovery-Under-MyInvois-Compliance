from __future__ import annotations

import torch
from torch import nn
from functools import lru_cache
from pathlib import Path

import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class ReceiptArtifactMaskNet(nn.Module):
    """Small U-Net trained only to localise paper artifacts, never text."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.bottleneck = ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.enc1(image)
        second = self.enc2(self.pool(first))
        centre = self.bottleneck(self.pool(second))
        second_up = self.dec2(torch.cat((self.up2(centre), second), dim=1))
        first_up = self.dec1(torch.cat((self.up1(second_up), first), dim=1))
        return self.head(first_up)


@lru_cache(maxsize=2)
def _load_model(checkpoint: str) -> tuple[ReceiptArtifactMaskNet, torch.device]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(checkpoint), map_location=device)
    width = int(payload.get("config", {}).get("training", {}).get("width", 16))
    model = ReceiptArtifactMaskNet(width).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, device


def predict_artifact_mask(
    image: np.ndarray, checkpoint: str | Path, threshold: float = 0.95
) -> np.ndarray:
    """Predict crease pixels without modifying or synthesising document content."""
    model, device = _load_model(str(Path(checkpoint).resolve()))
    height, width = image.shape
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode="edge")
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0, :height, :width]
    return probability.cpu().numpy() >= float(threshold)
