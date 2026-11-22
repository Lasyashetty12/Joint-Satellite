"""
m4_eo_cnn.py - image encoder for flood detection: ResNet-18 adapted to satellite bands.

* the first convolution takes C input bands instead of 3 (optical: 6, SAR: 2, both: 8)
* optional ImageNet initialisation (--pretrained): the RGB filters are averaged and copied to
  every input band, a common trick when the band count differs from 3
* forward(x) -> flood logit;  embed(x) -> 512-d feature used later by the fusion models
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EOCNN(nn.Module):
    def __init__(self, in_ch: int, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = net.conv1
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                net.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Linear(512, 1)
        self.in_ch = in_ch

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(1)


def load_encoder(path, device: str = "cpu") -> EOCNN:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = EOCNN(ck["in_ch"], pretrained=False)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def encode(model: EOCNN, x, batch: int = 64) -> tuple:
    """numpy [N,C,H,W] -> (embeddings [N,512] float32, probabilities [N])."""
    import numpy as np
    embs, probs = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(np.asarray(x[i:i + batch], dtype=np.float32))
        e = model.embed(xb)
        embs.append(e.numpy()); probs.append(torch.sigmoid(model.head(e).squeeze(1)).numpy())
    return np.concatenate(embs), np.concatenate(probs)
