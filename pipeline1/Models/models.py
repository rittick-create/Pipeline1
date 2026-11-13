"""Encoder interfaces for reversed PACX distillation and linear probing."""

import torch
from torch import nn
from .mae import vit_base_patch16
from .ecg_founder import ECGFounderEncoder as ECGFounderEncoder
from .ecg_founder import DEFAULT_ECG as DEFAULT_ECG


def read_checkpoint(path):
    # Project checkpoints contain trusted Lightning metadata and optimizer state.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


class MAECXREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        mae = vit_base_patch16()
        # Reuse the exact encoder including normalization eps and positional weights.
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed
        self.blocks = mae.blocks
        self.norm = mae.norm

    def forward(self, image):
        x = self.patch_embed(image) + self.pos_embed[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def load_mae(self, path):
        saved = read_checkpoint(path)
        if saved["hyper_parameters"].get("stage") != "mae":
            raise ValueError("Expected a Pipeline1 Stage 1 MAE checkpoint")
        weights = {
            k.removeprefix("model."): v
            for k, v in saved["state_dict"].items()
            if k.startswith("model.")
            and not k.startswith(("model.decoder", "model.mask_token"))
        }
        self.load_state_dict(weights, strict=True)


def load_labs(encoder, path):
    saved = read_checkpoint(path)
    if saved["hyper_parameters"].get("stage") != "labs":
        raise ValueError("Expected a Pipeline1 Stage 1 labs checkpoint")
    weights = {
        k.removeprefix("model.encoder."): v
        for k, v in saved["state_dict"].items()
        if k.startswith("model.encoder.")
    }
    encoder.load_state_dict(weights, strict=True)
