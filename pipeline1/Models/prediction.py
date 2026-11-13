"""Frozen Stage 2 student features and one linear probe."""

import torch
from torch import nn
from .models import ECGFounderEncoder, read_checkpoint
from .labs_dae import LabsDenoisingAE


class StudentClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.ecg_encoder = ECGFounderEncoder()
        self.labs_encoder = LabsDenoisingAE(50).encoder
        self.ecg_encoder.requires_grad_(False)
        self.labs_encoder.requires_grad_(False)
        self.head = nn.Linear(1280, num_classes)
        self.train()

    def load_students(self, path):
        saved = read_checkpoint(path)
        if saved["hyper_parameters"].get("stage") != "distillation":
            raise ValueError("Expected a Pipeline1 Stage 2 checkpoint")
        for name in ("ecg_encoder", "labs_encoder"):
            prefix = name + "."
            state = {
                k[len(prefix) :]: v
                for k, v in saved["state_dict"].items()
                if k.startswith(prefix)
            }
            getattr(self, name).load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.ecg_encoder.eval()
        self.labs_encoder.eval()
        return self

    def forward(self, ecg, labs, labs_mask):
        with torch.no_grad():
            ecg = self.ecg_encoder(ecg)
            labs = self.labs_encoder(torch.cat([labs, labs_mask], dim=1))
        return self.head(torch.cat([ecg, labs], dim=1))
