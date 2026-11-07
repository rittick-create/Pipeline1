"""Pretrained ECGFounder backbone with DrFuse-style shared/distinct branches."""

from pathlib import Path

import torch
from torch import nn

from third_party.ecgfounder.net1d import Net1D


DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[2] / "ecgfounder"
                      / "checkpoint" / "12_lead_ECGFounder.pth")


class ECGFounderEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_classes=14, dropout=0.3,
                 checkpoint_path=None, freeze_backbone=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            n_classes=150, use_bn=False, use_do=False, verbose=False,
        )
        # Discard the original 150-label classifier, retaining pooled features.
        self.backbone.dense = nn.Identity()
        path = Path(checkpoint_path or DEFAULT_CHECKPOINT).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ECGFounder pretrained checkpoint not found: {path}")
        # Load only trusted official/local weights. Every backbone key must match.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        weights = {key: value for key, value in saved["state_dict"].items()
                   if not key.startswith("dense.")}
        self.backbone.load_state_dict(weights, strict=True)
        self.backbone.requires_grad_(not freeze_backbone)

        self.shared = nn.Sequential(
            nn.Linear(1024, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.distinct = nn.Sequential(
            nn.Linear(1024, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, ecg):
        """Preprocessed (B, 12, 5000) -> two (B, H) vectors and (B, 14)."""
        embedding = self.backbone(ecg)
        shared = self.shared(embedding)
        distinct = self.distinct(embedding)
        predictions = self.classifier(distinct).sigmoid()
        return shared, distinct, predictions
