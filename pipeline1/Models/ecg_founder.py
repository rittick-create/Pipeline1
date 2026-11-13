"""ECGFounder feature encoder for reproducible construction and checkpoint loading.

Net1D and its convolution, residual block and stage definitions are in net1d.py.
This matches the ECG backbone used by PACX-MAE's ECG extraction notebook.
Construct ECGFounderEncoder() for random initialization, then call initialize(path)
for the official pretrained weights. Stage 2 adapts only its last residual block.
"""

from pathlib import Path
import torch
from torch import nn
from .net1d import Net1D

DEFAULT_ECG = (
    Path(__file__).resolve().parents[2] / "ecgfounder/checkpoint/12_lead_ECGFounder.pth"
)


class ECGFounderEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=150,
            use_bn=False,
            use_do=False,
            verbose=False,
        )
        self.backbone.dense = nn.Identity()

    def initialize(self, path):
        weights = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        self.backbone.load_state_dict(
            {k: v for k, v in weights.items() if not k.startswith("dense.")},
            strict=True,
        )

    def forward(self, signal):
        return self.backbone(signal)
