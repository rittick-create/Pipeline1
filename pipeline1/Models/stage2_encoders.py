"""Stage 2 adapters around the original Stage 1 architectures."""
from functools import partial
from pathlib import Path
import hashlib

import torch
from torch import nn
from .ecg_founder import Net1D
from .labs_encoder import LabsDenoisingAE
from .mae import get_2d_sincos_pos_embed
from .models import MAECXREncoder


class ECGFounderEncoder(Net1D):
    def __init__(self):
        # Matches ft_12lead_ECGFounder in the original extraction notebook.
        super().__init__(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
            stride=2, groups_width=16, n_classes=1,
            use_bn=False, use_do=False, return_features=True, verbose=False,
        )
        self.dense = nn.Identity()  # No ECG classification head in Stage 2.

    def forward(self, ecg):
        return super().forward(ecg)[1]


class LabEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LabsDenoisingAE(input_dim=50, latent_dim=256).encoder

    def forward(self, values, observed):
        observed = observed.float()
        values = torch.where(observed.bool(), values.float(), 0.0)
        return self.encoder(torch.cat([values, observed], dim=1))


def make_cxr_teacher():
    model = MAECXREncoder(
        embed_dim=768, depth=12, num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    positions = get_2d_sincos_pos_embed(768, 14, cls_token=True)
    with torch.no_grad():
        model.pos_embed.copy_(torch.from_numpy(positions).float().unsqueeze(0))
        nn.init.normal_(model.cls_token, std=0.02)
    return model


def load_encoder_weights(model, path, modality):
    """Load trusted Stage 1 checkpoints; reject incompatible backbone keys."""
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('model.'):
            key = key[len('model.'):]
        if modality == 'cxr' and (key.startswith('decoder_') or key == 'mask_token'):
            continue
        if modality == 'ecg' and key.startswith('dense.'):
            continue
        if modality == 'labs' and key.startswith('decoder.'):
            continue
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}

