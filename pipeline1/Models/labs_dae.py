import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

class LabsDenoisingAE(nn.Module):
    """
    Mask-aware denoising autoencoder for lab percentiles.
    Encoder sees (x, m); decoder reconstructs x only.
    """

    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, m):
        z = self.encoder(torch.cat([x, m], dim=1))
        x_hat = self.decoder(z)
        return z, x_hat
    
    def encode(self, x, m):
        return self.encoder(torch.cat([x, m], dim=1))

