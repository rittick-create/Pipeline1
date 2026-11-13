"""Adapted from PACX-MAE Models/labs_dae.py; import-safe training module."""
import math
import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from Models.labs_dae import LabsDenoisingAE

class LabsDAELightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        corruption_prob: float = 0.15,  # denoising strength
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LabsDenoisingAE(input_dim, latent_dim)

    def forward(self, x, m):
        return self.model(x, m)

    def _corrupt(self, x, m):
        """
        m: observed mask (1=present, 0=missing)
        Returns: x_corrupt, m_corrupt, drop_mask (bool)
        """
        x = x.float()
        m = m.float()

        # don't let any placeholder at missing positions leak in
        x = x * m

        rand = torch.rand_like(m)
        drop_mask = (rand < self.hparams.corruption_prob) & (m > 0.5)  # only drop observed

        x_corrupt = x.clone()
        m_corrupt = m.clone()

        x_corrupt[drop_mask] = 0.0
        m_corrupt[drop_mask] = 0.0

        return x_corrupt, m_corrupt, drop_mask

    def _masked_mse(self, x_hat, x, m):
        diff = (x_hat - x) ** 2
        return (diff * m).sum() / (m.sum() + 1e-6)

    def training_step(self, batch, batch_idx):
        x, m = batch

        x_corrupt, m_corrupt, _ = self._corrupt(x, m)
        _, x_hat = self(x_corrupt, m_corrupt)

        loss = self._masked_mse(x_hat, x, m)

        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def _validation_step(self, batch, batch_idx):
        x, m = batch
        
        # --- Task A: Clean Reconstruction (Autoencoder) ---
        # Checks if model can compress/decompress without losing info
        z_clean, x_hat_clean = self(x, m)
        clean_loss = self._masked_mse(x_hat_clean, x, m)
        
        # --- Task B: Denoising Performance (Imputation) ---
        # Checks if model understands correlations (e.g., inferring missing values)
        x_corrupt, m_corrupt, drop_mask = self._corrupt(x, m)
        _, x_hat_denoise = self(x_corrupt, m_corrupt)
        
        # Loss 1: Overall Denoising Error (Global structure)
        denoise_loss = self._masked_mse(x_hat_denoise, x, m)
        
        # Loss 2: Imputation Specific Error (Did it guess the missing parts right?)
        # We look specifically at the indices where drop_mask is True
        if drop_mask.sum() > 0:
            imputation_error = (x_hat_denoise - x) ** 2
            imputation_mse = imputation_error[drop_mask].mean()
        else:
            imputation_mse = torch.tensor(0.0, device=x.device)

        # --- Task C: Latent Space Health ---
        # Check for posterior collapse (if std is near 0, the encoder is dead)
        z_std = z_clean.std()

        # Logging
        self.log_dict({
            "val/clean_recon_mse": clean_loss,
            "val/denoising_mse": denoise_loss,
            "val/imputation_mse": imputation_mse, # The most important metric for you
            "val/z_std": z_std
        }, prog_bar=True)
        
        return denoise_loss
    
    def validation_step(self, batch, batch_idx):
        # Repeat validation corruption across epochs without resetting training RNG.
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(42 + batch_idx)
            return self._validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (
                step - self.hparams.warmup_steps
            ) / max(1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

