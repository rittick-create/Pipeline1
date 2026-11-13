"""PACX-MAE projection heads and losses adapted to ECG/labs students."""

import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from Models.models import MAECXREncoder, ECGFounderEncoder, load_labs
from Models.labs_dae import LabsDenoisingAE
from Utils.utils import build_param_groups


class ProjectionHead(nn.Module):
    # Original PACX-MAE head architecture and initialization.
    def __init__(self, input_dim, output_dim=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        nn.init.trunc_normal_(self.mlp[1].weight, std=0.02)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x):
        return self.mlp(x)


class CLIPRegressionLoss(nn.Module):
    def __init__(self, clip_weight=0.5, regression_weight=1.0, temperature=0.07):
        super().__init__()
        self.clip_weight, self.regression_weight = clip_weight, regression_weight
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def clip_loss(self, x, y):
        x, y = F.normalize(x, dim=-1, eps=1e-6), F.normalize(y, dim=-1, eps=1e-6)
        logits = self.logit_scale.exp().clamp(1, 100) * (x @ y.T)
        labels = torch.arange(x.size(0), device=x.device)
        return 0.5 * (
            F.cross_entropy(logits, labels, label_smoothing=0.02)
            + F.cross_entropy(logits.T, labels, label_smoothing=0.02)
        )

    @staticmethod
    def regression_loss(pred, target):
        return (
            1
            - (
                F.normalize(pred, dim=-1, eps=1e-6)
                * F.normalize(target.detach(), dim=-1, eps=1e-6)
            )
            .sum(-1)
            .mean()
        )

    def forward(self, out):
        contrastive = 0.5 * (
            self.clip_loss(out["cxr_shared"], out["ecg_shared"])
            + self.clip_loss(out["cxr_shared"], out["labs_shared"])
        )
        regression = 0.5 * (
            self.regression_loss(out["predicted_cxr_ecg"], out["cxr_emb"])
            + self.regression_loss(out["predicted_cxr_labs"], out["cxr_emb"])
        )
        total = self.clip_weight * contrastive + self.regression_weight * regression
        return total, {
            "total": total,
            "contrastive": contrastive,
            "regression": regression,
        }


class CrossModalDistillation(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=0.05,
        max_epochs=50,
        warmup_epochs=5,
        clip_weight=0.5,
        regression_weight=1.0,
        temperature=0.07,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cxr_encoder = MAECXREncoder()
        self.ecg_encoder = ECGFounderEncoder()
        self.labs_encoder = LabsDenoisingAE(50).encoder
        self.cxr_to_shared = ProjectionHead(768)
        self.ecg_to_shared = ProjectionHead(1024)
        self.labs_to_shared = ProjectionHead(256)
        self.ecg_to_cxr = ProjectionHead(1024, 768)
        self.labs_to_cxr = ProjectionHead(256, 768)
        self.loss_module = CLIPRegressionLoss(
            clip_weight, regression_weight, temperature
        )
        self.cxr_encoder.requires_grad_(False)
        self.ecg_encoder.requires_grad_(False)
        self.ecg_encoder.backbone.stage_list[-1].block_list[-1].requires_grad_(True)
        self.train()

    def initialize_encoders(self, cxr_checkpoint, ecg_checkpoint, labs_checkpoint):
        self.cxr_encoder.load_mae(cxr_checkpoint)
        self.ecg_encoder.initialize(ecg_checkpoint)
        load_labs(self.labs_encoder, labs_checkpoint)
        self.save_hyperparameters(
            {
                "encoder_sources": {
                    "cxr": str(cxr_checkpoint),
                    "ecg": str(ecg_checkpoint),
                    "labs": str(labs_checkpoint),
                }
            }
        )

    def train(self, mode=True):
        super().train(mode)
        self.cxr_encoder.eval()
        self.ecg_encoder.eval()
        self.ecg_encoder.backbone.stage_list[-1].block_list[-1].train(mode)
        return self

    def forward(self, cxr, ecg, labs, labs_mask):
        with torch.no_grad():
            teacher = self.cxr_encoder(cxr)
        ecg = self.ecg_encoder(ecg)
        labs = self.labs_encoder(torch.cat([labs, labs_mask], dim=1))
        return {
            "cxr_emb": teacher,
            "ecg_emb": ecg,
            "labs_emb": labs,
            "cxr_shared": self.cxr_to_shared(teacher),
            "ecg_shared": self.ecg_to_shared(ecg),
            "labs_shared": self.labs_to_shared(labs),
            "predicted_cxr_ecg": self.ecg_to_cxr(ecg),
            "predicted_cxr_labs": self.labs_to_cxr(labs),
        }

    def _step(self, batch, mode):
        out = self(batch["cxr"], batch["ecg"], batch["labs"], batch["labs_mask"])
        loss, metrics = self.loss_module(out)
        self.log_dict(
            {f"{mode}/loss_{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch["ecg"].size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        if len(batch["ecg"]) < 2:
            raise ValueError(
                "Stage 2 contrastive training needs at least two admissions per batch"
            )
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            build_param_groups(self, self.hparams.weight_decay), lr=self.hparams.lr
        )

        def schedule(epoch):
            warmup = self.hparams.warmup_epochs
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, self.hparams.max_epochs - warmup)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
        }
