"""Reversed PACX: ECG + labs students, fully frozen CXR teacher.

Projection architecture, symmetric CLIP matching and cosine regression are
adapted from the original Modules/pacx_lit.py. Stage 1 is unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from Models.stage2_encoders import (
    ECGFounderEncoder, LabEncoder, make_cxr_teacher, load_encoder_weights,
)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.mlp(x)


class CLIPRegressionLoss(nn.Module):
    def __init__(self, clip_weight=0.5, regression_weight=1.0,
                 temperature=0.07, label_smoothing=0.02):
        super().__init__()
        if temperature <= 0 or clip_weight < 0 or regression_weight < 0:
            raise ValueError('Temperature must be positive; loss weights nonnegative.')
        if clip_weight + regression_weight == 0:
            raise ValueError('At least one loss weight must be positive.')
        self.clip_weight = clip_weight
        self.regression_weight = regression_weight
        self.label_smoothing = label_smoothing
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def clip_loss(self, local_z, global_z, targets):
        z1 = F.normalize(local_z, dim=-1, eps=1e-6)
        z2 = F.normalize(global_z, dim=-1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
        return F.cross_entropy(scale * (z1 @ z2.t()), targets,
                               label_smoothing=self.label_smoothing)

    def regression_loss(self, pred, target):
        pred = F.normalize(pred, dim=-1, eps=1e-6)
        target = F.normalize(target.detach(), dim=-1, eps=1e-6)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    def forward(self, outputs, global_embeddings=None, targets=None):
        local = {k: outputs[k + '_shared'] for k in ('cxr', 'ecg', 'labs')}
        global_embeddings = local if global_embeddings is None else global_embeddings
        if targets is None:
            targets = torch.arange(local['cxr'].size(0), device=local['cxr'].device)
        pairs = {}
        for modality in ('ecg', 'labs'):
            pairs[modality] = 0.5 * (
                self.clip_loss(local['cxr'], global_embeddings[modality], targets)
                + self.clip_loss(local[modality], global_embeddings['cxr'], targets)
            )
        contrastive = 0.5 * (pairs['ecg'] + pairs['labs'])
        regression = self.regression_loss(outputs['predicted_cxr'], outputs['cxr_emb'])
        total = self.clip_weight * contrastive + self.regression_weight * regression
        return total, {'loss_total': total, 'loss_contrastive': contrastive,
                       'loss_regression': regression, 'clip_cxr_ecg': pairs['ecg'],
                       'clip_cxr_labs': pairs['labs']}


class ReversePACXDistillation(pl.LightningModule):
    def __init__(self, shared_dim=256, prediction_hidden_dim=768,
                 clip_weight=0.5, regression_weight=1.0, temperature=0.07,
                 label_smoothing=0.02, learning_rate=1e-4, weight_decay=0.05,
                 max_epochs=50, warmup_epochs=5):
        super().__init__()
        self.save_hyperparameters()
        self.cxr_encoder = make_cxr_teacher()
        self.ecg_encoder = ECGFounderEncoder()
        self.labs_encoder = LabEncoder()
        self.cxr_encoder.requires_grad_(False)
        self.cxr_encoder.eval()
        self.ecg_encoder.requires_grad_(True)
        self.labs_encoder.requires_grad_(True)
        self.cxr_to_shared = ProjectionHead(768, shared_dim)
        self.ecg_to_shared = ProjectionHead(1024, shared_dim)
        self.labs_to_shared = ProjectionHead(256, shared_dim)
        self.cxr_prediction = nn.Sequential(
            nn.Linear(1280, prediction_hidden_dim), nn.GELU(),
            nn.Linear(prediction_hidden_dim, 768),
        )
        self.loss_module = CLIPRegressionLoss(
            clip_weight, regression_weight, temperature, label_smoothing,
        )
        self.encoder_sources = {'initialization': 'random; software checks only'}

    def initialize_encoders(self, cxr_checkpoint, ecg_checkpoint, labs_checkpoint):
        self.encoder_sources = {
            'cxr': load_encoder_weights(self.cxr_encoder, cxr_checkpoint, 'cxr'),
            'ecg': load_encoder_weights(self.ecg_encoder, ecg_checkpoint, 'ecg'),
            'labs': load_encoder_weights(self.labs_encoder, labs_checkpoint, 'labs'),
        }
        self.cxr_encoder.requires_grad_(False)
        self.cxr_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.cxr_encoder.eval()  # Lightning calls train() again every epoch.
        return self

    def encode_students(self, ecg, labs, labs_mask):
        ecg_emb = self.ecg_encoder(ecg)
        labs_emb = self.labs_encoder(labs, labs_mask)
        fused = torch.cat([ecg_emb, labs_emb], dim=-1)
        return {'ecg_emb': ecg_emb, 'labs_emb': labs_emb, 'fused': fused,
                'predicted_cxr': self.cxr_prediction(fused)}

    def student_features(self, ecg, labs, labs_mask, representation='fused'):
        """Stage 3 interface: no CXR input or teacher computation required."""
        if representation not in ('fused', 'predicted_cxr'):
            raise ValueError('representation must be fused or predicted_cxr')
        return self.encode_students(ecg, labs, labs_mask)[representation]

    def forward(self, ecg, labs, labs_mask, cxr=None):
        outputs = self.encode_students(ecg, labs, labs_mask)
        if cxr is not None:
            with torch.no_grad():
                outputs['cxr_emb'] = self.cxr_encoder(cxr)
            # This head stays outside no_grad: it must learn contrastive alignment.
            outputs['cxr_shared'] = self.cxr_to_shared(outputs['cxr_emb'])
            outputs['ecg_shared'] = self.ecg_to_shared(outputs['ecg_emb'])
            outputs['labs_shared'] = self.labs_to_shared(outputs['labs_emb'])
        return outputs

    def _step(self, batch, stage):
        outputs = self(batch['ecg'], batch['labs'], batch['labs_mask'], batch['cxr'])
        local = {k: outputs[k + '_shared'] for k in ('cxr', 'ecg', 'labs')}
        size = batch['cxr'].size(0)
        if self.trainer.world_size > 1:
            # Equal local batch sizes are required by the original all-gather loss.
            sizes = self.all_gather(torch.tensor(size, device=self.device))
            if not torch.all(sizes == size):
                raise ValueError('Contrastive DDP requires equal per-rank batch sizes.')
            global_embeddings = {
                key: self.all_gather(value, sync_grads=(stage == 'train')).flatten(0, 1)
                for key, value in local.items()
            }
        else:
            global_embeddings = local
        targets = torch.arange(size, device=self.device) + self.global_rank * size
        loss, metrics = self.loss_module(outputs, global_embeddings, targets)
        self.log_dict({stage + '/' + k: v for k, v in metrics.items()},
                      on_step=(stage == 'train'), on_epoch=True,
                      batch_size=size, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['encoder_sources'] = self.encoder_sources
        checkpoint['pipeline_direction'] = 'ecg_labs_to_cxr'

    def on_load_checkpoint(self, checkpoint):
        self.encoder_sources = checkpoint.get('encoder_sources', {})
        self.cxr_encoder.requires_grad_(False)
        self.cxr_encoder.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
        )

        def schedule(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / max(1, self.hparams.warmup_epochs)
            progress = (epoch - self.hparams.warmup_epochs) / max(
                1, self.hparams.max_epochs - self.hparams.warmup_epochs)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


