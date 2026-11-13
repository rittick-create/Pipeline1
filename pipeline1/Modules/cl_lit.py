"""PACX-style linear probing adapted to 14 masked clinical labels."""

import torch
from torch.nn import functional as F
import lightning.pytorch as pl
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_auroc,
)
from Models.prediction import StudentClassifier
from Data.targets import TARGETS
from Utils.utils import build_param_groups


def clinical_metrics(predictions, labels, masks):
    ap = torch.full((len(TARGETS),), float("nan"))
    roc = ap.clone()
    for i in range(len(TARGETS)):
        valid = masks[:, i].bool()
        y = labels[valid, i].long()
        p = predictions[valid, i]
        if y.numel() and y.sum() > 0:
            ap[i] = binary_average_precision(p, y)
        if y.unique().numel() == 2:
            roc[i] = binary_auroc(p, y)
    return ap, roc


class ClassificationLightningModule(pl.LightningModule):
    def __init__(self, lr=3e-3, weight_decay=0.01, max_epochs=40):
        super().__init__()
        self.save_hyperparameters()
        self.model = StudentClassifier(len(TARGETS))
        self.val_rows = []
        self.test_rows = []
        self.test_results = None

    def _step(self, batch, mode):
        logits = self.model(batch["ecg"], batch["labs"], batch["labs_mask"])
        mask = batch["label_mask"].to(logits)
        loss = (
            F.binary_cross_entropy_with_logits(
                logits, batch["labels"], reduction="none"
            )
            * mask
        ).sum() / mask.sum().clamp_min(1)
        self.log(
            f"{mode}/loss", loss, on_step=False, on_epoch=True, batch_size=len(logits)
        )
        return loss, {
            "predictions": logits.sigmoid().detach().cpu(),
            "labels": batch["labels"].detach().cpu(),
            "label_mask": batch["label_mask"].cpu(),
            "hadm_ids": batch["hadm_id"].cpu(),
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        loss, row = self._step(batch, "val")
        self.val_rows.append(row)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_rows:
            return
        rows = {k: torch.cat([r[k] for r in self.val_rows]) for k in self.val_rows[0]}
        ap, roc = clinical_metrics(
            rows["predictions"], rows["labels"], rows["label_mask"]
        )
        self.log("val/auroc_macro", roc.nanmean().to(self.device), prog_bar=True)
        self.log("val/ap_macro", ap.nanmean().to(self.device), prog_bar=True)
        for i, name in enumerate(TARGETS):
            self.log(f"val/auroc/{name}", roc[i].to(self.device))
            self.log(f"val/ap/{name}", ap[i].to(self.device))
        self.val_rows.clear()

    def on_test_epoch_start(self):
        self.test_rows.clear()

    def test_step(self, batch, batch_idx):
        _, row = self._step(batch, "test")
        self.test_rows.append(row)

    def on_test_epoch_end(self):
        rows = {k: torch.cat([r[k] for r in self.test_rows]) for k in self.test_rows[0]}
        ap, roc = clinical_metrics(
            rows["predictions"], rows["labels"], rows["label_mask"]
        )
        self.test_results = {
            **rows,
            "ap_per_finding": ap,
            "auroc_per_finding": roc,
            "macro_ap": float(ap.nanmean()),
            "macro_auroc": float(roc.nanmean()),
            "targets": TARGETS,
        }
        self.log("test/ap_macro", ap.nanmean().to(self.device))
        self.log("test/auroc_macro", roc.nanmean().to(self.device))
        self.test_rows.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            build_param_groups(self, self.hparams.weight_decay), lr=self.hparams.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
