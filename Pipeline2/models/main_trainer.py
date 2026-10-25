import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_auroc,
)

import lightning.pytorch as pl

from .main_model import MainModel


class JSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, cxr, ecg, labs, modality_mask):
        # Sigmoid features; mask order: [CXR, ECG, labs].
        p = torch.stack([cxr, ecg, labs], dim=1)
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1 - eps)
        mask = modality_mask.to(p)
        count = mask.sum(dim=1, keepdim=True)
        weights = mask / count.clamp_min(1)
        p = torch.stack([p, 1 - p], dim=-1)
        m = (p * weights[:, :, None, None]).sum(dim=1, keepdim=True)
        divergence = self.kl(m.clamp_min(eps).log(), p.log()).sum(-1).mean(-1)
        loss = (divergence * weights).sum(dim=1)
        valid = (count.squeeze(1) >= 2).to(p.dtype)
        return (loss * valid).sum() / valid.sum().clamp_min(1)


class MainTrainer(pl.LightningModule):
    def __init__(self, args, label_names):
        super().__init__()
        self.model = MainModel(
            hidden_size=args.hidden_size,
            num_classes=len(label_names),
            ecg_dropout=args.dropout,
            ecg_n_head=args.ecg_n_head,
            ecg_n_layers=args.ecg_n_layers,
            labs_dropout=args.dropout,
        )

        self.save_hyperparameters(args)  # args goes to self.hparams.

        self.pred_criterion = nn.BCELoss(reduction="none")
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.jsd = JSD()

        self.val_preds = {k: [] for k in ["final", "ecg", "labs", "cxr"]}
        self.val_labels = []
        self.val_label_masks = []
        self.val_modality_masks = []

        self.test_preds = []
        self.test_labels = []
        self.test_label_masks = []
        self.test_modality_masks = []
        self.test_feats = {k: [] for k in [
            "feat_ecg_shared", "feat_ecg_distinct",
            "feat_labs_shared", "feat_labs_distinct",
            "feat_cxr_shared", "feat_cxr_distinct",
        ]}
        self.test_attns = []

        self.label_names = label_names

    def _compute_masked_pred_loss(self, input, target, mask):
        mask = mask.to(input)
        loss = self.pred_criterion(input, target)
        return (loss * mask).sum() / mask.sum().clamp_min(1)

    def _masked_abs_cos_sim(self, x, y, mask):
        mask = mask.to(x)
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / mask.sum().clamp_min(1)

    def _masked_cos_sim(self, x, y, mask):
        mask = mask.to(x)
        return (self.alignment_cos_sim(x, y) * mask).sum() / mask.sum().clamp_min(1)

    def _masked_mse(self, x, y, mask):
        mask = mask.to(x)
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / mask.sum().clamp_min(1)

    def _disentangle_loss_jsd(self, model_output, modality_mask, log=True, mode="train"):
        # Availability order: CXR, ECG, labs.
        losses = {}
        for i, name in enumerate(["cxr", "ecg", "labs"]):
            losses[name] = self._masked_abs_cos_sim(
                model_output[f"feat_{name}_shared"],
                model_output[f"feat_{name}_distinct"],
                modality_mask[:, i],
            )

        jsd = self.jsd(
            model_output["feat_cxr_shared"].sigmoid(),
            model_output["feat_ecg_shared"].sigmoid(),
            model_output["feat_labs_shared"].sigmoid(),
            modality_mask,
        )
        loss_disentanglement = (
            self.hparams.lambda_disentangle_shared * jsd
            + self.hparams.lambda_disentangle_ecg * losses["ecg"]
            + self.hparams.lambda_disentangle_labs * losses["labs"]
            + self.hparams.lambda_disentangle_cxr * losses["cxr"]
        )

        if log:
            self.log_dict({
                f"disentangle_{mode}/ecg_distinct": losses["ecg"].detach(),
                f"disentangle_{mode}/labs_distinct": losses["labs"].detach(),
                f"disentangle_{mode}/cxr_distinct": losses["cxr"].detach(),
                f"disentangle_{mode}/shared_jsd": jsd.detach(),
            }, on_epoch=True, on_step=False, batch_size=modality_mask.shape[0])

        return loss_disentanglement

    def _compute_prediction_losses(self, model_output, y_gt, label_mask,
                                   modality_mask, log=True, mode="train"):
        label_mask = label_mask.bool()
        cxr_mask = modality_mask[:, 0:1].bool()
        ecg_mask = modality_mask[:, 1:2].bool()
        labs_mask = modality_mask[:, 2:3].bool()
        shared_mask = modality_mask.bool().any(dim=1, keepdim=True)

        loss_pred_final = self._compute_masked_pred_loss(
            model_output["pred_final"], y_gt, label_mask & shared_mask)
        loss_pred_ecg = self._compute_masked_pred_loss(
            model_output["pred_ecg"], y_gt, label_mask & ecg_mask)
        loss_pred_labs = self._compute_masked_pred_loss(
            model_output["pred_labs"], y_gt, label_mask & labs_mask)
        loss_pred_cxr = self._compute_masked_pred_loss(
            model_output["pred_cxr"], y_gt, label_mask & cxr_mask)
        loss_pred_shared = self._compute_masked_pred_loss(
            model_output["pred_shared"], y_gt, label_mask & shared_mask)

        if log:
            self.log_dict({
                f"{mode}_loss/pred_final": loss_pred_final.detach(),
                f"{mode}_loss/pred_shared": loss_pred_shared.detach(),
                f"{mode}_loss/pred_ecg": loss_pred_ecg.detach(),
                f"{mode}_loss/pred_labs": loss_pred_labs.detach(),
                f"{mode}_loss/pred_cxr": loss_pred_cxr.detach(),
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return (loss_pred_final, loss_pred_ecg, loss_pred_labs,
                loss_pred_cxr, loss_pred_shared)

    def _compute_and_log_loss(self, model_output, y_gt, label_mask,
                              modality_mask, log=True, mode="train"):
        prediction_losses = self._compute_prediction_losses(
            model_output, y_gt, label_mask, modality_mask, log, mode)
        loss_pred_final, loss_pred_ecg, loss_pred_labs, loss_pred_cxr, loss_pred_shared = prediction_losses

        loss_prediction = (
            self.hparams.lambda_pred_shared * loss_pred_shared
            + self.hparams.lambda_pred_ecg * loss_pred_ecg
            + self.hparams.lambda_pred_labs * loss_pred_labs
            + self.hparams.lambda_pred_cxr * loss_pred_cxr
        )
        loss_prediction = loss_pred_final + loss_prediction
        loss_disentanglement = self._disentangle_loss_jsd(
            model_output, modality_mask, log, mode)
        loss_total = loss_prediction + loss_disentanglement

        # Auxiliary prediction losses define the preferred attention ordering.
        # Detach them so ranking trains attention, not the auxiliary predictions.
        raw_pred_loss_ecg = F.binary_cross_entropy(
            model_output["pred_ecg"].detach(), y_gt, reduction="none")
        raw_pred_loss_labs = F.binary_cross_entropy(
            model_output["pred_labs"].detach(), y_gt, reduction="none")
        raw_pred_loss_cxr = F.binary_cross_entropy(
            model_output["pred_cxr"].detach(), y_gt, reduction="none")
        raw_pred_loss_shared = F.binary_cross_entropy(
            model_output["pred_shared"].detach(), y_gt, reduction="none")

        label_mask = label_mask.bool()
        cxr_mask = modality_mask[:, 0:1].bool()
        ecg_mask = modality_mask[:, 1:2].bool()
        labs_mask = modality_mask[:, 2:3].bool()
        shared_mask = modality_mask.bool().any(dim=1, keepdim=True)

        # Attention token order: ECG distinct, labs distinct, shared, CXR distinct.
        attn_weights = model_output["attn_weights"]
        attn_ecg = attn_weights[:, :, 0]
        attn_labs = attn_weights[:, :, 1]
        attn_shared = attn_weights[:, :, 2]
        attn_cxr = attn_weights[:, :, 3]

        # Labs versus ECG.
        labs_overweights_ecg = 2 * (raw_pred_loss_labs < raw_pred_loss_ecg).to(y_gt.dtype) - 1
        mask1 = label_mask & labs_mask & ecg_mask
        mask1 = mask1 & (raw_pred_loss_labs != raw_pred_loss_ecg)
        loss_attn1 = mask1 * F.margin_ranking_loss(
            attn_labs, attn_ecg, labs_overweights_ecg, reduction="none")
        loss_attn1 = loss_attn1.sum() / (loss_attn1 > 0).sum().clamp_min(1)

        # Shared versus ECG.
        shared_overweights_ecg = 2 * (raw_pred_loss_shared < raw_pred_loss_ecg).to(y_gt.dtype) - 1
        mask2 = label_mask & shared_mask & ecg_mask
        mask2 = mask2 & (raw_pred_loss_shared != raw_pred_loss_ecg)
        loss_attn2 = mask2 * F.margin_ranking_loss(
            attn_shared, attn_ecg, shared_overweights_ecg, reduction="none")
        loss_attn2 = loss_attn2.sum() / (loss_attn2 > 0).sum().clamp_min(1)

        #CXR versus ECG.
        cxr_overweights_ecg = 2 * (raw_pred_loss_cxr < raw_pred_loss_ecg).to(y_gt.dtype) - 1
        mask3 = label_mask & cxr_mask & ecg_mask
        mask3 = mask3 & (raw_pred_loss_cxr != raw_pred_loss_ecg)
        loss_attn3 = mask3 * F.margin_ranking_loss(
            attn_cxr, attn_ecg, cxr_overweights_ecg, reduction="none")
        loss_attn3 = loss_attn3.sum() / (loss_attn3 > 0).sum().clamp_min(1)

        #Shared versus labs.
        shared_overweights_labs = 2 * (raw_pred_loss_shared < raw_pred_loss_labs).to(y_gt.dtype) - 1
        mask4 = label_mask & shared_mask & labs_mask
        mask4 = mask4 & (raw_pred_loss_shared != raw_pred_loss_labs)
        loss_attn4 = mask4 * F.margin_ranking_loss(
            attn_shared, attn_labs, shared_overweights_labs, reduction="none")
        loss_attn4 = loss_attn4.sum() / (loss_attn4 > 0).sum().clamp_min(1)

        # 5. CXR versus labs.
        cxr_overweights_labs = 2 * (raw_pred_loss_cxr < raw_pred_loss_labs).to(y_gt.dtype) - 1
        mask5 = label_mask & cxr_mask & labs_mask
        mask5 = mask5 & (raw_pred_loss_cxr != raw_pred_loss_labs)
        loss_attn5 = mask5 * F.margin_ranking_loss(
            attn_cxr, attn_labs, cxr_overweights_labs, reduction="none")
        loss_attn5 = loss_attn5.sum() / (loss_attn5 > 0).sum().clamp_min(1)

        #Shared versus CXR.
        shared_overweights_cxr = 2 * (raw_pred_loss_shared < raw_pred_loss_cxr).to(y_gt.dtype) - 1
        mask6 = label_mask & shared_mask & cxr_mask
        mask6 = mask6 & (raw_pred_loss_shared != raw_pred_loss_cxr)
        loss_attn6 = mask6 * F.margin_ranking_loss(
            attn_shared, attn_cxr, shared_overweights_cxr, reduction="none")
        loss_attn6 = loss_attn6.sum() / (loss_attn6 > 0).sum().clamp_min(1)

        loss_attn_ranking = (
            loss_attn1 + loss_attn2 + loss_attn3
            + loss_attn4 + loss_attn5 + loss_attn6
        ) / 6
        loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking

        if log:
            self.log_dict({
                f"{mode}_loss/attn_aux": loss_attn_ranking.detach(),
                f"{mode}_loss/prediction": loss_prediction.detach(),
                f"{mode}_loss/total": loss_total.detach(),
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self, batch):
        ecg = batch["ecg"].float().to(self.device)
        labs = batch["labs"].float().to(self.device)
        cxr = batch["cxr"].float().to(self.device)
        y = batch["labels"].float().to(self.device)
        label_mask = batch["label_mask"].bool().to(self.device)
        modality_mask = batch["modality_mask"].bool().to(self.device)
        return ecg, labs, cxr, y, label_mask, modality_mask

    def training_step(self, batch, batch_idx):
        ecg, labs, cxr, y, label_mask, modality_mask = self._get_batch_data(batch)
        # CXR hiding is already handled by the Dataset; use its mask directly.
        out = self.model(ecg, labs, cxr, modality_mask)
        return self._compute_and_log_loss(
            out, y_gt=y, label_mask=label_mask, modality_mask=modality_mask)

    def validation_step(self, batch, batch_idx):
        ecg, labs, cxr, y, label_mask, modality_mask = self._get_batch_data(batch)
        out = self.model(ecg, labs, cxr, modality_mask)
        loss = self._compute_and_log_loss(
            out, y_gt=y, label_mask=label_mask,
            modality_mask=modality_mask, mode="val")

        for name in self.val_preds:
            self.val_preds[name].append(out[f"pred_{name}"].detach().cpu())
        self.val_labels.append(y.detach().cpu())
        self.val_label_masks.append(label_mask.cpu())
        self.val_modality_masks.append(modality_mask.cpu())
        return loss

    def _compute_masked_metrics(self, preds, y_gt, label_mask):
        # Each disease has a different set of known labels.
        mlaps = preds.new_full((len(self.label_names),), float("nan"))
        mlroc = preds.new_full((len(self.label_names),), float("nan"))
        for i in range(len(self.label_names)):
            valid = label_mask[:, i].bool()
            target = y_gt[valid, i].long()
            scores = preds[valid, i]
            # AP needs positive examples; AUROC needs both classes.
            if target.numel() > 0 and target.sum() > 0:
                mlaps[i] = binary_average_precision(scores, target)
            if target.unique().numel() == 2:
                mlroc[i] = binary_auroc(scores, target)
        return mlaps, mlroc

    def on_validation_epoch_end(self):
        if not self.val_labels:
            return
        y_gt = torch.cat(self.val_labels, dim=0)
        label_mask = torch.cat(self.val_label_masks, dim=0)
        modality_mask = torch.cat(self.val_modality_masks, dim=0)

        for name in ["final", "ecg", "labs", "cxr"]:
            preds = torch.cat(self.val_preds[name], dim=0)
            mask = label_mask.clone()
            if name == "cxr":
                mask = mask & modality_mask[:, 0:1]
            elif name == "ecg":
                mask = mask & modality_mask[:, 1:2]
            elif name == "labs":
                mask = mask & modality_mask[:, 2:3]
            mlaps, mlroc = self._compute_masked_metrics(preds, y_gt, mask)

            if name == "final":
                self.log("Val_PRAUC", mlaps.nanmean(), logger=False, prog_bar=True)
                self.log("Val_AUROC", mlroc.nanmean(), logger=False, prog_bar=True)

            log_dict = {
                f"val_PRAUC_avg_over_dxs/{name}": mlaps.nanmean(),
                f"val_AUROC_avg_over_dxs/{name}": mlroc.nanmean(),
            }
            for i, label in enumerate(self.label_names):
                log_dict[f"val_PRAUC_per_dx_{name}/{label}"] = mlaps[i]
                log_dict[f"val_AUROC_per_dx_{name}/{label}"] = mlroc[i]
            self.log_dict(log_dict)

        for k in self.val_preds:
            self.val_preds[k].clear()
        self.val_labels.clear()
        self.val_label_masks.clear()
        self.val_modality_masks.clear()

    def test_step(self, batch, batch_idx):
        ecg, labs, cxr, y, label_mask, modality_mask = self._get_batch_data(batch)
        out = self.model(ecg, labs, cxr, modality_mask)

        self.test_preds.append(out["pred_final"].detach().cpu())
        self.test_labels.append(y.detach().cpu())
        self.test_label_masks.append(label_mask.cpu())
        self.test_modality_masks.append(modality_mask.cpu())
        self.test_attns.append(out["attn_weights"].detach().cpu())
        for k in self.test_feats:
            self.test_feats[k].append(out[k].detach().cpu())

    def on_test_epoch_end(self):
        if not self.test_labels:
            return
        y_gt = torch.cat(self.test_labels, dim=0)
        preds = torch.cat(self.test_preds, dim=0)
        label_mask = torch.cat(self.test_label_masks, dim=0)
        modality_mask = torch.cat(self.test_modality_masks, dim=0)
        attn_weights = torch.cat(self.test_attns, dim=0)
        mlaps, mlroc = self._compute_masked_metrics(preds, y_gt, label_mask)
        self.test_results = {
            "y_gt": y_gt,
            "preds": preds,
            "label_mask": label_mask,
            "modality_mask": modality_mask,
            "mlaps": mlaps,
            "mlroc": mlroc,
            "prauc": mlaps.nanmean().item(),
            "auroc": mlroc.nanmean().item(),
            "attn_weight": attn_weights,
            "label_names": self.label_names,
        }
        for k in self.test_feats:
            self.test_results[k] = torch.cat(self.test_feats[k], dim=0)
            self.test_feats[k].clear()
        self.test_labels.clear()
        self.test_preds.clear()
        self.test_label_masks.clear()
        self.test_modality_masks.clear()
        self.test_attns.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.wd)
        return optimizer
