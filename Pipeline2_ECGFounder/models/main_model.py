import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .ecg_founder import ECGFounderEncoder
from .labs_mlp import LabsMLP


class MainModel(nn.Module):
    def __init__(self, hidden_size=256, num_classes=14,
                 ecg_dropout=0.3, ecg_checkpoint=None, freeze_ecg=False,
                 labs_dropout=0.3):
        super().__init__()
        self.num_classes = num_classes

        # ECGFounder produces a 1,024-dimensional pooled embedding.
        self.ecg_model = ECGFounderEncoder(
            hidden_size=hidden_size, num_classes=num_classes,
            dropout=ecg_dropout, checkpoint_path=ecg_checkpoint,
            freeze_backbone=freeze_ecg,
        )

        # Labs: 50 percentile values followed by 50 observation indicators.
        self.labs_model = LabsMLP(
            input_size=100, num_classes=num_classes,
            d_model=hidden_size, dropout=labs_dropout,
        )

        # Common CXR feature extractor, following the original DrFuse.
        resnet = resnet50()
        self.cxr_model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        # CXR shared branch: pooled ResNet features -> hidden_size.
        resnet = resnet50()
        self.cxr_model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, hidden_size),
        )

        # CXR distinct branch, with its own independent parameters.
        resnet = resnet50()
        self.cxr_model_distinct = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, hidden_size),
        )

        # Apply the same projection to each modality's shared representation.
        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ECG and labs already contain their own auxiliary prediction heads.
        self.cxr_model_linear = nn.Linear(hidden_size, num_classes)
        self.fuse_model_shared = nn.Linear(hidden_size, num_classes)

        # Disease-wise attention: one query, one value, and one key per disease.
        # Complete inputs give four tokens; reduced tests omit unavailable tokens.
        self.attn_proj = nn.Linear(
            hidden_size, (2 + num_classes) * hidden_size
        )
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, ecg=None, labs=None, cxr=None):
        """Encode present modalities only; absent inputs have no feature slots."""
        inputs = {name: value for name, value in
                  [("cxr", cxr), ("ecg", ecg), ("labs", labs)] if value is not None}
        if not inputs:
            raise ValueError("At least one modality is required.")
        reference = next(iter(inputs.values()))
        if any(x.shape[0] != reference.shape[0] or x.device != reference.device
               for x in inputs.values()):
            raise ValueError("Available inputs must share a batch size and device.")

        shared, distinct, predictions = {}, {}, {}
        if ecg is not None:
            shared["ecg"], distinct["ecg"], predictions["pred_ecg"] = self.ecg_model(ecg)
        if labs is not None:
            shared["labs"], distinct["labs"], predictions["pred_labs"] = self.labs_model(labs)
        if cxr is not None:
            features = self.cxr_model_feat(cxr)
            shared["cxr"] = self.cxr_model_shared(features)
            distinct["cxr"] = self.cxr_model_distinct(features)
            predictions["pred_cxr"] = self.cxr_model_linear(distinct["cxr"]).sigmoid()

        shared = {name: self.shared_project(value) for name, value in shared.items()}
        shared_features = torch.stack(
            [shared[name] for name in ("cxr", "ecg", "labs") if name in shared], dim=1
        )
        # Logit pooling over present shared vectors. For one input it is identity.
        if len(shared) == 1:
            feat_avg_shared = shared_features[:, 0]
        else:
            feat_avg_shared = (
                torch.logsumexp(F.logsigmoid(shared_features), dim=1)
                - torch.logsumexp(F.logsigmoid(-shared_features), dim=1)
            )
        predictions["pred_shared"] = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Canonical order, omitting unavailable tokens entirely.
        tokens = {f"{name}_distinct": distinct[name]
                  for name in ("ecg", "labs") if name in distinct}
        tokens["shared"] = feat_avg_shared
        if "cxr" in distinct:
            tokens["cxr_distinct"] = distinct["cxr"]
        attn_input = torch.stack(list(tokens.values()), dim=1)
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)
        q_mean = q.mean(dim=1)
        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_weights = F.softmax(attn_logits / math.sqrt(q.shape[-1]), dim=-1)

        feat_final = torch.matmul(attn_weights, v)
        scores = self.final_pred_fc(feat_final)
        predictions["pred_final"] = torch.diagonal(scores, dim1=1, dim2=2).sigmoid()

        # Availability is metadata for losses/results, never a padded input.
        modality_mask = torch.tensor(
            [name in inputs for name in ("cxr", "ecg", "labs")],
            dtype=torch.bool, device=reference.device,
        ).unsqueeze(0).expand(reference.shape[0], -1)
        return {
            **{f"feat_{name}_shared": value for name, value in shared.items()},
            **{f"feat_{name}_distinct": value for name, value in distinct.items()},
            **predictions,
            "feat_avg_shared": feat_avg_shared,
            "feat_final": feat_final,
            "attn_weights": attn_weights,
            "attn_token_names": tuple(tokens),
            "modalities": tuple(inputs),
            "modality_mask": modality_mask,
        }
