import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .ecg_transformer import ECGTransformer
from .labs_mlp import LabsMLP


class MainModel(nn.Module):
    def __init__(self, hidden_size=256, num_classes=14,
                 ecg_dropout=0.3, ecg_n_layers=1, ecg_n_head=8,
                 labs_dropout=0.3):
        super().__init__()
        self.num_classes = num_classes

        # ECG: 12 leads at each of the 5,000 time positions.
        self.ecg_model = ECGTransformer(
            input_size=12, num_classes=num_classes,
            d_model=hidden_size, n_head=ecg_n_head,
            n_layers_feat=1, n_layers_shared=ecg_n_layers,
            n_layers_distinct=ecg_n_layers,
            dropout=ecg_dropout, max_len=5000,
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
        # The forward pass will stack four tokens: ECG distinct, labs distinct,
        # fused shared, and CXR distinct.
        self.attn_proj = nn.Linear(
            hidden_size, (2 + num_classes) * hidden_size
        )
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, ecg, labs, cxr, modality_mask):
        # The DataLoader mask order is [CXR, ECG, labs]; True means available.
        modality_mask = modality_mask.to(device=ecg.device, dtype=torch.bool)
        cxr_available = modality_mask[:, 0]

        feat_ecg_shared, feat_ecg_distinct, pred_ecg = self.ecg_model(ecg)
        feat_labs_shared, feat_labs_distinct, pred_labs = self.labs_model(labs)

        # Encode available CXRs only, so placeholders do not affect BatchNorm.
        feat_cxr_shared = torch.zeros_like(feat_ecg_shared)
        feat_cxr_distinct = torch.zeros_like(feat_ecg_distinct)
        if cxr_available.any():
            feat_cxr = self.cxr_model_feat(cxr[cxr_available])
            feat_cxr_shared[cxr_available] = self.cxr_model_shared(feat_cxr)
            feat_cxr_distinct[cxr_available] = self.cxr_model_distinct(feat_cxr)

        # Auxiliary CXR predictions are supervised only where CXR is available.
        pred_cxr = self.cxr_model_linear(feat_cxr_distinct).sigmoid()

        feat_ecg_shared = self.shared_project(feat_ecg_shared)
        feat_labs_shared = self.shared_project(feat_labs_shared)
        feat_cxr_shared = self.shared_project(feat_cxr_shared)

        #DrFuse's fusion to three modalities:
        # logit(mean(sigmoid(shared features))) over available modalities.
        # Log-space calculations keep this stable without clipping probabilities.
        shared_features = torch.stack(
            [feat_cxr_shared, feat_ecg_shared, feat_labs_shared], dim=1
        )
        shared_mask = modality_mask.unsqueeze(-1)
        log_positive = F.logsigmoid(shared_features).masked_fill(
            ~shared_mask, float('-inf')
        )
        log_negative = F.logsigmoid(-shared_features).masked_fill(
            ~shared_mask, float('-inf')
        )
        feat_avg_shared = (
            torch.logsumexp(log_positive, dim=1)
            - torch.logsumexp(log_negative, dim=1)
        )
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # Disease-wise attention over the three distinct and one shared token.
        attn_input = torch.stack(
            [feat_ecg_distinct, feat_labs_distinct,
             feat_avg_shared, feat_cxr_distinct], dim=1
        )
        attn_mask = torch.stack(
            [modality_mask[:, 1], modality_mask[:, 2],
             modality_mask.any(dim=1), cxr_available], dim=1
        )
        qkvs = self.attn_proj(attn_input)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)

        # Average queries from available tokens only.
        query_mask = attn_mask.unsqueeze(-1).to(q.dtype)
        q_mean = (q * query_mask).sum(dim=1) / query_mask.sum(dim=1)

        ks = torch.stack(k, dim=1)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])
        attn_logits = attn_logits.masked_fill(
            ~attn_mask.unsqueeze(1), float('-inf')
        )
        attn_weights = F.softmax(attn_logits, dim=-1)

        # One feature vector per disease, followed by its corresponding score.
        feat_final = torch.matmul(attn_weights, v)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        return {
            'feat_ecg_shared': feat_ecg_shared,
            'feat_labs_shared': feat_labs_shared,
            'feat_cxr_shared': feat_cxr_shared,
            'feat_ecg_distinct': feat_ecg_distinct,
            'feat_labs_distinct': feat_labs_distinct,
            'feat_cxr_distinct': feat_cxr_distinct,
            'feat_avg_shared': feat_avg_shared,
            'feat_final': feat_final,
            'pred_final': pred_final,
            'pred_shared': pred_shared,
            'pred_ecg': pred_ecg,
            'pred_labs': pred_labs,
            'pred_cxr': pred_cxr,
            'attn_weights': attn_weights,
            'modality_mask': modality_mask,
        }
