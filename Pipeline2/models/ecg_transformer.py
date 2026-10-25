import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # One trainable embedding vector per time position.
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.1, 0.1)

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ECGTransformer(nn.Module):
    def __init__(self, input_size=12, num_classes=14,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Project the 12 lead values at each time point into an embedding.
        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(
            d_model, dropout=0, max_len=max_len
        )

        # Initial Transformer feature extractor.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            batch_first=True, dropout=dropout
        )
        self.model_feat = nn.TransformerEncoder(
            layer, num_layers=n_layers_feat
        )

        # Shared representation branch.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            batch_first=True, dropout=dropout
        )
        self.model_shared = nn.TransformerEncoder(
            layer, num_layers=n_layers_shared
        )

        # Distinct representation branch with an auxiliary prediction head.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            batch_first=True, dropout=dropout
        )
        self.model_distinct = nn.TransformerEncoder(
            layer, num_layers=n_layers_distinct
        )
        
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # ECG input from the loader: (B, 1, 5000, 12).
        x = x.squeeze(1)

        x = self.emb(x)
        x = self.pos_encoder(x)
        feat = self.model_feat(x)
        h_shared = self.model_shared(feat)
        h_distinct = self.model_distinct(feat)

        # Average the 5,000 time positions in each branch.
        rep_shared = h_shared.mean(dim=1)
        rep_distinct = h_distinct.mean(dim=1)

        pred_distinct = self.fc_distinct(rep_distinct).sigmoid()
        return rep_shared, rep_distinct, pred_distinct
