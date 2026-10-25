from torch import nn


class LabsMLP(nn.Module):
    def __init__(self, input_size=100, num_classes=5,
                 d_model=256, dropout=0.3):
        super().__init__()
        self.d_model = d_model

        # Encode the lab values together with their observation indicators.
        self.model_feat = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared representation branch.
        self.model_shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Distinct representation branch.
        self.model_distinct = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Auxiliary disease prediction from the distinct representation.
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 100), already concatenated by the DataLoader.
        feat = self.model_feat(x)
        rep_shared = self.model_shared(feat)
        rep_distinct = self.model_distinct(feat)
        pred_distinct = self.fc_distinct(rep_distinct).sigmoid()

        return rep_shared, rep_distinct, pred_distinct
