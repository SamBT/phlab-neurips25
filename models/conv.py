import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + res)

class ContrastiveEncoder1D(nn.Module):
    def __init__(self, input_channels=2, embedding_dim=64, stem_ks=31, hidden_dim=64, dropout_p=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=stem_ks, stride=1, padding=stem_ks//2, groups=input_channels),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # channel mixing
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.backbone = nn.Sequential(
            ResBlock1D(hidden_dim, 2*hidden_dim, dilation=2),
            nn.Dropout(dropout_p),
            ResBlock1D(2*hidden_dim, 4*hidden_dim, dilation=4),
            nn.Dropout(dropout_p),
            ResBlock1D(4*hidden_dim, 4*hidden_dim, dilation=8),
            nn.Dropout(dropout_p),
            ResBlock1D(4*hidden_dim, 4*hidden_dim, dilation=16),
            nn.Dropout(dropout_p)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2*hidden_dim, embedding_dim)
        )

    def forward(self, x):  # x: (B, 2, T)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.projector(x)

class ContrastiveEncoder2D(nn.Module):
    def __init__(self, input_channels=2, embedding_dim=64, stem_ks=31, hidden_dim=64, dropout_p=0.2):
        super().__init__()
        self.stem2d = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(input_channels, stem_ks), stride=(1, 1), padding=(0, stem_ks//2)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout_p)
        )
        self.backbone = nn.Sequential(
            ResBlock1D(hidden_dim, 2*hidden_dim, dilation=2),
            nn.Dropout(dropout_p),
            ResBlock1D(2*hidden_dim, 4*hidden_dim, dilation=4),
            nn.Dropout(dropout_p),
            ResBlock1D(4*hidden_dim, 4*hidden_dim, dilation=8),
            nn.Dropout(dropout_p),
            ResBlock1D(4*hidden_dim, 4*hidden_dim, dilation=16),
            nn.Dropout(dropout_p)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(4*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2*hidden_dim, embedding_dim)
        )

    def forward(self, x):  # x: (B, 2, T)
        x = x.unsqueeze(1)  # (B, 1, 2, T)
        x = self.stem2d(x)  # (B, 32, 1, T/2)
        x = x.squeeze(2)    # (B, 32, T/2)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.projector(x)