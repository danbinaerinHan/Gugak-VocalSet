"""sigimsae Event Detection model

Models:
  EDTCN: Encoder-Decoder Temporal Convolutional Network
    - Periodic Padding (reflects chromagram frequency-axis periodicity, disabled for mel)
    - Dilated Convolutions (long-range temporal dependencies)
    - Encoder: 4 layers (downsample) / Decoder: 4 layers (upsample)
    - Time-distributed classification layer

  BaselineCRNN: 2D Conv + BiGRU + Time-distributed Dense

Input: (B, F, T) — F=frequency bins (chroma 120 / mel 128), T=time frames
Output: (B, C, T) — C=num_classes, per-frame logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Periodic Padding ──────────────────────────────────────

class PeriodicPad1d(nn.Module):
    """Periodic padding on the frequency axis.

    Since chromagram has a 12-tone periodic structure,
    the top/bottom are wrap-around to preserve periodicity.

    Input: (B, C, T) — C=frequency (chroma) axis
    """

    def __init__(self, pad_size: int):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x):
        if self.pad_size == 0:
            return x
        top = x[:, -self.pad_size:, :]     # last p rows → attach to top
        bottom = x[:, :self.pad_size, :]   # first p rows → attach to bottom
        return torch.cat([top, x, bottom], dim=1)


# ─── ED-TCN Encoder Layer ──────────────────────────────────

class EncoderLayer(nn.Module):
    """Conv1d + ReLU + MaxPool(2) + SpatialDropout."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 dilation=1, dropout=0.3):
        super().__init__()
        # Time-axis padding: 'same' padding for dilated conv
        self.pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad, dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout2d(dropout)  # Spatial dropout (channel-wise)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        # Spatial dropout: (B, C, T) → (B, C, 1, T) → dropout → squeeze
        x = self.dropout(x.unsqueeze(2)).squeeze(2)
        return x


# ─── ED-TCN Decoder Layer ──────────────────────────────────

class DecoderLayer(nn.Module):
    """Upsample(2) + Conv1d + ReLU + SpatialDropout."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 dilation=1, dropout=0.3):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation // 2
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad, dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x.unsqueeze(2)).squeeze(2)
        return x


# ─── ED-TCN Model ──────────────────────────────────────────

class EDTCN(nn.Module):
    """Encoder-Decoder TCN for frame-level event detection.

    Paper Section V-B implementation:
      Encoder: 4 layers [32, 64, 128, 256], dilation [1, 2, 3, 4]
      Decoder: 4 layers [256, 128, 64, 32], dilation [4, 3, 2, 1]
      + Periodic Padding on input (frequency axis, chromagram-only)
      + Time-distributed Dense classifier

    Input: (B, F, T) — F=frequency bins, T=time frames
    Output: (B, C, T) — C=num_classes, per-frame logits

    Note:
      periodic_pad leverages the 12-tone periodicity of chromagram.
      For non-periodic inputs such as mel spectrogram, set periodic_pad=0.
    """

    def __init__(
        self,
        n_freq: int = 120,
        n_classes: int = 10,
        encoder_filters: tuple = (32, 64, 128, 256),
        decoder_filters: tuple = (256, 128, 64, 64),
        kernel_size: int = 5,
        encoder_dilations: tuple = (1, 2, 3, 4),
        decoder_dilations: tuple = (4, 3, 2, 1),
        dropout: float = 0.3,
        periodic_pad: int = 2,
        # Legacy compatibility
        n_chroma: int = None,
    ):
        super().__init__()
        if n_chroma is not None:
            n_freq = n_chroma
        self.n_classes = n_classes
        self.periodic_pad = PeriodicPad1d(periodic_pad)
        input_channels = n_freq + 2 * periodic_pad

        # Encoder
        enc_layers = []
        in_ch = input_channels
        for filters, dilation in zip(encoder_filters, encoder_dilations):
            enc_layers.append(
                EncoderLayer(in_ch, filters, kernel_size, dilation, dropout)
            )
            in_ch = filters
        self.encoder = nn.ModuleList(enc_layers)

        # Decoder
        dec_layers = []
        in_ch = encoder_filters[-1]
        for filters, dilation in zip(decoder_filters, decoder_dilations):
            dec_layers.append(
                DecoderLayer(in_ch, filters, kernel_size, dilation, dropout)
            )
            in_ch = filters
        self.decoder = nn.ModuleList(dec_layers)

        # Time-distributed classifier
        self.classifier = nn.Conv1d(decoder_filters[-1], n_classes, kernel_size=1)

    def forward(self, x):
        """
        x: (B, F, T)  — F=chromagram bins, T=time frames
        returns: (B, C, T) — C=n_classes, per-frame logits
        """
        original_T = x.shape[2]

        # Periodic padding on frequency axis
        x = self.periodic_pad(x)

        # Encoder
        for layer in self.encoder:
            x = layer(x)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Match original time length (correct pooling/upsampling error)
        if x.shape[2] != original_T:
            x = F.interpolate(x, size=original_T, mode="linear", align_corners=False)

        # Classification
        logits = self.classifier(x)  # (B, C, T)
        return logits


# ─── Loss ──────────────────────────────────────────────────

class DontCareCrossEntropyLoss(nn.Module):
    """Cross-Entropy Loss excluding don't-care frames (paper Eq. 3).

    Frames with don't_care_label(-1) are completely excluded from loss computation.
    """

    def __init__(self, weight=None, dont_care_label=-1):
        super().__init__()
        self.dont_care_label = dont_care_label
        self.weight = weight

    def forward(self, logits, targets):
        """
        logits: (B, C, T)
        targets: (B, T) — values may include dont_care_label
        """
        B, C, T = logits.shape

        # flatten
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
        targets_flat = targets.reshape(-1)  # (B*T,)

        # don't-care mask
        mask = targets_flat != self.dont_care_label
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits_flat[mask]
        targets_valid = targets_flat[mask]

        return F.cross_entropy(logits_valid, targets_valid, weight=self.weight)


class DontCareFocalLoss(nn.Module):
    """Don't care aware Focal Loss.

    Class weight is applied independently of focal modulation:
      focal_term = (1 - pt)^gamma * (-log(pt))
      weighted  = class_weight[target] * focal_term
    This prevents class weight and focal from being multiplied together twice.
    (The previous implementation used F.cross_entropy(weight=...), which already
     multiplied the weight internally and then applied focal modulation on top,
     posing a risk of gradient explosion on minority classes.)
    """

    def __init__(self, gamma=2.0, weight=None, dont_care_label=-1):
        super().__init__()
        self.gamma = gamma
        self.dont_care_label = dont_care_label
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        B, C, T = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        mask = targets_flat != self.dont_care_label
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits_flat[mask]
        targets_valid = targets_flat[mask]

        # CE without class weight (weight is applied separately below)
        ce = F.cross_entropy(logits_valid, targets_valid,
                             reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-7, max=1.0)
        focal = ((1 - pt) ** self.gamma) * ce

        # Apply class weight separately, only once
        if self.weight is not None:
            w = self.weight[targets_valid]
            focal = focal * w

        return focal.mean()


# ─── MERT Head (uses pre-extracted features) ─────────────

class MERTHead(nn.Module):
    """Detection head placed on top of pre-extracted MERT hidden states.

    Uses the same head structure as MERTEventDetector (LayerNorm + BiGRU + classifier),
    but takes pre-extracted (B, D, T) features as input without a backbone.

    Input: (B, D, T) — pre-extracted MERT hidden state
           D = 768 (MERT-v1-95M, CultureMERT-95M) or 1024 (MERT-v1-330M)
    Output: (B, C, T) — per-frame logits
    """

    MERT_HOP = 320
    MERT_SR = 24000

    def __init__(
        self,
        n_classes: int = 9,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
        n_freq: int = None,  # feature dim D (event_dataset auto-extracts from shape and passes it)
        mert_dim: int = None,  # alias for n_freq (takes precedence when explicitly specified)
    ):
        super().__init__()
        self.n_classes = n_classes
        # dim priority: mert_dim > n_freq > 1024 (legacy default)
        D = mert_dim if mert_dim is not None else (n_freq if n_freq is not None else 1024)
        self.mert_dim = D

        self.head_norm = nn.LayerNorm(D)
        self.head_gru = nn.GRU(
            input_size=D,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head_classifier = nn.Linear(gru_hidden * 2, n_classes)

    def forward(self, x):
        """x: (B, D, T) → (B, C, T)"""
        x = x.permute(0, 2, 1)              # (B, T, D)
        x = self.head_norm(x)               # (B, T, D)
        x, _ = self.head_gru(x)             # (B, T, gru_hidden*2)
        x = self.head_dropout(x)
        logits = self.head_classifier(x)     # (B, T, n_classes)
        logits = logits.permute(0, 2, 1)     # (B, C, T)
        return logits

    @property
    def hop_sec(self):
        return self.MERT_HOP / self.MERT_SR


# ─── Model registry ────────────────────────────────────────

EVENT_MODEL_REGISTRY = {
    "EDTCN": EDTCN,
    "MERTHead": MERTHead,
}
