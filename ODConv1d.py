from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
#  Attention1d  —  four-head shared encoder
# ════════════════════════════════════════════════════════════════════════════

class Attention1d(nn.Module):
    """
    Shared attention encoder that produces four per-sample scalings.

    Heads:
        αci  channel  [B, C_in,  1]       skipped when C_in=1 (no cross-channel info)
        αfi  filter   [B, C_out, 1]       skipped for depth-wise conv
        αsi  spatial  [B, 1, 1, k]        skipped when k=1
        αwi  kernel   [B, n, 1, 1, 1]     skipped when n=1

    Architecture (shared for all layers regardless of C_in):
        x [B, C_in, L]
          → AdaptiveAvgPool1d(1)                   [B, C_in,     1]
          → Conv1d(C_in → attn_ch, k=1)            [B, attn_ch,  1]  bottleneck
          → BN + ReLU
          → channel_fc  → sigmoid                  [B, C_in,  1]   (if C_in > 1)
          → filter_fc   → sigmoid                  [B, C_out, 1]
          → spatial_fc  → sigmoid                  [B, 1, 1,  k]   (if k > 1)
          → kernel_fc   → softmax                  [B, n, 1, 1, 1] (if n > 1)
    """

    def __init__(
        self,
        in_planes:   int,
        out_planes:  int,
        kernel_size: int,
        groups:      int   = 1,
        reduction:   float = 0.0625,
        kernel_num:  int   = 4,
        min_channel: int   = 16,
    ) -> None:
        super().__init__()

        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size  = kernel_size
        self.kernel_num   = kernel_num
        self.temperature  = 1.0

        # ── Shared encoder ───────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Conv1d(in_planes, attention_channel, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm1d(attention_channel)
        self.relu    = nn.ReLU(inplace=True)

        # ── Head 1: channel attention ────────────────────────────────────────
        # Skipped for C_in=1: only one channel — no cross-channel selection possible
        if in_planes == 1:
            self.func_channel: Callable = self._skip
        else:
            self.channel_fc   = nn.Conv1d(attention_channel, in_planes,  kernel_size=1, bias=True)
            self.func_channel = self._get_channel_attention

        # ── Head 2: filter attention ─────────────────────────────────────────
        # Skipped for depth-wise conv (in == groups == out)
        if in_planes == groups and in_planes == out_planes:
            self.func_filter: Callable = self._skip
        else:
            self.filter_fc   = nn.Conv1d(attention_channel, out_planes, kernel_size=1, bias=True)
            self.func_filter = self._get_filter_attention

        # ── Head 3: temporal kernel-position attention ───────────────────────
        # Skipped for point-wise k=1
        if kernel_size == 1:
            self.func_spatial: Callable = self._skip
        else:
            self.spatial_fc   = nn.Conv1d(attention_channel, kernel_size, kernel_size=1, bias=True)
            self.func_spatial = self._get_spatial_attention

        # ── Head 4: multi-kernel bank attention ──────────────────────────────
        # Skipped when n=1
        if kernel_num == 1:
            self.func_kernel: Callable = self._skip
        else:
            self.kernel_fc   = nn.Conv1d(attention_channel, kernel_num,  kernel_size=1, bias=True)
            self.func_kernel = self._get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def update_temperature(self, temperature: float) -> None:
        """Anneal sigmoid/softmax sharpness. Call once per epoch."""
        self.temperature = temperature

    @staticmethod
    def _skip(_: torch.Tensor) -> float:
        return 1.0

    def _get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.channel_fc(x) / self.temperature)   # [B, C_in,  1]

    def _get_filter_attention(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.filter_fc(x) / self.temperature)    # [B, C_out, 1]

    def _get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size)
        return torch.sigmoid(s / self.temperature)                     # [B, 1, 1, k]

    def _get_kernel_attention(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1)
        return F.softmax(k / self.temperature, dim=1)                  # [B, n, 1, 1, 1]

    def forward(self, x: torch.Tensor):
        """x: [B, C_in, L]  →  (channel_attn, filter_attn, spatial_attn, kernel_attn)"""
        x = self.avgpool(x)    # [B, C_in,     1]
        x = self.fc(x)         # [B, attn_ch,  1]
        x = self.bn(x)
        x = self.relu(x)
        return (
            self.func_channel(x),   # [B, C_in,  1]      or 1.0
            self.func_filter(x),    # [B, C_out, 1]      or 1.0
            self.func_spatial(x),   # [B, 1, 1,  k]      or 1.0
            self.func_kernel(x),    # [B, n, 1, 1, 1]    or 1.0
        )


# ════════════════════════════════════════════════════════════════════════════
#  ODConv1d  —  fixed, works for any C_in
# ════════════════════════════════════════════════════════════════════════════

class ODConv1d(nn.Module):
    """
    Omni-Dimensional Dynamic Convolution — 1-D, fixed generic variant.

    Drop-in replacement for nn.Conv1d with any C_in (including C_in=1).

    The four fixes over the original single-channel ECG variant:

        FIX 1  weight shape:  (n, C_out, 1, k)  →  (n, C_out, C_in//groups, k)
        FIX 2  batch fold:    reshape(1, B, L)   →  reshape(1, B*C_in, L)
        FIX 3  kernel view:   view(B*Co, 1, k)   →  view(B*Co, C_in//g, k)
        FIX 4  conv groups:   groups=B            →  groups=B*groups

    Grouped-convolution trick:
        Folds the batch dimension into the groups dimension of F.conv1d
        so every sample in the batch gets its own assembled kernel —
        all processed in a single GPU-efficient call.

    Attention heads active per call:
        C_in = 1  →  αfi, αsi, αwi             (αci skipped)
        C_in > 1  →  αci, αfi, αsi, αwi        (all four)

    Input : [B, C_in, L]
    Output: [B, C_out, L_out]
    where  L_out = floor((L + 2·pad - dil·(k-1) - 1) / stride + 1)
    """

    def __init__(
        self,
        in_planes:   int,
        out_planes:  int,
        kernel_size: int,
        stride:      int   = 1,
        padding:     int   = 0,
        dilation:    int   = 1,
        groups:      int   = 1,
        reduction:   float = 0.0625,
        kernel_num:  int   = 4,
    ) -> None:
        super().__init__()

        if in_planes % groups != 0:
            raise ValueError(f"in_planes ({in_planes}) must be divisible by groups ({groups}).")
        if out_planes % groups != 0:
            raise ValueError(f"out_planes ({out_planes}) must be divisible by groups ({groups}).")
        if kernel_num < 1:
            raise ValueError(f"kernel_num must be ≥ 1, got {kernel_num}.")

        self.in_planes   = in_planes
        self.out_planes  = out_planes
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.dilation    = dilation
        self.groups      = groups
        self.kernel_num  = kernel_num

        self.attention = Attention1d(
            in_planes, out_planes, kernel_size,
            groups=groups, reduction=reduction, kernel_num=kernel_num,
        )

        # FIX 1: weight shape uses in_planes // groups, not hardcoded 1
        self.weight = nn.Parameter(
            torch.empty(kernel_num, out_planes, in_planes // groups, kernel_size),
            requires_grad=True,
        )
        self._initialize_weights()

        if kernel_size == 1 and kernel_num == 1:
            self._forward_impl = self._forward_pw1x
        else:
            self._forward_impl = self._forward_common

    def _initialize_weights(self) -> None:
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out", nonlinearity="relu")

    def update_temperature(self, temperature: float) -> None:
        self.attention.update_temperature(temperature)

    def _forward_common(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward — all four attention dimensions applied.

        Tensor shapes step by step:
        ────────────────────────────────────────────────────────────────
        x input                      [B, C_in,  L]

        channel_attention            [B, C_in,  1]      or 1.0
        filter_attention             [B, C_out, 1]      or 1.0
        spatial_attention            [B, 1, 1,  k]      or 1.0
        kernel_attention             [B, n, 1, 1, 1]    or 1.0

        x * channel_attention        [B, C_in,  L]      head 1 on input

        FIX 2 ↓
        x.reshape(1, B*C_in, L)      [1, B*C_in, L]     batch folded in

        weight.unsqueeze(0)          [1, n, C_out, C_in/g, k]
        × spatial_attention          [B, 1,    1,     1,   k]  broadcast
        × kernel_attention           [B, n,    1,     1,   1]  broadcast
        = aggregate_weight           [B, n, C_out, C_in/g, k]
        .sum(dim=1)                  [B,    C_out, C_in/g, k]

        FIX 3 ↓
        .view(B*C_out, C_in/g, k)    [B*C_out, C_in/g, k]  grouped kernel

        FIX 4 ↓
        F.conv1d groups=B*groups     [1, B*C_out, L_out]
        .view(B, C_out, L_out)       [B, C_out,   L_out]

        × filter_attention           [B, C_out, 1]     head 2 on output
        = output                     [B, C_out, L_out]
        """
        channel_attention, filter_attention, spatial_attention, kernel_attention = \
            self.attention(x)

        batch_size, _, length = x.size()

        # Head 1 — scale input channels
        x = x * channel_attention                              # [B, C_in, L]

        # FIX 2: fold batch × C_in into channel dim
        x = x.reshape(1, batch_size * self.in_planes, length) # [1, B*C_in, L]

        # Assemble per-sample kernel from n banks
        aggregate_weight = (
            self.weight.unsqueeze(0)    # [1, n, C_out, C_in/g, k]
            * spatial_attention         # [B, 1,    1,      1,   k]
            * kernel_attention          # [B, n,    1,      1,   1]
        )                               # [B, n, C_out, C_in/g, k]

        aggregate_weight = aggregate_weight.sum(dim=1)         # [B, C_out, C_in/g, k]

        # FIX 3: reshape uses C_in // groups, not 1
        aggregate_weight = aggregate_weight.view(
            batch_size * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size,
        )                                                      # [B*C_out, C_in/g, k]

        # FIX 4: groups = B * groups (not just B)
        output = F.conv1d(
            x,
            weight   = aggregate_weight,
            bias     = None,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups * batch_size,               # ← FIX 4
        )                                                      # [1, B*C_out, L_out]

        output = output.view(batch_size, self.out_planes, -1)  # [B, C_out, L_out]

        # Head 2 — scale output filters
        return output * filter_attention                        # [B, C_out, L_out]

    def _forward_pw1x(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path for k=1, n=1. Spatial and kernel attentions are 1.0."""
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        output = F.conv1d(
            x,
            weight   = self.weight.squeeze(0),   # [C_out, C_in/g, 1]
            bias     = None,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups,
        )
        return output * filter_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ODConv1d expects [B, C, L], got {tuple(x.shape)}.")
        if x.size(1) != self.in_planes:
            raise ValueError(
                f"Channel mismatch: input has C={x.size(1)}, "
                f"but ODConv1d was built with in_planes={self.in_planes}."
            )
        return self._forward_impl(x)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_planes}, out={self.out_planes}, "
            f"k={self.kernel_size}, stride={self.stride}, "
            f"pad={self.padding}, groups={self.groups}, kernel_num={self.kernel_num}"
        )


# ════════════════════════════════════════════════════════════════════════════
#  U-Net building blocks — 100% ODConv1d, no ConvBnRelu fallback
# ════════════════════════════════════════════════════════════════════════════

class ODConvBlock(nn.Module):
    """
    Two consecutive ODConv1d layers, each followed by BN + ReLU.
    Same-padding keeps signal length L unchanged.

    Both conv1 AND conv2 use ODConv1d for any in_ch, including in_ch > 1.
    This is possible because the four fixes above removed the C_in=1 constraint.

    Attention heads per sub-layer:
        conv1: in_ch=C_in  → αci (if C_in>1), αfi, αsi, αwi
        conv2: in_ch=C_out → αci (if C_out>1), αfi, αsi, αwi

    Shape: [B, in_ch, L] → [B, out_ch, L]
    """

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int,
        kernel_size: int   = 9,
        kernel_num:  int   = 4,
        reduction:   float = 0.0625,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2   # same-padding

        self.block = nn.Sequential(
            # conv1: handles any in_ch — FIX 1-4 make this work for in_ch > 1
            ODConv1d(in_ch,  out_ch, kernel_size,
                     padding=pad, kernel_num=kernel_num, reduction=reduction),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),

            # conv2: out_ch is always > 1 in practice — all four heads active
            ODConv1d(out_ch, out_ch, kernel_size,
                     padding=pad, kernel_num=kernel_num, reduction=reduction),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder up-sampling block.

        ConvTranspose1d  ×2 upsample, halve channels
        concat with encoder skip connection
        ODConvBlock      fuse with full ODConv1d attention

    Shape:
        x    [B, in_ch,  L]
        skip [B, out_ch, L*2]
        out  [B, out_ch, L*2]
    """

    def __init__(
        self,
        in_ch:       int,
        out_ch:      int,
        kernel_size: int   = 9,
        kernel_num:  int   = 4,
        reduction:   float = 0.0625,
    ) -> None:
        super().__init__()
        # ConvTranspose: doubles L, halves channels
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concat: channels = out_ch (up) + out_ch (skip) = in_ch
        self.conv = ODConvBlock(in_ch, out_ch, kernel_size, kernel_num, reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                                   # [B, out_ch, L*2]
        if x.size(-1) != skip.size(-1):                  # odd-length guard
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        x = torch.cat([skip, x], dim=1)                 # [B, in_ch, L*2]
        return self.conv(x)                              # [B, out_ch, L*2]


# ════════════════════════════════════════════════════════════════════════════
#  ECG U-Net — every conv layer is ODConv1d
# ════════════════════════════════════════════════════════════════════════════

class ECGUNet(nn.Module):
    """
    1-D U-Net for single-channel ECG signal processing.

    Every convolutional layer — encoder (conv1 AND conv2), bottleneck,
    decoder — is an ODConv1d with all applicable attention heads active.

    No ConvBnRelu fallback anywhere. This is possible because the four
    fixes make ODConv1d generic over any C_in.

    Attention heads per encoder level:
        enc1 conv1  C_in=  1  → αfi, αsi, αwi
        enc1 conv2  C_in= 32  → αci, αfi, αsi, αwi
        enc2 conv1  C_in= 32  → αci, αfi, αsi, αwi
        enc2 conv2  C_in= 64  → αci, αfi, αsi, αwi
        enc3 conv1  C_in= 64  → αci, αfi, αsi, αwi
        enc3 conv2  C_in=128  → αci, αfi, αsi, αwi
        bottleneck  C_in=128  → αci, αfi, αsi, αwi
        decoder     C_in=any  → αci, αfi, αsi, αwi

    Args:
        in_channels:  Input channels (1 for single-lead ECG).
        out_channels: 1 = heatmap/regression, 2 = binary segmentation.
        base_filters: Channel count at enc1. Doubles each level: [f,2f,4f,8f]
        kernel_size:  Temporal kernel size for all ODConv1d layers.
        kernel_num:   Number of parallel weight banks n (≥ 1).
        reduction:    Attention bottleneck ratio.

    Input : [B, 1, L]
    Output: [B, out_channels, L]  raw logits — apply sigmoid for probabilities

    Channel flow (base_filters=32, L=1024):
        [B,   1, 1024]  →enc1→   [B,  32, 1024]  skip1
        pool            →         [B,  32,  512]
        →enc2→          →         [B,  64,  512]  skip2
        pool            →         [B,  64,  256]
        →enc3→          →         [B, 128,  256]  skip3
        pool            →         [B, 128,  128]
        →bottleneck→    →         [B, 256,  128]
        →up1+skip3→     →         [B, 128,  256]
        →up2+skip2→     →         [B,  64,  512]
        →up3+skip1→     →         [B,  32, 1024]
        →head→          →         [B,   1, 1024]
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        out_channels: int   = 1,
        base_filters: int   = 32,
        kernel_size:  int   = 9,
        kernel_num:   int   = 4,
        reduction:    float = 0.0625,
    ) -> None:
        super().__init__()
        f = base_filters

        # ── Encoder: both conv layers in each block are ODConv1d ─────────────
        self.enc1 = ODConvBlock(in_channels, f,   kernel_size, kernel_num, reduction)
        self.enc2 = ODConvBlock(f,           f*2, kernel_size, kernel_num, reduction)
        self.enc3 = ODConvBlock(f*2,         f*4, kernel_size, kernel_num, reduction)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ── Bottleneck: ODConvBlock with C_in > 1, all four heads active ─────
        self.bottleneck = ODConvBlock(f*4, f*8, kernel_size, kernel_num, reduction)

        # ── Decoder: ODConvBlock in each UpBlock, all four heads active ──────
        self.up1 = UpBlock(f*8, f*4, kernel_size, kernel_num, reduction)
        self.up2 = UpBlock(f*4, f*2, kernel_size, kernel_num, reduction)
        self.up3 = UpBlock(f*2, f,   kernel_size, kernel_num, reduction)

        # ── Head: plain 1×1 conv — final channel projection ─────────────────
        self.head = nn.Conv1d(f, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.head.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.head.bias, 0.0)

    def update_temperature(self, temperature: float) -> None:
        """Propagate temperature to every ODConv1d in the network."""
        for m in self.modules():
            if isinstance(m, ODConv1d):
                m.update_temperature(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, L]
        Returns:
            raw logits [B, out_channels, L]
            apply torch.sigmoid() to get probabilities
        """
        
        
        if x.dim() != 3:
            raise ValueError(f"Expected [B, 1, L], got {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"ECGUNet requires C_in=1, got C_in={x.size(1)}.")

        # ── Encoder ──────────────────────────────────────────────────────────
        s1   = self.enc1(x)                   # [B,  f,   L]
        s2   = self.enc2(self.pool(s1))        # [B,  f*2, L/2]
        s3   = self.enc3(self.pool(s2))        # [B,  f*4, L/4]

        # ── Bottleneck ───────────────────────────────────────────────────────
        neck = self.bottleneck(self.pool(s3))  # [B,  f*8, L/8]

        # ── Decoder ──────────────────────────────────────────────────────────
        d1   = self.up1(neck, s3)              # [B,  f*4, L/4]
        d2   = self.up2(d1,   s2)              # [B,  f*2, L/2]
        d3   = self.up3(d2,   s1)              # [B,  f,   L]

        # ── Head ─────────────────────────────────────────────────────────────
        return self.head(d3)                   # [B,  out_channels, L]
    
    
    