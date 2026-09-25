"""
CS-TDS-CT: Channel-Shared TDS + Circular Transformer
=====================================================

A novel architecture for sEMG-to-pose estimation that addresses all three
generalisation challenges simultaneously:

  1. User generalisation   — log-magnitude compression + shared weights
  2. Stage generalisation  — compositional design: muscle primitives → synergies → dynamics
  3. Rotational invariance — exact equivariance chain end-to-end

Pipeline:
  (B, 16, T)
    → reshape (B*16, 1, T)
    → Shared Temporal CNN (same weights for all 16 channels)   ← equivariant
    → |x| + log compression                                    ← user-invariant
    → (B, T', 16, d) → Linear(d → h_dim)                      ← shared across channels
    → N × [Spatial CyRoPE attn | Temporal RoPE attn]           ← ring-topology aware
    → Attention pooling over channels                           ← permutation-equivariant
    → MLP head → (B, 20, T')

Key insight: process each channel with the SAME CNN so the features are
identical regardless of which ring-position a channel occupies.  Novel
gestures are novel COMPOSITIONS of known muscle activations → the factored
design (atoms → synergies → dynamics) generalises to unseen stages.

References:
  - Hannun et al., "Time-Depth Separable Convolutions", Interspeech 2019
  - Su et al., "RoFormer: Rotary Position Embedding", Neurocomputing 2024
  - Salter et al., "emg2pose", NeurIPS Datasets Track 2024
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class CSTDSConfig:
    """Configuration for CS-TDS-CT model."""
    # Input / Output
    input_channels: int = 16
    output_dim: int = 20

    # Channel-Shared Temporal CNN
    # Smaller defaults prevent CNN overfitting while Transformer handles generalisation
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    cnn_kernels: List[int] = field(default_factory=lambda: [11, 5, 5, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [5, 2, 4, 2])
    n_residual_blocks: int = 1          # 1 residual block per stage (was 2, reduces CNN capacity)
    use_cs_tds: bool = False            # use ChannelSharedTDS blocks instead of ResidualConv

    # Factored Transformer
    # 2 spatial + 4 temporal: fewer spatial blocks = less cross-channel memorisation
    # temporal dynamics are more invariant across users/stages
    d_model: int = 256
    n_spatial_blocks: int = 2
    n_temporal_blocks: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    use_circular_spatial_rope: bool = True
    # Learned per-electrode identity, added before spatial attention.
    # Pair with use_circular_spatial_rope=False so mixing can be channel-specific.
    use_channel_embeddings: bool = False

    # Pooling
    use_attention_pooling: bool = True
    use_log_compression: bool = False

    # CNN weight sharing
    # True  → one shared CNN for all 16 channels (parameter-efficient, equivariant)
    # False → each channel gets its own CNN (unshared; use small channels to match param count)
    share_cnn: bool = True

    # Channel rotation augmentation (training only)
    # Full range of 8 = all 16 possible cyclic rotations seen during training
    channel_rotation_range: int = 8

    # Mild neighbour mixing simulates off-grid band placement shifts.
    channel_mix_strength: float = 0.0

    # Channel dropout — drop entire channels during training
    # Forces model to predict from incomplete electrode set → prevents cross-channel memorisation
    channel_drop_p: float = 0.10       # probability of zeroing each channel

    # Per-channel amplitude jitter (training only)
    # Log-normal gain simulates inter-user muscle volume variation
    # gain = exp(N(0, amp_jitter_std)), centered at 1.0, always positive
    amplitude_jitter_std: float = 0.15  # σ of log-normal → gain ∈ ~[0.76, 1.30] at 1σ

    # Temporal chunk masking (training only)
    chunk_mask_ratio: float = 0.0  # Set > 0 to enable
    # Batched GPU SpecAugment (EMGFormer RandomFrequencyMask, but not per-window FFT on CPU).
    freq_mask_num: int = 0
    freq_mask_max: int = 128
    gaussian_noise_prob: float = 0.0
    gaussian_noise_min_snr_db: float = 25.0
    gaussian_noise_max_snr_db: float = 35.0
    # Move fixed per-dataset normalization off DataLoader workers.
    input_mean: float = 0.0
    input_std: float = 1.0

    # Slow temporal state adapter
    slow_state_type: str = "none"
    slow_state_hidden_dim: int = 0
    slow_state_layers: int = 1
    slow_state_dropout: float = 0.1

    # Invariance-consistency training
    consistency_enabled: bool = False
    consistency_rotation_range: Optional[int] = None
    consistency_channel_mix_strength: float = 0.0
    consistency_amplitude_jitter_std: float = 0.0

    # Gradient reversal for stage de-biasing (optional)
    use_gradient_reversal: bool = False
    num_stages: int = 1  # set to actual number of stages if using GRL
    stage_reversal_lambda: float = 0.1

    # Legacy attention (pre-RoPE-bug-fix checkpoints)
    # When True, uses nn.MultiheadAttention with RoPE applied BEFORE projection.
    # Key names match old checkpoints exactly so load_from_checkpoint works
    # with strict=True and produces bit-identical results to the old code.
    # Set automatically by test_analysis when +legacy_checkpoint=true.
    use_legacy_attention: bool = False

    # Selects which attention forward-pass variant to instantiate.
    #   "fixed"        — current correct order: project Q,K,V → rotate Q,K → SDPA
    #   "legacy_mha"   — old nn.MultiheadAttention, RoPE on full features before
    #                    internal W_q/W_k. Equivalent to use_legacy_attention=True.
    #   "legacy_v2_pre_transpose" — same param shapes as fixed (w_q/w_k/w_v/w_o,
    #                    cyrope dim=head_dim), but RoPE applied to projected Q,K
    #                    BEFORE the (1,2) transpose, so the rotation runs over
    #                    n_heads instead of channels. Reproduces a pre-fix
    #                    forward pass for checkpoints whose state_dict already
    #                    has separate w_q/w_k/w_v keys.
    #   "legacy_v2_pre_proj" — same param shapes as fixed, but RoPE applied to
    #                    LayerNorm(h) reshaped to (B, C, n_heads, head_dim)
    #                    BEFORE the W_q/W_k projection (Q,K consume rotated h,
    #                    V uses unrotated h). Rotation again runs over n_heads.
    # When use_legacy_attention=True is left over from older configs, it is
    # treated as rope_mode="legacy_mha".
    rope_mode: str = "fixed"

    # ── CS-TDS-CT++ experimental architectural toggles ──────────────────
    # Each toggle is independent and defaults OFF (cs_tds_ct.yaml baseline
    # behaviour unchanged). Decision-tree style — flip any subset in the
    # cs_tds_ct_pp.yaml to ablate.

    # B.5 — Anatomy-conditioned autoregressive output head.
    # Predicts proximal → mid → distal joints in 3 chained steps so distal
    # predictions can attend to predicted proximal context. Self-conditioning
    # (no teacher forcing) — model sees its own upstream predictions
    # uniformly between train and inference, so no exposure bias.
    use_anatomy_head: bool = False

    # B.6 — Per-finger attention pooling readout.
    # Replaces single-query AttentionPooling with n_finger_queries learnable
    # queries (one per anatomical finger). Each query has dedicated readout
    # capacity. Permutation-equivariant when queries are channel-agnostic.
    use_per_finger_queries: bool = False
    n_finger_queries: int = 5

    # B.4 — Multi-rate (fast + slow) temporal frontend.
    # Parallel slow CNN branch with much lower stride (~16× vs ~80×) so
    # tonic / sub-25Hz envelope information (wrist orientation, postural
    # co-contraction) survives downsampling. Only valid when share_cnn=True.
    use_multirate_frontend: bool = False
    # Concat projected TDS tokens onto transformer tokens immediately
    # before channel pooling, then Linear(2d→d). Default OFF.
    use_tds_skip: bool = False
    # If True: spatial (CRoPE) on 16 channels → pool to (B, T, d) → temporal
    # attention once, like EMGFormer. Default False keeps 16 independent
    # time streams (B×16, T, d), which is ~6–8× slower per epoch.
    temporal_after_channel_pool: bool = False
    # If True: one circular Conv2d stem on the (ring × time) plane instead of
    # 16 independent 1-ch CNNs. Same cyclic equivariance; ~EMGFormer FLOPs
    # when paired with temporal_after_channel_pool.
    use_circular_stem: bool = False
    circular_ring_kernel: int = 3
    # Residual circular Conv1d along the electrode ring after the stem
    # LayerNorm. Extra neighbour mixing; still cyclic-equivariant.
    use_circular_channel_mix: bool = False
    circular_channel_mix_kernel: int = 5
    # Identity skip: preprocessor tokens (after pool/project) add onto
    # the temporal-encoder output immediately before the MLP head.
    use_encoder_skip: bool = False
    # If True, CRoPE/spatial runs at CNN width (cheap, seq=16, d=stem)
    # then pool → project → temporal at d_model. Needed so Circ-S is
    # ~EMGFormer FLOPs while keeping a ring mixer.
    spatial_at_stem_dim: bool = False
    # Recompute spatial-attn activations in backward so microbatch can be larger.
    use_spatial_checkpoint: bool = False
    # PoseModule downsamples labels to T' in train instead of upsampling
    # preds to 2 kHz, so autograd stays at encoder rate.
    keep_encoder_rate: bool = False
    # First CNN conv mixes all 16 EMG channels (EMGFormer TDS layer-1),
    # then remaining stages run on the mixed (B, d, T') stream. Breaks
    # cyclic invariance; spatial attention is skipped.
    mix_emg_channels_at_stem: bool = False
    # EMGFormer-style preprocessor: LayerNorm (no BatchNorm), a wide temporal
    # conv that extracts local MUAP structure, then residual TDS
    # (conv → full feature mix → GELU → skip → LN) repeated, then the
    # transformer. Only used with mix_emg_channels_at_stem.
    use_signal_tds_frontend: bool = False
    # Activity-Conditioned Cyclic Synergy Attention (ACSA).  Shared temporal
    # signal extraction is interleaved with content-dependent attention over
    # the electrode ring.  Attention uses only cyclic relative offsets, so a
    # cuff rotation rolls the latent channels instead of changing the result.
    use_acsa_frontend: bool = False
    acsa_heads: int = 4
    acsa_router_dim: int = 64
    acsa_route_strides: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    acsa_residual_init: float = 0.1
    acsa_dynamic_edges: bool = True
    # Fast Synergy-Adaptive Electrode Mixer (SAEM): one shallow shared temporal
    # stem, time-varying cross-electrode attention, immediate collapse to one
    # mixed stream, then regular full-mix Signal-TDS + temporal transformers.
    # Unlike ACSA, this does not preserve cuff-rotation equivariance; it is the
    # speed-first architecture for Stage generalization.
    use_saem_frontend: bool = False
    saem_local_dim: int = 16
    saem_heads: int = 4
    saem_route_stride: int = 8
    saem_residual_init: float = 0.1
    saem_dynamic_attention: bool = True
    # Fixed multiscale identity route from the post-electrode-collapse tensor
    # to the final frontend output. Prevents the deep strided stack from
    # learning a bias-only constant representation.
    saem_signal_skip: bool = False
    # Slow branch architecture — should have roughly the same depth as the
    # fast branch but smaller per-stage strides; defaults below produce a
    # ~16× total downsample vs ~80× for the default fast branch.
    multirate_slow_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    multirate_slow_kernels:  List[int] = field(default_factory=lambda: [7, 5, 5, 3])
    multirate_slow_strides:  List[int] = field(default_factory=lambda: [2, 2, 2, 2])

    # A.3 upgrade — consistency-loss formulation.
    #   "smooth_l1" (default) — primary↔aux output-space smooth-L1 (current)
    #   "infonce"             — SimCLR-style contrastive on pooled features
    consistency_mode: str = "smooth_l1"
    infonce_tau: float = 0.1


# ============================================================
# Circular RoPE (ring geometry, exact periodicity)
# ============================================================

class CircularRoPE(nn.Module):
    """
    Circular Rotary Position Embedding for the 16-channel ring.

    Frequencies are integer multiples of 2π/C, guaranteeing that
    embedding(k) == embedding(k + C).  Applying to Q and K makes
    attention weights depend only on angular *differences* mod C.
    """

    def __init__(self, dim: int, num_positions: int = 16):
        super().__init__()
        self.dim = dim
        self.num_positions = num_positions

        half_dim = dim // 2
        freq_steps = torch.arange(1, half_dim + 1).float()
        inv_freq = freq_steps * (2 * math.pi / num_positions)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return self._apply_rotary(x, emb)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        s = emb[..., : emb.shape[-1] // 2]
        c = emb[..., emb.shape[-1] // 2:]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


# ============================================================
# Linear RoPE (standard 1-D for temporal axis)
# ============================================================

class LinearRoPE(nn.Module):
    """Standard 1-D Rotary Position Embedding for the time axis."""

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        positions = torch.arange(T, device=x.device)
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return CircularRoPE._apply_rotary(x, emb)


# ============================================================
# Attention Blocks
# ============================================================

# Base classes used purely as isinstance() markers so the backbone
# dispatch in _forward_backbone works for every variant (fixed +
# every legacy flavour) without needing per-variant code paths.
class _SpatialBlockBase(nn.Module):
    pass


class _TemporalBlockBase(nn.Module):
    pass


# ============================================================
# Legacy Attention Blocks (pre-RoPE-bug-fix)
# These reproduce the OLD forward pass exactly:
#   cyrope/rope is applied to the full feature vector BEFORE
#   nn.MultiheadAttention's internal W_q/W_k projections.
# Key names match old checkpoints so strict loading works.
# Use only when loading pre-bug-fix checkpoints.
# ============================================================

class LegacySpatialAttentionBlock(_SpatialBlockBase):
    """Old SpatialAttentionBlock: RoPE before projection (buggy order)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 num_channels: int = 16, dropout: float = 0.1,
                 use_circular_rope: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        # Old: CRoPE initialised with d_model, not head_dim
        self.cyrope = (
            CircularRoPE(d_model, num_positions=num_channels)
            if use_circular_rope else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Old buggy order: rotate features, THEN MHA applies W_q/W_k
        qk = self.cyrope(h) if self.cyrope is not None else h
        att, _ = self.attn(qk, qk, h)
        x = x + att
        x = x + self.ffn(self.norm2(x))
        return x


class LegacyTemporalAttentionBlock(_TemporalBlockBase):
    """Old TemporalAttentionBlock: RoPE before projection (buggy order)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_time_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        # Old: RoPE initialised with d_model, not head_dim
        self.rope = LinearRoPE(d_model, max_len=max_time_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Old buggy order: rotate features, THEN MHA applies W_q/W_k
        h_rope = self.rope(h)
        att, _ = self.attn(h_rope, h_rope, h)
        x = x + att
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialAttentionBlock(_SpatialBlockBase):
    """
    Pre-norm Transformer block attending across C=16 channels.
    CyRoPE applied to projected Q and K — correct order for Z_16 equivariance.

    The bug in the previous version: CyRoPE was applied to features before
    nn.MultiheadAttention's internal W_q/W_k projections, giving
    score = x^T R_i^T (W_q^T W_k) R_j x — not relative.
    Correct order: project first, rotate Q and K, then dot-product.

    Input / output: (B*T', C, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 num_channels: int = 16, dropout: float = 0.1,
                 use_circular_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Explicit projections so RoPE sits between projection and dot-product
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

        # RoPE initialised with head_dim, not d_model
        self.cyrope = (
            CircularRoPE(self.head_dim, num_positions=num_channels)
            if use_circular_rope else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T', C, d_model)
        BT, C, _ = x.shape
        h = self.norm1(x)

        # 1. Project — W applied before R
        q = self.w_q(h).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)
        # shape: (BT, n_heads, C, head_dim)

        # 2. Rotate projected Q and K — now score = (W_q x)^T R_{j-i} (W_k x)
        if self.cyrope is not None:
            q = self.cyrope(q)
            k = self.cyrope(k)

        # 3. Scaled dot-product attention
        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        # 4. Merge heads and output projection
        out = out.transpose(1, 2).reshape(BT, C, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAttentionBlock(_TemporalBlockBase):
    """
    Pre-norm Transformer block attending across T' time steps.
    Linear RoPE applied to projected Q and K — correct order for relative PE.

    Input / output: (B*C, T', d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_time_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        # RoPE initialised with head_dim, not d_model
        self.rope = LinearRoPE(self.head_dim, max_len=max_time_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*C, T', d_model)
        BC, T, _ = x.shape
        h = self.norm1(x)

        # 1. Project — W applied before R
        q = self.w_q(h).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)
        # shape: (BC, n_heads, T', head_dim)

        # 2. Rotate projected Q and K — score = (W_q x)^T R_{j-i} (W_k x)
        q = self.rope(q)
        k = self.rope(k)

        # 3. Scaled dot-product attention
        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        # 4. Merge heads and output projection
        out = out.transpose(1, 2).reshape(BC, T, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Legacy V2 Attention Blocks
# Same parameter shapes as the fixed blocks (separate w_q/w_k/w_v/w_o,
# cyrope/rope dim = head_dim) but reproduce a *pre-fix* forward pass.
# These exist so that checkpoints whose state_dict already has separate
# w_q/w_k/w_v keys (i.e. NOT the nn.MultiheadAttention legacy) can still
# be evaluated with the exact forward pass that produced their training-
# time metrics.
#
# Two flavours are provided:
#   - LegacyV2*PreTransposeAttentionBlock — RoPE applied to projected Q,K
#     BEFORE the (1,2) transpose to head-major. seq_len that RoPE sees is
#     n_heads, not C/T → rotation runs over the wrong axis.
#   - LegacyV2*PreProjAttentionBlock — RoPE applied to LayerNorm(h) reshaped
#     to (..., n_heads, head_dim) BEFORE W_q/W_k. Q,K consume rotated h, V
#     uses unrotated h. Rotation again runs over n_heads.
# ============================================================


class LegacyV2PreTransposeSpatialAttentionBlock(_SpatialBlockBase):
    """Spatial block with current param shapes but RoPE applied before
    transpose-to-heads, so rotation runs over n_heads instead of C."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 num_channels: int = 16, dropout: float = 0.1,
                 use_circular_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.cyrope = (
            CircularRoPE(self.head_dim, num_positions=num_channels)
            if use_circular_rope else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, _ = x.shape
        h = self.norm1(x)

        # Project, then reshape — but DO NOT transpose yet.
        q = self.w_q(h).reshape(BT, C, self.n_heads, self.head_dim)
        k = self.w_k(h).reshape(BT, C, self.n_heads, self.head_dim)
        v = self.w_v(h).reshape(BT, C, self.n_heads, self.head_dim)

        # Apply RoPE BEFORE transpose. CircularRoPE looks at x.shape[-2],
        # which is now n_heads → rotation runs over heads, not channels.
        if self.cyrope is not None:
            q = self.cyrope(q)
            k = self.cyrope(k)

        # Now move to head-major for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # shape: (BT, n_heads, C, head_dim)

        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(BT, C, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


class LegacyV2PreTransposeTemporalAttentionBlock(_TemporalBlockBase):
    """Temporal block with current param shapes but RoPE applied before
    transpose-to-heads, so rotation runs over n_heads instead of T."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_time_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.rope = LinearRoPE(self.head_dim, max_len=max_time_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BC, T, _ = x.shape
        h = self.norm1(x)

        q = self.w_q(h).reshape(BC, T, self.n_heads, self.head_dim)
        k = self.w_k(h).reshape(BC, T, self.n_heads, self.head_dim)
        v = self.w_v(h).reshape(BC, T, self.n_heads, self.head_dim)

        # rope.forward looks at x.shape[-2] = n_heads → rotation over heads.
        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(BC, T, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


class LegacyV2PreProjSpatialAttentionBlock(_SpatialBlockBase):
    """Spatial block with current param shapes but RoPE applied to
    LayerNorm(h) BEFORE W_q/W_k. Q,K consume rotated h; V uses unrotated h."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 num_channels: int = 16, dropout: float = 0.1,
                 use_circular_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.cyrope = (
            CircularRoPE(self.head_dim, num_positions=num_channels)
            if use_circular_rope else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, _ = x.shape
        h = self.norm1(x)

        # Rotate h before projection. To make cyrope (dim=head_dim) line up,
        # reshape h into (BT, C, n_heads, head_dim). cyrope's seq_len is then
        # n_heads, so rotation again runs over the head axis.
        if self.cyrope is not None:
            h_4d = h.reshape(BT, C, self.n_heads, self.head_dim)
            h_rot = self.cyrope(h_4d).reshape(BT, C, self.d_model)
        else:
            h_rot = h

        # Q, K from rotated h; V from unrotated h.
        q = self.w_q(h_rot).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h_rot).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).reshape(BT, C, self.n_heads, self.head_dim).transpose(1, 2)

        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(BT, C, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


class LegacyV2PreProjTemporalAttentionBlock(_TemporalBlockBase):
    """Temporal block with current param shapes but RoPE applied to
    LayerNorm(h) BEFORE W_q/W_k. Q,K consume rotated h; V uses unrotated h."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_time_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self._attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.rope = LinearRoPE(self.head_dim, max_len=max_time_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BC, T, _ = x.shape
        h = self.norm1(x)

        h_4d = h.reshape(BC, T, self.n_heads, self.head_dim)
        h_rot = self.rope(h_4d).reshape(BC, T, self.d_model)

        q = self.w_q(h_rot).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h_rot).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).reshape(BC, T, self.n_heads, self.head_dim).transpose(1, 2)

        drop_p = self._attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(BC, T, self.d_model)
        x = x + self.w_o(out)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Attention Pooling over Channels
# ============================================================

class AttentionPooling(nn.Module):
    """
    Learnable query-based attention pooling over the channel dimension.
    Permutation-equivariant: attention is a set function.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BT', C, d_model) → (BT', d_model)"""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


# ============================================================
# Channel-Shared Temporal CNN
# ============================================================

class ResidualConvBlock(nn.Module):
    """1-D residual block: Conv-BN-GELU-Conv-BN + skip, matching TDS depth."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ChannelSharedTDSBlock(nn.Module):
    """
    Channel-Shared TDS block: mimics TDS inductive bias while remaining
    exactly channel-equivariant.

    Standard TDS has two sub-blocks per block:
      1. Temporal conv  — Conv2d over (1, kernel_width) per channel group
      2. FC             — Linear over full (C * feature_width) product

    Channel-shared version separates these so ALL channels use the same weights:
      1. Dense temporal conv    — Conv1d(d, d, k) with groups=1 (cross-feature mixing)
      2. Per-channel pointwise FC — Conv1d(d, d, 1) = Linear(d, d) per time step
                                    (shared across channels, NOT mixing channels)

    Both sub-blocks have LayerNorm + residual skip, matching TDS's formulation.
    Processing on tensor shape (B*C, d, T') so nn.Module is unaware of C.
    """

    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        # Sub-block 1: dense temporal conv (groups=1, mixes across feature dim)
        # NOTE: Unlike original TDS which uses depthwise (groups=channels),
        # we intentionally use dense conv here — it cross-correlates feature
        # maps along time, giving the shared CNN more representational power.
        # This is equivariant w.r.t. EMG ring channels because they sit in
        # the batch dimension (B*C), not the Conv1d channel dimension.
        self.temporal_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2)
        self.temporal_norm = nn.LayerNorm(channels)

        # Sub-block 2: pointwise FC applied per time step (= Conv1d k=1)
        # Two-layer FC like original TDS FC block
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.fc_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*C, d, T')
        # Sub-block 1: temporal conv
        residual = x
        h = self.temporal_conv(x)          # (B*C, d, T')
        h = h + residual
        # LayerNorm expects (*, d) — transpose
        h = self.temporal_norm(h.transpose(1, 2)).transpose(1, 2)

        # Sub-block 2: per-time-step FC
        residual = h
        h = h.transpose(1, 2)              # (B*C, T', d)
        h = self.fc(h)                     # (B*C, T', d)
        h = h.transpose(1, 2)              # (B*C, d, T')
        h = h + residual
        h = self.fc_norm(h.transpose(1, 2)).transpose(1, 2)
        return self.dropout(h)


class FullMixTDSBlock(nn.Module):
    """Hannun / EMGFormer TDS on an already-mixed (B, d, T) stream.

    ChannelSharedTDSBlock keeps EMG electrodes in the batch dim and never
    mixes them. After ``mix_emg_channels_at_stem`` the tensor is (B, d, T);
    this block remaps ``d`` into ``(n_groups, width)`` (16 groups when
    ``d % 16 == 0``, electrode-aligned) and, at every layer:

      1. Conv2d over time with ``groups=1`` (groups talk to each other)
      2. FC over the full ``d``-vector at each time step (true TDS mix)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        dropout: float = 0.1,
        n_groups: int | None = None,
    ):
        super().__init__()
        if n_groups is None:
            if channels % 16 == 0:
                n_groups = 16
            elif channels % 8 == 0:
                n_groups = 8
            else:
                n_groups = 1
        if channels % n_groups != 0:
            raise ValueError(f"channels={channels} not divisible by n_groups={n_groups}")
        self.n_groups = int(n_groups)
        self.width = channels // self.n_groups
        k = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.conv2d = nn.Conv2d(
            self.n_groups,
            self.n_groups,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=1,
            bias=True,
        )
        self.conv_norm = nn.LayerNorm(channels)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.fc_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, T = x.shape
        h = x.reshape(B, self.n_groups, self.width, T)
        h = F.relu(self.conv2d(h), inplace=True)
        h = h.reshape(B, d, T) + x
        h = self.conv_norm(h.transpose(1, 2)).transpose(1, 2)
        residual = h
        h = self.fc(h.transpose(1, 2)).transpose(1, 2) + residual
        h = self.fc_norm(h.transpose(1, 2)).transpose(1, 2)
        return self.dropout(h)


class _ChannelLN(nn.Module):
    """LayerNorm over the channel axis of (B, C, T)."""

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class SignalExtractConv(nn.Module):
    """Temporal conv that extracts local EMG structure, then GELU + LayerNorm.

    Wider kernels than a 2-sample stem so each stage sees more of a MUAP
    (~10–15 ms at 2 kHz) before downsampling. No BatchNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        k = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=int(stride),
            padding=k // 2,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = _ChannelLN(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.drop(self.act(self.conv(x))))


class SignalTDSBlock(nn.Module):
    """One residual TDS unit: conv → full mix → GELU → skip → LayerNorm.

    Conv2d ``groups=1`` mixes feature groups over time; the FC mixes the
    full ``d``-vector at each step (Hannun / EMGFormer TDS). GELU after
    the mix, then a residual add and LayerNorm — no BatchNorm, no ReLU.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        dropout: float = 0.1,
        n_groups: int | None = None,
    ):
        super().__init__()
        if n_groups is None:
            if channels % 16 == 0:
                n_groups = 16
            elif channels % 8 == 0:
                n_groups = 8
            else:
                n_groups = 1
        if channels % n_groups != 0:
            raise ValueError(f"channels={channels} not divisible by n_groups={n_groups}")
        self.n_groups = int(n_groups)
        self.width = channels // self.n_groups
        k = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.conv2d = nn.Conv2d(
            self.n_groups,
            self.n_groups,
            kernel_size=(1, k),
            padding=(0, k // 2),
            groups=1,
            bias=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.norm = _ChannelLN(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, T = x.shape
        h = x.reshape(B, self.n_groups, self.width, T)
        h = self.conv2d(h).reshape(B, d, T)
        h = self.fc(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.drop(h)
        return self.norm(h + x)


class SignalTDSFrontend(nn.Module):
    """Preprocessor: extract-conv (LN) → residual TDS mix, repeated.

    Each stage:
      1. Wide same-rate conv (more temporal context) + GELU + LayerNorm
      2. Strided conv if this stage downsamples, again LN
      3. ``n_residual_blocks`` × SignalTDSBlock (conv, full mix, GELU, skip, LN)

    Input:  (B, C, T) mixed EMG
    Output: (B, out_channels[-1], T')
    """

    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        n_residual_blocks: int = 2,
        dropout: float = 0.1,
        in_channels: int = 16,
    ):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides)
        stages: list[nn.Module] = []
        in_ch = int(in_channels)
        for out_ch, k, s in zip(channels, kernels, strides):
            extract_k = int(k) if int(k) % 2 == 1 else int(k) + 1
            stage: list[nn.Module] = [
                SignalExtractConv(in_ch, out_ch, extract_k, stride=1, dropout=dropout),
            ]
            if int(s) > 1:
                down_k = min(extract_k, 11)
                if down_k % 2 == 0:
                    down_k += 1
                stage.append(
                    SignalExtractConv(out_ch, out_ch, down_k, stride=int(s), dropout=dropout)
                )
            tds_k = min(extract_k, 9)
            if tds_k % 2 == 0:
                tds_k += 1
            for _ in range(int(n_residual_blocks)):
                stage.append(SignalTDSBlock(out_ch, kernel_size=tds_k, dropout=dropout))
            stages.extend(stage)
            in_ch = out_ch
        self.net = nn.Sequential(*stages)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CyclicSynergyAttention(nn.Module):
    """Activity-conditioned graph attention over an annular electrode array.

    The dynamic edge from electrode ``i`` to ``j`` is

        softmax(q_i^T k_j / sqrt(d) + b[(j-i) mod C]).

    Shared Q/K/V projections and a bias indexed only by relative cyclic
    displacement make the map exactly equivariant to a cyclic channel shift.
    An invariant context (mean over channels) gates the attention heads, so
    different activities can select different muscle-synergy graphs without
    introducing user/electrode identity.  Routing is computed at a lower
    temporal rate and interpolated back, keeping early channel communication
    affordable even for long 2 kHz windows.

    Input/output: (B, C, D, T).
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 16,
        n_heads: int = 4,
        router_dim: int = 64,
        route_stride: int = 1,
        dropout: float = 0.1,
        residual_init: float = 0.1,
        dynamic_edges: bool = True,
    ):
        super().__init__()
        if router_dim % n_heads != 0:
            raise ValueError(
                f"router_dim={router_dim} must be divisible by n_heads={n_heads}"
            )
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.num_channels = int(num_channels)
        self.n_heads = int(n_heads)
        self.router_dim = int(router_dim)
        self.route_stride = max(1, int(route_stride))
        self.qk_head_dim = self.router_dim // self.n_heads
        self.value_head_dim = self.dim // self.n_heads
        self.dynamic_edges = bool(dynamic_edges)

        self.pre_norm = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.router_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.router_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

        # One learnable prior per head and relative ring offset.  Initializing
        # at zero starts as content-only attention, not a hard local graph.
        self.relative_bias = nn.Parameter(
            torch.zeros(self.n_heads, self.num_channels)
        )
        offsets = (
            torch.arange(self.num_channels)[None, :]
            - torch.arange(self.num_channels)[:, None]
        ) % self.num_channels
        self.register_buffer("relative_offsets", offsets, persistent=False)

        # mean_C is invariant to ring rotation; it can therefore route heads
        # by activity without leaking absolute electrode identity.
        self.activity_gate = nn.Sequential(
            nn.Linear(self.dim, max(self.dim // 4, self.n_heads)),
            nn.GELU(),
            nn.Linear(max(self.dim // 4, self.n_heads), self.n_heads),
            nn.Sigmoid(),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.post_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.dim, self.dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, T = x.shape
        if C != self.num_channels or D != self.dim:
            raise ValueError(
                f"Expected (C,D)=({self.num_channels},{self.dim}), got ({C},{D})"
            )

        # Pool only to infer slowly varying synergy edges; values still
        # retain electrode-specific content and messages return to full rate.
        routed = x.reshape(B * C, D, T)
        if self.route_stride > 1:
            routed = F.avg_pool1d(
                routed,
                kernel_size=self.route_stride,
                stride=self.route_stride,
                ceil_mode=True,
            )
        Tr = routed.shape[-1]
        routed = routed.reshape(B, C, D, Tr).permute(0, 3, 1, 2)
        h = self.pre_norm(routed)  # (B, Tr, C, D)

        q = self.q_proj(h).reshape(
            B, Tr, C, self.n_heads, self.qk_head_dim
        ).permute(0, 1, 3, 2, 4)
        k = self.k_proj(h).reshape(
            B, Tr, C, self.n_heads, self.qk_head_dim
        ).permute(0, 1, 3, 2, 4)
        v = self.v_proj(h).reshape(
            B, Tr, C, self.n_heads, self.value_head_dim
        ).permute(0, 1, 3, 2, 4)

        ring_bias = self.relative_bias[:, self.relative_offsets]
        if self.dynamic_edges:
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores * (self.qk_head_dim ** -0.5)
            scores = scores + ring_bias[None, None, :, :, :]
        else:
            # Static circulant-graph ablation: topology is learned but cannot
            # adapt its edges or head usage to the current activity.
            scores = ring_bias[None, None, :, :, :].expand(
                B, Tr, -1, -1, -1
            )
        attn = self.attn_drop(scores.softmax(dim=-1))
        message = torch.matmul(attn, v)

        # Activity-conditioned head routing, invariant to channel rotation.
        if self.dynamic_edges:
            gate = self.activity_gate(h.mean(dim=2))  # (B, Tr, H)
            message = message * gate[:, :, :, None, None]
        message = message.permute(0, 1, 3, 2, 4).reshape(B, Tr, C, D)
        message = self.out_proj(message)
        if Tr != T:
            message = message.permute(0, 2, 3, 1).reshape(B * C, D, Tr)
            message = F.interpolate(
                message, size=T, mode="linear", align_corners=False
            )
            message = message.reshape(B, C, D, T).permute(0, 3, 1, 2)

        base = x.permute(0, 3, 1, 2)  # (B, T, C, D)
        y = self.post_norm(base + torch.tanh(self.residual_scale) * message)
        y = self.ffn_norm(y + self.ffn(y))
        return y.permute(0, 2, 3, 1).contiguous()


class ACSAFrontend(nn.Module):
    """Multi-scale temporal extraction interleaved with cyclic synergy routing.

    Each stage performs:
      shared temporal Conv1d -> GELU -> LayerNorm -> local residual TDS
      -> activity-conditioned cyclic channel attention.

    Local extraction is shared across electrodes, while ACSA is the only
    cross-electrode operation.  Therefore intermediate features remain
    channel-equivariant and attention/mean pooling can produce an exactly
    cuff-rotation-invariant pose representation.

    Input:  (B, C, T)
    Output: (B, D, C, T')
    """

    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        n_residual_blocks: int = 1,
        dropout: float = 0.1,
        input_channels: int = 16,
        n_heads: int = 4,
        router_dim: int = 64,
        route_strides: Optional[List[int]] = None,
        residual_init: float = 0.1,
        dynamic_edges: bool = True,
    ):
        super().__init__()
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("channels, kernels, and strides must have equal length")
        route_strides = route_strides or [1] * len(channels)
        if len(route_strides) != len(channels):
            raise ValueError("route_strides must have one value per CNN stage")

        stages = nn.ModuleList()
        in_dim = 1
        for out_dim, kernel, stride, route_stride in zip(
            channels, kernels, strides, route_strides
        ):
            k = int(kernel) if int(kernel) % 2 else int(kernel) + 1
            local = nn.ModuleList(
                [
                    SignalTDSBlock(
                        out_dim,
                        kernel_size=min(k, 9),
                        dropout=dropout,
                    )
                    for _ in range(int(n_residual_blocks))
                ]
            )
            stages.append(
                nn.ModuleDict(
                    {
                        "extract": SignalExtractConv(
                            in_dim,
                            out_dim,
                            kernel_size=k,
                            stride=int(stride),
                            dropout=dropout,
                        ),
                        "local_tds": local,
                        "synergy": CyclicSynergyAttention(
                            dim=out_dim,
                            num_channels=input_channels,
                            n_heads=n_heads,
                            router_dim=min(router_dim, out_dim),
                            route_stride=int(route_stride),
                            dropout=dropout,
                            residual_init=residual_init,
                            dynamic_edges=dynamic_edges,
                        ),
                    }
                )
            )
            in_dim = out_dim
        self.stages = stages
        self.input_channels = int(input_channels)
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} electrodes, got {C}")
        h = x.reshape(B * C, 1, T)
        for stage in self.stages:
            h = stage["extract"](h)
            for block in stage["local_tds"]:
                h = block(h)
            D, Ts = h.shape[1], h.shape[2]
            h = h.reshape(B, C, D, Ts)
            h = stage["synergy"](h)
            h = h.reshape(B * C, D, Ts)
        return h.reshape(B, C, self.out_channels, h.shape[-1]).permute(0, 2, 1, 3)


class SynergyAdaptiveElectrodeMixer(nn.Module):
    """Fast, time-varying attention over shallow electrode features.

    Electrode identity is intentionally available through a learned embedding:
    Stage generalization benefits from knowing which physical muscle region
    produced a feature.  Content attention makes the CxC functional graph
    change with the current activity, while a channel-mean context gates heads
    to select different synergy types.  The graph is inferred at a pooled
    temporal rate and interpolated once; all deeper processing is single-stream.

    Input/output: (B, C, D, T).
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 16,
        n_heads: int = 4,
        route_stride: int = 8,
        dropout: float = 0.1,
        residual_init: float = 0.1,
    ):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.num_channels = int(num_channels)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.route_stride = max(1, int(route_stride))

        self.pre_norm = nn.LayerNorm(self.dim)
        self.electrode_embedding = nn.Parameter(
            torch.empty(1, 1, self.num_channels, self.dim)
        )
        nn.init.trunc_normal_(self.electrode_embedding, std=0.02)
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.activity_gate = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.n_heads),
            nn.Sigmoid(),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.post_norm = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, T = x.shape
        if C != self.num_channels or D != self.dim:
            raise ValueError(
                f"Expected (C,D)=({self.num_channels},{self.dim}), got ({C},{D})"
            )
        route = x.reshape(B * C, D, T)
        if self.route_stride > 1:
            route = F.avg_pool1d(
                route,
                kernel_size=self.route_stride,
                stride=self.route_stride,
                ceil_mode=True,
            )
        Tr = route.shape[-1]
        route = route.reshape(B, C, D, Tr).permute(0, 3, 1, 2)
        h = self.pre_norm(route) + self.electrode_embedding
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def split_heads(z: torch.Tensor) -> torch.Tensor:
            return z.reshape(B, Tr, C, self.n_heads, self.head_dim).permute(
                0, 1, 3, 2, 4
            )

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = self.attn_drop(scores.softmax(dim=-1))
        message = torch.matmul(attn, v)

        # Channel-mean context changes head usage by activity without requiring
        # stage labels at train or test time.
        gate = self.activity_gate(route.mean(dim=2))
        message = message * gate[:, :, :, None, None]
        message = message.permute(0, 1, 3, 2, 4).reshape(B, Tr, C, D)
        message = self.out_drop(self.out_proj(message))
        if Tr != T:
            message = message.permute(0, 2, 3, 1).reshape(B * C, D, Tr)
            message = F.interpolate(
                message, size=T, mode="linear", align_corners=False
            )
            message = message.reshape(B, C, D, T).permute(0, 3, 1, 2)

        base = x.permute(0, 3, 1, 2)
        mixed = self.post_norm(
            base + torch.tanh(self.residual_scale) * message
        )
        return mixed.permute(0, 2, 3, 1).contiguous()


class SAEMFrontend(nn.Module):
    """Speed-first dynamic channel mixer followed by single-stream Signal-TDS.

    Pipeline:
      1. One shared per-electrode Conv1d at stride ``strides[0]``.
      2. One StageAdaptiveElectrodeMixer over all C electrodes.
      3. Flatten C x local_dim and project immediately to ``channels[0]``.
      4. Remaining strided convolutions + full-mix residual Signal-TDS blocks.

    Only steps 1-2 retain C streams.  The expensive deep frontend and all
    temporal transformers process one stream, keeping epoch speed close to
    EMGFormer / mixed-stem CycloFormer rather than full CycloFormer.

    Input/output: (B, C, T) -> (B, channels[-1], T').
    """

    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        n_residual_blocks: int = 2,
        dropout: float = 0.1,
        input_channels: int = 16,
        local_dim: int = 16,
        n_heads: int = 4,
        route_stride: int = 8,
        residual_init: float = 0.1,
        dynamic_attention: bool = True,
        signal_skip: bool = False,
    ):
        super().__init__()
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("channels, kernels, and strides must have equal length")
        if len(channels) < 2:
            raise ValueError("SAEMFrontend requires at least two stages")
        self.input_channels = int(input_channels)
        self.local_dim = int(local_dim)
        self.dynamic_attention = bool(dynamic_attention)
        self.signal_skip = bool(signal_skip)

        first_k = int(kernels[0]) if int(kernels[0]) % 2 else int(kernels[0]) + 1
        self.local_extract = SignalExtractConv(
            1,
            self.local_dim,
            kernel_size=first_k,
            stride=int(strides[0]),
            dropout=dropout,
        )
        self.electrode_mixer = SynergyAdaptiveElectrodeMixer(
            dim=self.local_dim,
            num_channels=self.input_channels,
            n_heads=n_heads,
            route_stride=route_stride,
            dropout=dropout,
            residual_init=residual_init,
        )
        self.collapse = SignalExtractConv(
            self.input_channels * self.local_dim,
            int(channels[0]),
            kernel_size=1,
            stride=1,
            dropout=dropout,
        )

        deep: list[nn.Module] = []
        in_dim = int(channels[0])
        # TDS at the collapsed first scale, then conv/TDS for later scales.
        for _ in range(int(n_residual_blocks)):
            deep.append(
                SignalTDSBlock(
                    in_dim, kernel_size=min(first_k, 9), dropout=dropout
                )
            )
        for out_dim, kernel, stride in zip(
            channels[1:], kernels[1:], strides[1:]
        ):
            k = int(kernel) if int(kernel) % 2 else int(kernel) + 1
            deep.append(
                SignalExtractConv(
                    in_dim,
                    int(out_dim),
                    kernel_size=k,
                    stride=int(stride),
                    dropout=dropout,
                )
            )
            for _ in range(int(n_residual_blocks)):
                deep.append(
                    SignalTDSBlock(
                        int(out_dim), kernel_size=min(k, 9), dropout=dropout
                    )
                )
            in_dim = int(out_dim)
        self.deep = nn.Sequential(*deep)
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} electrodes, got {C}")
        h = self.local_extract(x.reshape(B * C, 1, T))
        D, T1 = h.shape[1], h.shape[2]
        h = h.reshape(B, C, D, T1)
        if self.dynamic_attention:
            h = self.electrode_mixer(h)
        h = h.reshape(B, C * D, T1)
        h = self.collapse(h)
        skip = h
        h = self.deep(h)
        if self.signal_skip:
            skip = F.adaptive_avg_pool1d(skip, h.shape[-1])
            repeats = math.ceil(h.shape[1] / skip.shape[1])
            skip = skip.repeat(1, repeats, 1)[:, : h.shape[1], :]
            h = (h + skip) * (2.0 ** -0.5)
        return h


class CircularRingStem(nn.Module):
    """Rotation-equivariant ring×time frontend.

    Treats the 16 electrodes as a circular spatial axis and time as the
    other. One fused Conv2d per stage (not 16 independent 1-ch streams).
    Cyclic shift of the ring shifts the feature map; channel pooling
    after CRoPE is then rotation-invariant.

    Input:  (B, C, T)
    Output: (B, out_channels[-1], C, T')
    """

    def __init__(
        self,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        ring_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides)
        assert ring_kernel % 2 == 1, "ring_kernel must be odd for centred circular pad"
        self.ring_kernel = int(ring_kernel)
        stages = nn.ModuleList()
        in_ch = 1
        for out_ch, k_t, s_t in zip(channels, kernels, strides):
            stages.append(nn.ModuleDict(dict(
                conv=nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(self.ring_kernel, k_t),
                    stride=(1, s_t),
                    padding=0,
                    bias=False,
                ),
                bn=nn.BatchNorm2d(out_ch),
                act=nn.GELU(),
                drop=nn.Dropout(dropout),
            )))
            in_ch = out_ch
        self.stages = stages
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        pad_c = self.ring_kernel // 2
        for st in self.stages:
            k_t = int(st["conv"].kernel_size[1])
            pad_t = k_t // 2
            x = F.pad(x, (0, 0, pad_c, pad_c), mode="circular")
            x = F.pad(x, (pad_t, pad_t, 0, 0), mode="constant", value=0.0)
            x = st["drop"](st["act"](st["bn"](st["conv"](x))))
        return x


class CircularChannelMix(nn.Module):
    """Learned residual mix along the electrode ring.

    Input/output: (B, T', C, d). Circular pad on C, so a ring shift of
    the electrodes still just shifts the feature map.
    """

    def __init__(self, dim: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        assert kernel % 2 == 1, "circular mix kernel must be odd"
        self.kernel = int(kernel)
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=self.kernel, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape
        h = self.norm(x).reshape(B * T, C, D).permute(0, 2, 1)  # (B*T, d, C)
        pad = self.kernel // 2
        h = F.pad(h, (pad, pad), mode="circular")
        h = self.drop(self.act(self.bn(self.conv(h))))
        h = h.permute(0, 2, 1).reshape(B, T, C, D)
        return x + h


class SharedTemporalCNN(nn.Module):
    """
    Deep residual Conv1d applied identically to each channel.

    Architecture per stage:
      Strided Conv1d (downsample) + BN + GELU
      + N residual blocks (Conv-BN-GELU-Conv-BN + skip)

    in_channels = 1 for the stem → SAME weights for all 16 EMG channels,
    giving exact permutation equivariance.

    The deeper CNN learns richer temporal primitives (multi-scale MUAP
    shapes, burst patterns, firing-rate modulation, onset/offset envelopes)
    compared to the original shallow 4-layer version.

    Input:  (B*C, 1, T)
    Output: (B*C, out_channels[-1], T')
    """

    def __init__(self, channels: List[int], kernels: List[int],
                 strides: List[int], n_residual_blocks: int = 2,
                 use_cs_tds: bool = False,
                 dropout: float = 0.1,
                 in_channels: int = 1):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides)

        # Mixed stem (in_channels>1): EMGFormer-style TDS that mixes all
        # feature groups at every block. Per-electrode CNN (in_channels=1)
        # keeps ChannelSharedTDSBlock so the ring stays equivariant.
        if use_cs_tds:
            BlockClass = FullMixTDSBlock if int(in_channels) > 1 else ChannelSharedTDSBlock
        else:
            BlockClass = ResidualConvBlock

        stages = []
        in_ch = int(in_channels)
        for out_ch, k, s in zip(channels, kernels, strides):
            # Strided downsampling conv
            stage = [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s,
                          padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            # Inner blocks at this resolution
            for _ in range(n_residual_blocks):
                stage.append(BlockClass(
                    out_ch, kernel_size=min(k, 9 if use_cs_tds else 5),
                    dropout=dropout))
            stages.extend(stage)
            in_ch = out_ch

        self.net = nn.Sequential(*stages)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnsharedTemporalCNN(nn.Module):
    """
    Per-channel CNN: each of the C EMG channels has its OWN independent
    CNN (no weight sharing).

    Ablation counterpart to SharedTemporalCNN.  Use smaller cnn_channels
    so the total CNN parameter count stays comparable to the shared version
    (shared has N params; unshared-matched uses N/16 params per channel-CNN
    → total ≈ N).

    Input:  (B, C, T)
    Output: (B, C, d, T')  [different layout from SharedTemporalCNN]
    """

    def __init__(self, n_channels: int, channels: List[int], kernels: List[int],
                 strides: List[int], n_residual_blocks: int = 1,
                 use_cs_tds: bool = False, dropout: float = 0.1):
        super().__init__()
        self.cnns = nn.ModuleList([
            SharedTemporalCNN(channels, kernels, strides,
                              n_residual_blocks=n_residual_blocks,
                              use_cs_tds=use_cs_tds,
                              dropout=dropout)
            for _ in range(n_channels)
        ])
        self.out_channels = channels[-1]
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, d, T')"""
        outs = []
        for c, cnn in enumerate(self.cnns):
            x_c = x[:, c:c+1, :]   # (B, 1, T)
            outs.append(cnn(x_c))   # (B, d, T')
        return torch.stack(outs, dim=1)  # (B, C, d, T')


# ============================================================
# Optional: Gradient Reversal Layer (for stage de-biasing)
# ============================================================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class StageClassifier(nn.Module):
    """Auxiliary head predicting stage labels; gradient is reversed."""

    def __init__(self, d_model: int, num_stages: int, lam: float = 0.1):
        super().__init__()
        self.lam = lam
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_stages),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T', d_model) → (B, num_stages)"""
        x_rev = GradientReversal.apply(x.mean(dim=1), self.lam)
        return self.head(x_rev)


class SlowStateAdapter(nn.Module):
    """A tiny recurrent adapter that models slow temporal drift at low cost.

    We intentionally use GRUCell unrolling instead of nn.GRU here. Lightning's
    StochasticWeightAveraging callback deepcopies the full LightningModule in
    setup(), and the fused RNN modules can fail that deepcopy path on some
    PyTorch builds. The cell-based implementation keeps the GRU-style dynamics
    while remaining deepcopy-safe for SWA.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            [
                nn.GRUCell(d_model if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )
        self.proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_input = x
        batch_size, seq_len, _ = x.shape

        for layer_idx, cell in enumerate(self.cells):
            hidden = x.new_zeros(batch_size, self.hidden_dim)
            outputs = []
            for t in range(seq_len):
                hidden = cell(layer_input[:, t], hidden)
                outputs.append(hidden)
            layer_input = torch.stack(outputs, dim=1)
            if layer_idx < len(self.cells) - 1:
                layer_input = self.dropout(layer_input)

        h = self.dropout(self.proj(layer_input))
        return self.norm(x + h)


# ============================================================
# CS-TDS-CT++ experimental modules — toggled in config, default OFF
# ============================================================

# JOINT indices grouped by anatomical chain depth, matching emg2pose
# constants.JOINTS order. proximal = CMC of thumb + MCP of fingers (incl AA);
# mid = thumb MCP + PIP of fingers; distal = thumb IP + DIP of fingers.
_PROX_IDXS  = (0, 1, 4, 5, 8, 9, 12, 13, 16, 17)   # 10 joints
_MID_IDXS   = (2, 6, 10, 14, 18)                   # 5 joints
_DISTAL_IDXS = (3, 7, 11, 15, 19)                  # 5 joints
# Per-finger groups (used by per-finger queries readout if enabled).
_FINGER_JOINT_IDXS = (
    (0, 1, 2, 3),       # thumb
    (4, 5, 6, 7),       # index
    (8, 9, 10, 11),     # middle
    (12, 13, 14, 15),   # ring
    (16, 17, 18, 19),   # pinky
)


class AnatomyConditionedHead(nn.Module):
    """Autoregressive output head — proximal → mid → distal.

    Replaces the single 256→512→20 MLP head with three chained sub-heads,
    each conditioning on the predicted upstream angles. Forces the
    decoder to compute in a kinematically-meaningful order so distal
    predictions cannot quietly compensate for wrong proximal predictions
    (the silent failure mode of chain-rolled prediction). Self-conditioning:
    each level always uses its own predictions, train and inference alike
    — no teacher forcing, no exposure bias.

    Output reassembles into the canonical (B, T', 20) JOINT-index order
    so all downstream metrics keep working.
    """

    def __init__(self, d_model: int, d_ff: int, output_dim: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        assert output_dim == 20, "AnatomyConditionedHead assumes 20-joint output"
        self.output_dim = output_dim
        self.n_prox = len(_PROX_IDXS)
        self.n_mid  = len(_MID_IDXS)
        self.n_dist = len(_DISTAL_IDXS)

        # Sub-head 1: proximal — from features alone.
        self.head_prox = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, self.n_prox),
        )
        # Sub-head 2: mid — conditions on features + predicted proximal.
        self.head_mid = nn.Sequential(
            nn.LayerNorm(d_model + self.n_prox),
            nn.Linear(d_model + self.n_prox, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, self.n_mid),
        )
        # Sub-head 3: distal — conditions on features + proximal + mid.
        self.head_dist = nn.Sequential(
            nn.LayerNorm(d_model + self.n_prox + self.n_mid),
            nn.Linear(d_model + self.n_prox + self.n_mid, d_ff),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, self.n_dist),
        )

        # Pre-build long-tensor indexers for scatter-assembly.
        self.register_buffer("prox_idxs", torch.tensor(_PROX_IDXS, dtype=torch.long))
        self.register_buffer("mid_idxs",  torch.tensor(_MID_IDXS,  dtype=torch.long))
        self.register_buffer("dist_idxs", torch.tensor(_DISTAL_IDXS, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T', d_model) → out: (B, T', 20)
        prox = self.head_prox(x)
        mid  = self.head_mid(torch.cat([x, prox], dim=-1))
        dist = self.head_dist(torch.cat([x, prox, mid], dim=-1))

        out = x.new_zeros(*x.shape[:-1], self.output_dim)
        out.index_copy_(-1, self.prox_idxs, prox)
        out.index_copy_(-1, self.mid_idxs,  mid)
        out.index_copy_(-1, self.dist_idxs, dist)
        return out


class PerFingerAttentionPool(nn.Module):
    """Multi-query attention pooling — one learnable query per finger.

    Replaces the single global AttentionPooling query with `n_queries`
    queries, each producing finger-specific pooled features over the C
    channel positions. Concatenated and projected back to d_model so the
    downstream output head signature is unchanged.

    Permutation-equivariance: the queries themselves carry no channel
    bias, so the readout is still equivariant under cyclic channel
    rotation (consistent with CyRoPE's ring invariance).
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_queries: int = 5):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_proj = nn.Linear(n_queries * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BT', C, d_model) → (BT', d_model)"""
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)                  # (BT', n_q, d)
        return self.out_proj(pooled.flatten(1))         # (BT', d)


class MultiRateCNN(nn.Module):
    """Parallel fast + slow CNN branches, concatenated along feature dim.

    The fast branch matches the standard SharedTemporalCNN (~80× stride
    in default config). The slow branch keeps higher temporal resolution
    (~16× stride) so sub-25 Hz tonic activity — postural co-contractions
    governing wrist orientation — survives downsampling. Slow output is
    average-pooled in time down to the fast-branch length so they
    concatenate cleanly.

    Both branches are channel-shared (process (B*C, 1, T) input), so
    the channel-permutation-equivariance of the model is preserved.

    Output: (B*C, fast_d + slow_d, T_fast) — feeds the rest of the
    pipeline transparently because `out_channels` reports the sum.
    """

    def __init__(self,
                 fast_channels: List[int], fast_kernels: List[int],
                 fast_strides: List[int],
                 slow_channels: List[int], slow_kernels: List[int],
                 slow_strides: List[int],
                 n_residual_blocks: int = 1,
                 use_cs_tds: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.fast = SharedTemporalCNN(
            channels=fast_channels, kernels=fast_kernels, strides=fast_strides,
            n_residual_blocks=n_residual_blocks, use_cs_tds=use_cs_tds,
            dropout=dropout,
        )
        self.slow = SharedTemporalCNN(
            channels=slow_channels, kernels=slow_kernels, strides=slow_strides,
            n_residual_blocks=n_residual_blocks, use_cs_tds=use_cs_tds,
            dropout=dropout,
        )
        self.out_channels = fast_channels[-1] + slow_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast_out = self.fast(x)                          # (B*C, d_fast, T_fast)
        slow_out = self.slow(x)                          # (B*C, d_slow, T_slow)
        # Pool slow to fast's time resolution. Adaptive pool handles any
        # ratio cleanly; if T_slow < T_fast (shouldn't happen with sane
        # stride configs), interpolate up.
        if slow_out.shape[-1] != fast_out.shape[-1]:
            slow_out = F.adaptive_avg_pool1d(slow_out, fast_out.shape[-1])
        return torch.cat([fast_out, slow_out], dim=1)    # (B*C, d_total, T_fast)


# ============================================================
# Main Model
# ============================================================

class CSTDSCT(nn.Module):
    """
    Channel-Shared TDS + Circular Transformer (CS-TDS-CT).

    End-to-end pipeline:
        (B, C, T)
          → reshape (B*C, 1, T)
          → Deep Residual SharedCNN (shared weights → equivariant)
          → InstanceNorm per channel (user-amplitude invariance)
          → (B, T', C, d) → Linear(d → h_dim) (shared across channels)
          → N × [Spatial CyRoPE | Temporal RoPE]
          → AttentionPooling over channels
          → MLP head → (B, output_dim, T')
    """

    def __init__(self, config: CSTDSConfig):
        super().__init__()
        self.config = config
        C = config.input_channels

        # ── 1. Temporal CNN (circular stem / shared / unshared / multi-rate)
        if config.use_saem_frontend:
            if config.mix_emg_channels_at_stem:
                raise ValueError(
                    "use_saem_frontend performs its own early channel collapse "
                    "and is mutually exclusive with mix_emg_channels_at_stem"
                )
            if (
                config.use_acsa_frontend
                or config.use_circular_stem
                or config.use_multirate_frontend
            ):
                raise ValueError(
                    "use_saem_frontend is mutually exclusive with ACSA, "
                    "circular, and multi-rate frontends"
                )
            self.cnn = SAEMFrontend(
                channels=config.cnn_channels,
                kernels=config.cnn_kernels,
                strides=config.cnn_strides,
                n_residual_blocks=config.n_residual_blocks,
                dropout=config.dropout,
                input_channels=C,
                local_dim=config.saem_local_dim,
                n_heads=config.saem_heads,
                route_stride=config.saem_route_stride,
                residual_init=config.saem_residual_init,
                dynamic_attention=config.saem_dynamic_attention,
                signal_skip=config.saem_signal_skip,
            )
        elif config.use_acsa_frontend:
            if config.mix_emg_channels_at_stem:
                raise ValueError(
                    "use_acsa_frontend preserves an explicit electrode axis and "
                    "is mutually exclusive with mix_emg_channels_at_stem"
                )
            if config.use_circular_stem or config.use_multirate_frontend:
                raise ValueError(
                    "use_acsa_frontend is mutually exclusive with circular and "
                    "multi-rate frontends"
                )
            self.cnn = ACSAFrontend(
                channels=config.cnn_channels,
                kernels=config.cnn_kernels,
                strides=config.cnn_strides,
                n_residual_blocks=config.n_residual_blocks,
                dropout=config.dropout,
                input_channels=C,
                n_heads=config.acsa_heads,
                router_dim=config.acsa_router_dim,
                route_strides=config.acsa_route_strides,
                residual_init=config.acsa_residual_init,
                dynamic_edges=config.acsa_dynamic_edges,
            )
        elif config.use_circular_stem:
            if config.use_multirate_frontend:
                raise ValueError("use_circular_stem is mutually exclusive with use_multirate_frontend")
            self.cnn = CircularRingStem(
                channels=config.cnn_channels,
                kernels=config.cnn_kernels,
                strides=config.cnn_strides,
                ring_kernel=config.circular_ring_kernel,
                dropout=config.dropout,
            )
        elif config.use_multirate_frontend:
            assert config.share_cnn, (
                "use_multirate_frontend requires share_cnn=True; the "
                "parallel slow branch only makes sense with channel-shared "
                "weights so equivariance is preserved across both rates."
            )
            self.cnn = MultiRateCNN(
                fast_channels=config.cnn_channels,
                fast_kernels=config.cnn_kernels,
                fast_strides=config.cnn_strides,
                slow_channels=config.multirate_slow_channels,
                slow_kernels=config.multirate_slow_kernels,
                slow_strides=config.multirate_slow_strides,
                n_residual_blocks=config.n_residual_blocks,
                use_cs_tds=config.use_cs_tds,
                dropout=config.dropout,
            )
        elif config.mix_emg_channels_at_stem:
            if config.use_circular_stem or config.use_multirate_frontend:
                raise ValueError(
                    "mix_emg_channels_at_stem is mutually exclusive with "
                    "use_circular_stem and use_multirate_frontend"
                )
            if getattr(config, "use_signal_tds_frontend", False):
                self.cnn = SignalTDSFrontend(
                    channels=config.cnn_channels,
                    kernels=config.cnn_kernels,
                    strides=config.cnn_strides,
                    n_residual_blocks=config.n_residual_blocks,
                    dropout=config.dropout,
                    in_channels=C,
                )
            else:
                self.cnn = SharedTemporalCNN(
                    channels=config.cnn_channels,
                    kernels=config.cnn_kernels,
                    strides=config.cnn_strides,
                    n_residual_blocks=config.n_residual_blocks,
                    use_cs_tds=config.use_cs_tds,
                    dropout=config.dropout,
                    in_channels=C,
                )
        elif config.share_cnn:
            self.cnn = SharedTemporalCNN(
                channels=config.cnn_channels,
                kernels=config.cnn_kernels,
                strides=config.cnn_strides,
                n_residual_blocks=config.n_residual_blocks,
                use_cs_tds=config.use_cs_tds,
                dropout=config.dropout,
            )
        else:
            self.cnn = UnsharedTemporalCNN(
                n_channels=C,
                channels=config.cnn_channels,
                kernels=config.cnn_kernels,
                strides=config.cnn_strides,
                n_residual_blocks=config.n_residual_blocks,
                use_cs_tds=config.use_cs_tds,
                dropout=config.dropout,
            )
        cnn_out_dim = self.cnn.out_channels  # d

        # ── 1b. LayerNorm after CNN features ─────────────────────────────
        # LayerNorm over feature dim has ZERO train/eval mismatch (unlike InstanceNorm
        # which uses running stats at eval but instance stats at train).
        # Normalises each (B*C, T') position over d features independently.
        self.feature_norm = nn.LayerNorm(cnn_out_dim)
        if config.use_circular_channel_mix:
            self.circular_channel_mix = CircularChannelMix(
                cnn_out_dim,
                kernel=config.circular_channel_mix_kernel,
                dropout=config.dropout,
            )
        else:
            self.circular_channel_mix = None
        self.use_encoder_skip = bool(config.use_encoder_skip)

        # ── 2. Projection to Transformer dim (shared across channels) ─────
        self.input_projection = nn.Linear(cnn_out_dim, config.d_model)

        # ── 3. Factored transformer blocks (alternating spatial/temporal) ─
        self.blocks = nn.ModuleList()
        n_s, n_t = 0, 0

        # Resolve rope_mode. The legacy `use_legacy_attention` flag is mapped
        # to "legacy_mha" so older configs keep working.
        rope_mode = config.rope_mode
        if config.use_legacy_attention and rope_mode == "fixed":
            rope_mode = "legacy_mha"

        spatial_cls_by_mode = {
            "fixed":                    SpatialAttentionBlock,
            "legacy_mha":               LegacySpatialAttentionBlock,
            "legacy_v2_pre_transpose":  LegacyV2PreTransposeSpatialAttentionBlock,
            "legacy_v2_pre_proj":       LegacyV2PreProjSpatialAttentionBlock,
        }
        temporal_cls_by_mode = {
            "fixed":                    TemporalAttentionBlock,
            "legacy_mha":               LegacyTemporalAttentionBlock,
            "legacy_v2_pre_transpose":  LegacyV2PreTransposeTemporalAttentionBlock,
            "legacy_v2_pre_proj":       LegacyV2PreProjTemporalAttentionBlock,
        }
        if rope_mode not in spatial_cls_by_mode:
            raise ValueError(
                f"Unknown rope_mode={rope_mode!r}. "
                f"Valid options: {sorted(spatial_cls_by_mode)}"
            )
        SpatialCls  = spatial_cls_by_mode[rope_mode]
        TemporalCls = temporal_cls_by_mode[rope_mode]
        self.spatial_at_stem_dim = bool(config.spatial_at_stem_dim)
        if self.spatial_at_stem_dim:
            spat_d = cnn_out_dim
            spat_heads = config.n_heads
            if spat_d % spat_heads != 0:
                spat_heads = 4 if spat_d % 4 == 0 else 2
            spat_ff = max(spat_d * 2, 64)
            pool_d = spat_d
        else:
            spat_d = config.d_model
            spat_heads = config.n_heads
            spat_ff = config.d_ff
            pool_d = config.d_model
        if config.use_channel_embeddings:
            self.channel_embedding = nn.Parameter(torch.empty(1, 1, C, spat_d))
            nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        else:
            self.channel_embedding = None
        for _ in range(config.n_spatial_blocks + config.n_temporal_blocks):
            if n_s < config.n_spatial_blocks:
                self.blocks.append(SpatialCls(
                    spat_d, spat_heads, spat_ff,
                    num_channels=C,
                    dropout=config.dropout,
                    use_circular_rope=config.use_circular_spatial_rope,
                ))
                n_s += 1
            if n_t < config.n_temporal_blocks:
                self.blocks.append(TemporalCls(
                    config.d_model, config.n_heads, config.d_ff,
                    dropout=config.dropout))
                n_t += 1

        # ── 4. Attention pooling over channels ────────────────────────────
        # Three modes (mutually exclusive, ordered by precedence):
        #   use_per_finger_queries=True  →  multi-query (B.6 toggle)
        #   use_attention_pooling=True   →  single learnable query (default)
        #   neither                       →  mean-pool fallback
        # DEFERRED: concat TDS/CNN features onto the sequence immediately
        # before this pooling step, then MLP. Not wired yet.
        if config.use_per_finger_queries:
            self.channel_pooling = PerFingerAttentionPool(
                pool_d,
                n_heads=spat_heads,
                n_queries=config.n_finger_queries,
            )
        elif config.use_attention_pooling:
            self.channel_pooling = AttentionPooling(pool_d, spat_heads)
        else:
            self.channel_pooling = None  # fallback: mean pooling

        # ── 4b. Slow temporal state adapter ───────────────────────────────
        slow_state_type = config.slow_state_type.lower()
        if slow_state_type == "none":
            self.slow_state = None
        elif slow_state_type == "gru":
            hidden_dim = config.slow_state_hidden_dim or max(config.d_model // 2, 64)
            self.slow_state = SlowStateAdapter(
                d_model=config.d_model,
                hidden_dim=hidden_dim,
                num_layers=config.slow_state_layers,
                dropout=config.slow_state_dropout,
            )
        else:
            raise ValueError(f"Unsupported slow_state_type={config.slow_state_type!r}")

        # ── 5. Output head ────────────────────────────────────────────────
        # Two flavours, chosen by use_anatomy_head:
        #   - Default: flat MLP over pooled features → 20 angles in one shot
        #   - Anatomy-conditioned: predicts proximal → mid → distal in
        #     three chained sub-heads (B.5 toggle)
        # Both consume (B, T', d_model) and emit (B, T', output_dim).
        if config.use_anatomy_head:
            self.output_head = AnatomyConditionedHead(
                d_model=config.d_model,
                d_ff=config.d_ff,
                output_dim=config.output_dim,
                dropout=config.dropout,
            )
        else:
            self.output_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.output_dim),
            )

        if config.use_tds_skip:
            self.tds_skip_fuse = nn.Sequential(
                nn.LayerNorm(2 * config.d_model),
                nn.Linear(2 * config.d_model, config.d_model),
            )
        else:
            self.tds_skip_fuse = None

        # ── 6. Optional stage classifier with gradient reversal ──────────
        if config.use_gradient_reversal and config.num_stages > 1:
            self.stage_classifier = StageClassifier(
                config.d_model, config.num_stages, config.stage_reversal_lambda)
        else:
            self.stage_classifier = None

        # Compatibility with emg2pose training loop
        self.left_context = 0
        self.right_context = 0
        self.keep_encoder_rate = bool(getattr(config, "keep_encoder_rate", False))
        self.mix_emg_channels_at_stem = bool(
            getattr(config, "mix_emg_channels_at_stem", False)
            or getattr(config, "use_saem_frontend", False)
        )

        self._has_printed = False
        self._latest_aux_losses: dict[str, torch.Tensor] = {}

    @property
    def output_size(self) -> int:
        return self.config.output_dim

    # ------------------------------------------------------------------
    def _apply_channel_rotation(
        self,
        x: torch.Tensor,
        rotation_range: Optional[int] = None,
        training_only: bool = True,
    ) -> torch.Tensor:
        """Cyclic channel rotation augmentation (training only)."""
        if training_only and not self.training:
            return x
        r = self.config.channel_rotation_range if rotation_range is None else rotation_range
        if r <= 0:
            return x
        shift = torch.randint(-r, r + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=1)

    # ------------------------------------------------------------------
    def _apply_channel_mix(
        self,
        x: torch.Tensor,
        mix_strength: float,
        training_only: bool = True,
    ) -> torch.Tensor:
        """Mix each channel with its ring neighbours to simulate mild re-donning."""
        if training_only and not self.training:
            return x
        if mix_strength <= 0:
            return x
        alpha = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) * mix_strength
        neighbours = 0.5 * (torch.roll(x, shifts=1, dims=1) + torch.roll(x, shifts=-1, dims=1))
        return (1.0 - alpha) * x + alpha * neighbours

    # ------------------------------------------------------------------
    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero entire channels (training only).

        Forces the model to predict from incomplete electrode subsets, preventing
        the CNN from memorising cross-channel co-activation patterns specific to
        training sessions.  Each channel is dropped independently.
        """
        if not self.training or self.config.channel_drop_p <= 0:
            return x
        B, C, T = x.shape
        # Bernoulli mask: 1 = keep, 0 = drop  (broadcast over time)
        keep = torch.bernoulli(
            torch.full((B, C, 1), 1.0 - self.config.channel_drop_p, device=x.device)
        )
        return x * keep

    # ------------------------------------------------------------------
    def _apply_amplitude_jitter(
        self,
        x: torch.Tensor,
        jitter_std: Optional[float] = None,
        training_only: bool = True,
    ) -> torch.Tensor:
        """Per-channel log-normal amplitude jitter (training only).

        Simulates inter-user and inter-session amplitude variation caused by
        differences in muscle volume, electrode contact quality, and skin
        impedance.  Using log-normal (gain = exp(N(0, σ))) ensures:
          - gain is always positive (amplitude cannot flip)
          - gain is centered around 1.0 (median, not mean)
          - gain is asymmetric: slight positive skew matches real EMG variation

        Applied per (batch, channel) independently, broadcast over time.
        """
        if training_only and not self.training:
            return x
        jitter_std = (
            self.config.amplitude_jitter_std if jitter_std is None else jitter_std
        )
        if jitter_std <= 0:
            return x
        B, C, T = x.shape
        log_gain = jitter_std * torch.randn(B, C, 1, device=x.device)
        gain = torch.exp(log_gain)   # log-normal: always > 0, median = 1.0
        return x * gain

    # ------------------------------------------------------------------
    def _apply_temporal_chunk_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out contiguous temporal chunks (training only)."""
        if not self.training or self.config.chunk_mask_ratio <= 0:
            return x
        B, C, T = x.shape
        mask_len = int(T * self.config.chunk_mask_ratio)
        start = torch.randint(0, T - mask_len, (B,))
        mask = torch.ones(B, 1, T, device=x.device)
        for i in range(B):
            mask[i, :, start[i]:start[i] + mask_len] = 0.0
        return x * mask

    # ------------------------------------------------------------------
    def _apply_log_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Optional signed log compression to reduce user-specific amplitude spread."""
        if not self.config.use_log_compression:
            return x
        return torch.sign(x) * torch.log1p(x.abs())

    # ------------------------------------------------------------------
    def _apply_training_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_channel_rotation(x)
        x = self._apply_channel_mix(x, self.config.channel_mix_strength)
        x = self._apply_amplitude_jitter(x)
        x = self._apply_channel_dropout(x)
        x = self._apply_temporal_chunk_mask(x)
        x = self._apply_frequency_mask(x)
        x = self._apply_gaussian_noise(x)
        return x

    def _apply_frequency_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Batched rFFT band drop. Same SpecAugment as EMGFormer, on GPU."""
        n_masks = int(self.config.freq_mask_num)
        max_size = int(self.config.freq_mask_max)
        if not self.training or n_masks <= 0 or max_size <= 0:
            return x
        B, _, T = x.shape
        spec = torch.fft.rfft(x, dim=-1)
        n_freq = spec.shape[-1]
        freqs = torch.arange(n_freq, device=x.device)
        for _ in range(n_masks):
            width = torch.randint(1, min(max_size, n_freq) + 1, (B, 1, 1), device=x.device)
            start_hi = (n_freq - width.squeeze(-1).squeeze(-1)).clamp(min=1)
            start = (torch.rand(B, 1, 1, device=x.device) * start_hi.view(B, 1, 1).float()).long()
            spec = spec.masked_fill((freqs >= start) & (freqs < start + width), 0)
        return torch.fft.irfft(spec, n=T, dim=-1)

    def _apply_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window SNR noise on GPU, matching EMGFormer aug_best."""
        prob = float(self.config.gaussian_noise_prob)
        if not self.training or prob <= 0:
            return x
        B = x.shape[0]
        power = x.square().mean(dim=(1, 2), keepdim=True)
        snr = torch.empty(B, 1, 1, device=x.device).uniform_(
            float(self.config.gaussian_noise_min_snr_db),
            float(self.config.gaussian_noise_max_snr_db),
        )
        noise_std = (power / torch.pow(10.0, snr / 10.0)).sqrt()
        apply = torch.rand(B, 1, 1, device=x.device) < prob
        return x + torch.randn_like(x) * noise_std * apply

    # ------------------------------------------------------------------
    def _sample_consistency_view(self, x: torch.Tensor) -> torch.Tensor:
        rotation_range = self.config.consistency_rotation_range
        view = self._apply_channel_rotation(
            x,
            rotation_range=rotation_range,
            training_only=False,
        )
        mix_strength = self.config.consistency_channel_mix_strength
        if mix_strength <= 0:
            mix_strength = self.config.channel_mix_strength
        view = self._apply_channel_mix(
            view,
            mix_strength=mix_strength,
            training_only=False,
        )
        jitter_std = self.config.consistency_amplitude_jitter_std
        if jitter_std <= 0:
            jitter_std = self.config.amplitude_jitter_std
        view = self._apply_amplitude_jitter(
            view,
            jitter_std=jitter_std,
            training_only=False,
        )
        return view

    # ------------------------------------------------------------------
    def _forward_backbone(
        self,
        x: torch.Tensor,
        should_print: bool = False,
    ) -> torch.Tensor:
        B, C, T = x.shape
        input_std = max(float(self.config.input_std), 1e-6)
        if self.config.input_mean != 0.0 or input_std != 1.0:
            x = (x - float(self.config.input_mean)) / input_std
        x = self._apply_log_compression(x)

        # Mixed first conv (EMGFormer TDS layer-1): (B, C, T) → (B, d, T').
        if self.mix_emg_channels_at_stem:
            features = self.cnn(x)
            d, T_prime = features.shape[1], features.shape[2]
            features = features.permute(0, 2, 1).contiguous()
            features = self.feature_norm(features)
            features = self.input_projection(features)
            if should_print:
                total_stride = max(T // max(T_prime, 1), 1)
                if isinstance(self.cnn, SAEMFrontend):
                    frontend_name = "SAEM"
                    frontend_detail = (
                        ", dynamic electrode attention + Signal-TDS"
                        if self.cnn.dynamic_attention
                        else ", static channel collapse + Signal-TDS"
                    )
                elif isinstance(self.cnn, SignalTDSFrontend):
                    frontend_name = "SignalTDS"
                    frontend_detail = ", LN+GELU TDS"
                else:
                    frontend_name = "MixedStemCNN"
                    frontend_detail = ""
                print(
                    f"  {frontend_name}:"
                    f" ({B}, {C}, {T}) → ({B}, {d}, {T_prime})  "
                    f"[{total_stride}× downsample, channels mixed"
                    f"{frontend_detail}]"
                )
            pooled = features
            encoder_skip = pooled if self.use_encoder_skip else None
            for i, block in enumerate(self.blocks):
                if isinstance(block, _TemporalBlockBase):
                    pooled = block(pooled)
                    if should_print and i < 2:
                        print(
                            f"  Temporal block after mix-stem: "
                            f"(B={B}, T'={T_prime}, h={pooled.shape[-1]})"
                        )
            if encoder_skip is not None:
                pooled = pooled + encoder_skip
            if self.slow_state is not None:
                pooled = self.slow_state(pooled)
            output = self.output_head(pooled).transpose(1, 2)
            if should_print:
                print(f"  Output: {tuple(output.shape)}")
                params = sum(p.numel() for p in self.parameters())
                print(f"  Total params: {params:,}")
                self._has_printed = True
            return output, pooled

        # ── 1. Temporal CNN (circular stem / shared / unshared) ───────────
        if isinstance(self.cnn, (CircularRingStem, ACSAFrontend)):
            features = self.cnn(x)                     # (B, d, C, T')
            d, T_prime = features.shape[1], features.shape[3]
            features = features.permute(0, 3, 2, 1).contiguous()  # (B, T', C, d)
            features = self.feature_norm(features)
            if self.circular_channel_mix is not None:
                features = self.circular_channel_mix(features)
            if should_print:
                total_stride = max(T // max(T_prime, 1), 1)
                mix = " + circular mix" if self.circular_channel_mix is not None else ""
                stem_name = (
                    "ACSAFrontend"
                    if isinstance(self.cnn, ACSAFrontend)
                    else "CircularRingStem"
                )
                print(f"  {stem_name}{mix}: ({B}, {C}, {T}) → ({B}, {d}, {C}, {T_prime})  "
                      f"[{total_stride}× downsample]")
        elif isinstance(self.cnn, UnsharedTemporalCNN):
            # Each channel has its own CNN: (B, C, T) → (B, C, d, T')
            features = self.cnn(x)
            d, T_prime = features.shape[2], features.shape[3]
            features = features.permute(0, 1, 3, 2).reshape(B * C, T_prime, d)
        else:
            # Shared CNN: all channels processed by the same weights
            x_flat = x.reshape(B * C, 1, T)
            features = self.cnn(x_flat)                # (B*C, d, T')
            d, T_prime = features.shape[1], features.shape[2]
            features = features.permute(0, 2, 1)       # (B*C, T', d)

        if not isinstance(self.cnn, (CircularRingStem, ACSAFrontend)):
            # Both 1-ch-CNN paths now: features → (B*C, T', d)
            if should_print:
                total_stride = T // T_prime
                cnn_label = "UnsharedCNN" if isinstance(self.cnn, UnsharedTemporalCNN) else "SharedCNN"
                print(f"  {cnn_label}: ({B}×{C}, 1, {T}) → ({B}×{C}, {d}, {T_prime})  "
                      f"[{total_stride}× downsample]")

            # ── 2. LayerNorm over feature dim (zero train/eval mismatch) ──
            features = self.feature_norm(features)          # (B*C, T', d)
            features = features.reshape(B, C, T_prime, d)
            features = features.permute(0, 2, 1, 3)        # (B, T', C, d)
            if self.circular_channel_mix is not None:
                features = self.circular_channel_mix(features)

            if should_print:
                print(f"  LayerNorm → ({B}, {T_prime}, {C}, {d})")

        # ── 3. Project to Transformer dimension (shared) ──────────────────
        # Linear(d → h_dim) applied identically per channel, unless CRoPE
        # runs at stem width (spatial_at_stem_dim) — then project after pool.
        h = self.config.d_model
        stem_spatial = bool(getattr(self, "spatial_at_stem_dim", False))
        if not stem_spatial:
            features = self.input_projection(features)     # (B, T', C, h_dim)
        tds_tokens = features if self.tds_skip_fuse is not None else None
        temporal_after_pool = bool(self.config.temporal_after_channel_pool)
        spatial_h = features.shape[-1]

        if self.channel_embedding is not None:
            features = features + self.channel_embedding.to(
                dtype=features.dtype, device=features.device
            )

        if should_print:
            print(f"  tokens: ({B}, {T_prime}, {C}, {spatial_h})"
                  + (" [CRoPE at stem dim]" if stem_spatial else f" [projected {h}]"))
            if self.channel_embedding is not None:
                print("  channel embeddings: learned electrode IDs (no cyclic invariance)")
            if temporal_after_pool:
                print("  temporal_after_channel_pool: spatial → pool → temporal (B, T', d)")

        def _apply_spatial(feat: torch.Tensor) -> torch.Tensor:
            dh = feat.shape[-1]
            feat = feat.reshape(B * T_prime, C, dh)
            # SDPA's efficient kernel rejects batch > 65535 (B*T' at
            # Circ-S T'≈156 overflows around batch 420). Chunk along BT.
            max_bt = 32768
            spatial_blocks = [b for b in self.blocks if isinstance(b, _SpatialBlockBase)]
            if not spatial_blocks:
                return feat.reshape(B, T_prime, C, dh)
            use_ckpt = bool(getattr(self.config, "use_spatial_checkpoint", False)) and self.training

            def _run_blocks(tokens: torch.Tensor) -> torch.Tensor:
                for block in spatial_blocks:
                    if use_ckpt:
                        tokens = torch.utils.checkpoint.checkpoint(
                            block, tokens, use_reentrant=False
                        )
                    else:
                        tokens = block(tokens)
                return tokens

            if feat.shape[0] <= max_bt:
                feat = _run_blocks(feat)
            else:
                chunks = []
                for i in range(0, feat.shape[0], max_bt):
                    chunks.append(_run_blocks(feat[i:i + max_bt]))
                feat = torch.cat(chunks, dim=0)
            return feat.reshape(B, T_prime, C, dh)

        if temporal_after_pool:
            # Keep CRoPE / ring mixing, then one time-axis transformer like EMGFormer.
            features = _apply_spatial(features)
            if self.tds_skip_fuse is not None and tds_tokens is not None and not stem_spatial:
                features = self.tds_skip_fuse(torch.cat([features, tds_tokens], dim=-1))
                if should_print:
                    print(f"  TDS skip fuse: ({B}, {T_prime}, {C}, {h})")
            if self.channel_pooling is not None:
                pooled = self.channel_pooling(features.reshape(B * T_prime, C, features.shape[-1]))
                pooled = pooled.reshape(B, T_prime, features.shape[-1])
            else:
                pooled = features.mean(dim=2)
            if stem_spatial:
                pooled = self.input_projection(pooled)  # (B, T', d_model)
            if should_print:
                print(f"  Channel pool: ({B}, {T_prime}, {pooled.shape[-1]})")
            encoder_skip = pooled if self.use_encoder_skip else None
            for i, block in enumerate(self.blocks):
                if isinstance(block, _TemporalBlockBase):
                    pooled = block(pooled)
                    if should_print and i < 2:
                        print(f"  Temporal block after pool: (B={B}, T'={T_prime}, h={pooled.shape[-1]})")
            if encoder_skip is not None:
                pooled = pooled + encoder_skip
                if should_print:
                    print(f"  Encoder skip add: ({B}, {T_prime}, {pooled.shape[-1]})")
        else:
            # ── 4. Factored spatial-temporal attention ────────────────────────
            for i, block in enumerate(self.blocks):
                if isinstance(block, _SpatialBlockBase):
                    # (B, T', C, h) → (B*T', C, h) → spatial → (B, T', C, h)
                    features = features.reshape(B * T_prime, C, h)
                    features = block(features)
                    features = features.reshape(B, T_prime, C, h)
                    if should_print and i < 2:
                        print(f"  Spatial block {i}: (B×T'={B*T_prime}, C={C}, h={h})")

                elif isinstance(block, _TemporalBlockBase):
                    # (B, T', C, h) → (B*C, T', h) → temporal → (B, T', C, h)
                    features = features.permute(0, 2, 1, 3).reshape(B * C, T_prime, h)
                    features = block(features)
                    features = features.reshape(B, C, T_prime, h).permute(0, 2, 1, 3)
                    if should_print and i < 2:
                        print(f"  Temporal block {i}: (B×C={B*C}, T'={T_prime}, h={h})")

            if self.tds_skip_fuse is not None and tds_tokens is not None:
                features = self.tds_skip_fuse(torch.cat([features, tds_tokens], dim=-1))
                if should_print:
                    print(f"  TDS skip fuse: ({B}, {T_prime}, {C}, {h})")

            # ── 5. Channel aggregation ────────────────────────────────────────
            # (B, T', C, h) → (B, T', h)
            if self.channel_pooling is not None:
                pooled = features.reshape(B * T_prime, C, h)
                pooled = self.channel_pooling(pooled)       # (B*T', h)
                pooled = pooled.reshape(B, T_prime, h)
            else:
                pooled = features.mean(dim=2)

            if should_print:
                print(f"  Channel pool: ({B}, {T_prime}, {h})")

        # ── 6. Slow temporal state ───────────────────────────────────────
        if self.slow_state is not None:
            pooled = self.slow_state(pooled)
            if should_print:
                print(f"  Slow state: ({B}, {T_prime}, {h})")

        # ── 7. MLP head ──────────────────────────────────────────────────
        output = self.output_head(pooled)               # (B, T', output_dim)

        # Transpose to (B, output_dim, T') for emg2pose compatibility
        output = output.transpose(1, 2)

        if should_print:
            print(f"  Output: {tuple(output.shape)}")
            params = sum(p.numel() for p in self.parameters())
            print(f"  Total params: {params:,}")
            self._has_printed = True

        return output, pooled   # also return pooled for encode()

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features without the output head.

        Returns pooled: (B, T', d_model)
        Suitable for downstream classification after temporal mean pooling.
        """
        aug = self._apply_training_augmentations(x) if self.training else x
        _, pooled = self._forward_backbone(aug)
        return pooled   # (B, T', d_model)

    # ------------------------------------------------------------------
    def consume_aux_losses(self) -> dict[str, torch.Tensor]:
        aux_losses = self._latest_aux_losses
        self._latest_aux_losses = {}
        return aux_losses

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — raw 16-channel sEMG at 2 kHz
        Returns:
            output: (B, output_dim, T') — per-frame joint angle predictions
        """
        should_print = verbose or (not self._has_printed and self.training)
        if should_print:
            B, C, T = x.shape
            print(f"\n[CS-TDS-CT] Forward pass:")
            print(f"  Input: ({B}, {C}, {T})")
            if self.config.consistency_enabled:
                print(
                    "  Consistency: enabled "
                    f"(rot={self.config.consistency_rotation_range or self.config.channel_rotation_range}, "
                    f"mix={self.config.consistency_channel_mix_strength or self.config.channel_mix_strength:.2f}, "
                    f"amp={self.config.consistency_amplitude_jitter_std or self.config.amplitude_jitter_std:.2f})"
                )

        self._latest_aux_losses = {}

        primary_input = self._apply_training_augmentations(x)
        output, pooled = self._forward_backbone(primary_input, should_print=should_print)

        if self.training and self.config.consistency_enabled:
            aux_input = self._sample_consistency_view(x)
            aux_output, aux_pooled = self._forward_backbone(aux_input, should_print=False)
            mode = self.config.consistency_mode
            if mode == "smooth_l1":
                # Output-space agreement — current default
                self._latest_aux_losses["consistency"] = F.smooth_l1_loss(output, aux_output)
            elif mode == "infonce":
                # Feature-space contrastive (SimCLR-style) — primary view of
                # each sample must be closer to its own aux view than to
                # other samples'. Strong invariance pressure on the encoder.
                self._latest_aux_losses["consistency"] = self._infonce_loss(pooled, aux_pooled)
            else:
                raise ValueError(
                    f"Unknown consistency_mode={mode!r}. "
                    "Valid: 'smooth_l1', 'infonce'."
                )

        return output

    # ------------------------------------------------------------------
    def _infonce_loss(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE on time-pooled feature embeddings.

        feats_a, feats_b: (B, T', d_model). Pool over T', L2-normalise, then
        compute the standard SimCLR objective. Treats every other sample in
        the batch as a negative — requires batch_size > 1 to be informative.
        Returns a scalar smooth-L1-comparable loss.
        """
        if feats_a.shape[0] < 2:
            # Single-sample fallback — InfoNCE needs negatives. Behave like
            # smooth-L1 so loss is still well-defined for smoke tests.
            return F.smooth_l1_loss(feats_a, feats_b)
        a = F.normalize(feats_a.mean(dim=1), dim=-1)
        b = F.normalize(feats_b.mean(dim=1), dim=-1)
        tau = max(self.config.infonce_tau, 1e-4)
        logits = (a @ b.t()) / tau                       # (B, B)
        labels = torch.arange(a.size(0), device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))


# ============================================================
# Factory Function (Hydra-compatible)
# ============================================================

def create_cs_tds_ct(
    # Input / Output
    input_channels: int = 16,
    output_dim: int = 20,
    # CNN — shared or unshared
    share_cnn: bool = True,             # False → per-channel independent CNN (ablation)
    cnn_channels: Optional[List[int]] = None,
    cnn_kernels: Optional[List[int]] = None,
    cnn_strides: Optional[List[int]] = None,
    n_residual_blocks: int = 1,
    use_cs_tds: bool = False,           # True → ChannelSharedTDSBlock, False → ResidualConvBlock
    # Transformer
    d_model: int = 256,
    n_spatial_blocks: int = 2,
    n_temporal_blocks: int = 4,
    n_heads: int = 8,
    d_ff: int = 512,
    dropout: float = 0.1,
    use_circular_spatial_rope: bool = True,
    use_channel_embeddings: bool = False,
    # Pooling
    use_attention_pooling: bool = True,
    use_log_compression: bool = False,
    # Augmentation
    channel_rotation_range: int = 8,
    channel_mix_strength: float = 0.0,
    channel_drop_p: float = 0.10,
    amplitude_jitter_std: float = 0.15,
    chunk_mask_ratio: float = 0.0,
    freq_mask_num: int = 0,
    freq_mask_max: int = 128,
    gaussian_noise_prob: float = 0.0,
    gaussian_noise_min_snr_db: float = 25.0,
    gaussian_noise_max_snr_db: float = 35.0,
    input_mean: float = 0.0,
    input_std: float = 1.0,
    # Slow temporal state
    slow_state_type: str = "none",
    slow_state_hidden_dim: int = 0,
    slow_state_layers: int = 1,
    slow_state_dropout: float = 0.1,
    # Invariance consistency
    consistency_enabled: bool = False,
    consistency_rotation_range: Optional[int] = None,
    consistency_channel_mix_strength: float = 0.0,
    consistency_amplitude_jitter_std: float = 0.0,
    # Stage de-biasing
    use_gradient_reversal: bool = False,
    num_stages: int = 1,
    stage_reversal_lambda: float = 0.1,
    # Legacy attention (pre-RoPE-bug-fix checkpoints)
    use_legacy_attention: bool = False,
    rope_mode: str = "fixed",
    # CS-TDS-CT++ experimental architectural toggles (all default OFF)
    use_anatomy_head: bool = False,
    use_per_finger_queries: bool = False,
    n_finger_queries: int = 5,
    use_multirate_frontend: bool = False,
    use_tds_skip: bool = False,
    temporal_after_channel_pool: bool = False,
    use_circular_stem: bool = False,
    circular_ring_kernel: int = 3,
    use_circular_channel_mix: bool = False,
    circular_channel_mix_kernel: int = 5,
    use_encoder_skip: bool = False,
    spatial_at_stem_dim: bool = False,
    use_spatial_checkpoint: bool = False,
    keep_encoder_rate: bool = False,
    mix_emg_channels_at_stem: bool = False,
    use_signal_tds_frontend: bool = False,
    use_acsa_frontend: bool = False,
    acsa_heads: int = 4,
    acsa_router_dim: int = 64,
    acsa_route_strides: Optional[List[int]] = None,
    acsa_residual_init: float = 0.1,
    acsa_dynamic_edges: bool = True,
    use_saem_frontend: bool = False,
    saem_local_dim: int = 16,
    saem_heads: int = 4,
    saem_route_stride: int = 8,
    saem_residual_init: float = 0.1,
    saem_dynamic_attention: bool = True,
    saem_signal_skip: bool = False,
    multirate_slow_channels: Optional[List[int]] = None,
    multirate_slow_kernels: Optional[List[int]] = None,
    multirate_slow_strides: Optional[List[int]] = None,
    consistency_mode: str = "smooth_l1",
    infonce_tau: float = 0.1,
    **kwargs,
) -> CSTDSCT:
    """Hydra-compatible factory function."""
    config = CSTDSConfig(
        input_channels=input_channels,
        output_dim=output_dim,
        share_cnn=share_cnn,
        cnn_channels=cnn_channels or [16, 32, 64, 64],
        cnn_kernels=cnn_kernels or [11, 5, 5, 3],
        cnn_strides=cnn_strides or [5, 2, 4, 2],
        n_residual_blocks=n_residual_blocks,
        use_cs_tds=use_cs_tds,
        d_model=d_model,
        n_spatial_blocks=n_spatial_blocks,
        n_temporal_blocks=n_temporal_blocks,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        use_circular_spatial_rope=use_circular_spatial_rope,
        use_channel_embeddings=use_channel_embeddings,
        use_attention_pooling=use_attention_pooling,
        use_log_compression=use_log_compression,
        channel_rotation_range=channel_rotation_range,
        channel_mix_strength=channel_mix_strength,
        channel_drop_p=channel_drop_p,
        amplitude_jitter_std=amplitude_jitter_std,
        chunk_mask_ratio=chunk_mask_ratio,
        freq_mask_num=freq_mask_num,
        freq_mask_max=freq_mask_max,
        gaussian_noise_prob=gaussian_noise_prob,
        gaussian_noise_min_snr_db=gaussian_noise_min_snr_db,
        gaussian_noise_max_snr_db=gaussian_noise_max_snr_db,
        input_mean=input_mean,
        input_std=input_std,
        slow_state_type=slow_state_type,
        slow_state_hidden_dim=slow_state_hidden_dim,
        slow_state_layers=slow_state_layers,
        slow_state_dropout=slow_state_dropout,
        consistency_enabled=consistency_enabled,
        consistency_rotation_range=consistency_rotation_range,
        consistency_channel_mix_strength=consistency_channel_mix_strength,
        consistency_amplitude_jitter_std=consistency_amplitude_jitter_std,
        use_gradient_reversal=use_gradient_reversal,
        num_stages=num_stages,
        stage_reversal_lambda=stage_reversal_lambda,
        use_legacy_attention=use_legacy_attention,
        rope_mode=rope_mode,
        use_anatomy_head=use_anatomy_head,
        use_per_finger_queries=use_per_finger_queries,
        n_finger_queries=n_finger_queries,
        use_multirate_frontend=use_multirate_frontend,
        use_tds_skip=use_tds_skip,
        temporal_after_channel_pool=temporal_after_channel_pool,
        use_circular_stem=use_circular_stem,
        circular_ring_kernel=circular_ring_kernel,
        use_circular_channel_mix=use_circular_channel_mix,
        circular_channel_mix_kernel=circular_channel_mix_kernel,
        use_encoder_skip=use_encoder_skip,
        spatial_at_stem_dim=spatial_at_stem_dim,
        use_spatial_checkpoint=use_spatial_checkpoint,
        keep_encoder_rate=keep_encoder_rate,
        mix_emg_channels_at_stem=mix_emg_channels_at_stem,
        use_signal_tds_frontend=use_signal_tds_frontend,
        use_acsa_frontend=use_acsa_frontend,
        acsa_heads=acsa_heads,
        acsa_router_dim=acsa_router_dim,
        acsa_route_strides=acsa_route_strides or [8, 4, 2, 1],
        acsa_residual_init=acsa_residual_init,
        acsa_dynamic_edges=acsa_dynamic_edges,
        use_saem_frontend=use_saem_frontend,
        saem_local_dim=saem_local_dim,
        saem_heads=saem_heads,
        saem_route_stride=saem_route_stride,
        saem_residual_init=saem_residual_init,
        saem_dynamic_attention=saem_dynamic_attention,
        saem_signal_skip=saem_signal_skip,
        multirate_slow_channels=multirate_slow_channels or [16, 32, 64, 64],
        multirate_slow_kernels=multirate_slow_kernels or [7, 5, 5, 3],
        multirate_slow_strides=multirate_slow_strides or [2, 2, 2, 2],
        consistency_mode=consistency_mode,
        infonce_tau=infonce_tau,
    )
    return CSTDSCT(config)
