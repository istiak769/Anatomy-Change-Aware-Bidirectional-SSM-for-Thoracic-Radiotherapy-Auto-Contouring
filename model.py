"""
DAMM-Net++ v5: Dynamic Anatomical Motion Memory Network
========================================================
with Selective State Space Memory + Memory-Guided Boundary-Aware Decoder
     + Multi-Scale Attention Motion Branch + Uncertainty Estimation

Changes from v4:
  Adds **confidence-aware contour prediction** via learned per-pixel
  aleatoric uncertainty estimation (Kendall & Gal, 2017).

  1. **UncertaintyHead**: predicts per-pixel log-variance from decoded
     features.  High variance = the model knows it is uncertain (organ
     boundaries, ambiguous anatomy, rare structures).

  2. **Confidence map**: σ_conf = 1 / (1 + exp(log_var)) provides a
     clinically interpretable 0–1 confidence score.  Pixels below a
     threshold (e.g. 0.7) are flagged for manual review.

  3. **Uncertainty-attenuated loss** (in loss file): the predicted
     variance modulates per-pixel loss contributions so the model
     learns to be uncertain where ground truth is ambiguous, while
     a regularisation term prevents the trivial all-uncertain solution.

  Clinical motivation:
     In radiotherapy, silently providing a wrong contour is worse than
     flagging it for expert review.  Uncertainty maps give physicists
     a per-voxel quality indicator, enabling selective manual correction
     and quantified contouring confidence — directly supporting AAPM
     TG-275 recommendations for AI-assisted contouring QA.

Input  : (B, T, 3, H, W)
Outputs: dict  'seg'         (B, T, n_cls, H, W)
               'boundary'    (B, T, 1,     H, W)
               'sdm'         (B, T, n_cls, H, W)
               'presence'    (B, T, n_cls)
               'uncertainty' (B, T, 1,     H, W)   log-variance
               'confidence'  (B, T, 1,     H, W)   sigmoid confidence
               'deep_bnd'    list[T] of list[3] boundary maps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# 1. ConvNeXt Encoder  (unchanged)
# ===================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion=4, drop_path=0.0):
        super().__init__()
        hidden = dim * expansion
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class ConvNeXtStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, drop_path=0.0):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.GroupNorm(1, in_ch, eps=1e-6),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
        ) if in_ch != out_ch else nn.Identity()
        dp_rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_ch, drop_path=dp_rates[i]) for i in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_ch=3, dims=(48, 96, 192, 384), depths=(2, 2, 6, 2), drop_path=0.1):
        super().__init__()
        self.dims = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6),
        )
        self.stages = nn.ModuleList()
        for i in range(4):
            if i == 0:
                stage = nn.Sequential(*[
                    ConvNeXtBlock(dims[0], drop_path=drop_path * j / max(depths[0] - 1, 1))
                    for j in range(depths[0])
                ])
            else:
                stage = ConvNeXtStage(dims[i - 1], dims[i], depths[i], drop_path)
            self.stages.append(stage)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# ===================================================================
# 2. Multi-Scale Attention-Enhanced Motion Branch  (NEW in v3.1)
# ===================================================================

class MotionScaleBranch(nn.Module):
    """
    Single-scale motion feature extractor.

    Processes ΔF at one resolution with two 3×3 convolutions.
    Used as a building block inside the multi-scale pyramid.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MotionChannelAttention(nn.Module):
    """
    Squeeze-Excite channel attention for motion features.

    Not all feature channels change equally between slices.  A bladder-
    filling event activates different channels than a bowel-loop shift.
    This module learns to weight channels by their motion informativeness.

    Architecture:
        GAP → Linear(C → C//r) → GELU → Linear(C//r → C) → Sigmoid
    """

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.GELU(),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x).unsqueeze(-1).unsqueeze(-1)


class MotionSpatialGate(nn.Module):
    """
    Spatial attention gate for motion features.

    Highlights regions with significant anatomical change and suppresses
    static background — clinically, the body contour and spine rarely move
    between slices while OAR boundaries shift substantially.

    Uses both max-pool and avg-pool channel reductions to capture both
    the strongest motion signal and the average motion magnitude.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, 7, padding=3, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)              # (B, 1, h, w)
        max_out = x.amax(dim=1, keepdim=True)              # (B, 1, h, w)
        spatial_descriptor = torch.cat([avg_out, max_out], dim=1)  # (B, 2, h, w)
        return x * self.conv(spatial_descriptor)


class MotionDifferenceBranch(nn.Module):
    """
    Multi-Scale Attention-Enhanced Motion Difference Branch.

    Processes the inter-slice feature difference ΔF = F_z − F_{z−1} at
    three spatial scales to capture anatomical changes at different
    granularities:

        Scale 1 (1×):   fine motion — bowel loop shifts, small vessel changes
        Scale 2 (½×):   medium motion — bladder/rectum filling, organ deformation
        Scale 3 (¼×):   coarse motion — body contour shift, large organ appearance

    The multi-scale features are upsampled to the original resolution and
    fused, then refined with channel attention (which channels carry motion?)
    and spatial attention (where is the motion?).

    This design is clinically motivated: radiation oncologists assess
    anatomical change at multiple spatial scales when contouring
    sequential slices, paying more attention to regions and structures
    that are actively changing.
    """

    def __init__(self, in_ch: int):
        super().__init__()

        # Multi-scale branches
        self.scale1 = MotionScaleBranch(in_ch)       # 1× resolution
        self.scale2 = MotionScaleBranch(in_ch)       # ½× resolution
        self.scale3 = MotionScaleBranch(in_ch)       # ¼× resolution

        # Downsampling for scale 2 and 3
        self.down2 = nn.AvgPool2d(2)
        self.down4 = nn.AvgPool2d(4)

        # Fuse multi-scale features (3C → C)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, 1, bias=False),
            nn.GroupNorm(8, in_ch),
            nn.GELU(),
        )

        # Dual attention refinement
        self.channel_attn = MotionChannelAttention(in_ch, reduction=4)
        self.spatial_gate = MotionSpatialGate(in_ch)

        # Learnable residual scale (starts at 0 for stable init)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_curr: torch.Tensor, f_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_curr: (B, C, h, w) current slice bottleneck features
            f_prev: (B, C, h, w) previous slice features (zeros for first slice)
        Returns:
            (B, C, h, w) multi-scale attention-refined motion features
        """
        delta = f_curr - f_prev
        h, w = delta.shape[2:]

        # Multi-scale extraction
        m1 = self.scale1(delta)                                             # (B, C, h, w)

        delta_half = self.down2(delta)                                      # (B, C, h/2, w/2)
        m2 = self.scale2(delta_half)
        m2 = F.interpolate(m2, size=(h, w), mode='bilinear', align_corners=False)

        delta_quarter = self.down4(delta)                                   # (B, C, h/4, w/4)
        m3 = self.scale3(delta_quarter)
        m3 = F.interpolate(m3, size=(h, w), mode='bilinear', align_corners=False)

        # Fuse scales
        fused = self.fuse(torch.cat([m1, m2, m3], dim=1))                  # (B, C, h, w)

        # Dual attention
        fused = self.channel_attn(fused)
        fused = self.spatial_gate(fused)

        # Residual with learnable scale
        return self.gamma * fused + delta


# ===================================================================
# 3. Selective State Space Memory  (unchanged from v2)
# ===================================================================

class SelectiveSSM(nn.Module):
    """Single-step Mamba-style Selective State Space Model."""

    def __init__(self, d_model, d_state=16, dt_rank=None, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or max(d_model // 16, 1)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(dt_init_floor))
            + math.log(dt_init_floor)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.d_inner, self.d_state, device=device)

    def forward(self, x, h_prev):
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)
        x_branch = F.silu(x_branch)

        x_proj_out = self.x_proj(x_branch)
        dt_raw = x_proj_out[:, :self.dt_rank]
        B = x_proj_out[:, self.dt_rank:self.dt_rank + self.d_state]
        C = x_proj_out[:, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(1)

        h_new = A_bar * h_prev + B_bar * x_branch.unsqueeze(-1)
        y = torch.einsum('bdn,bn->bd', h_new, C)
        y = y + self.D * x_branch
        y = y * F.silu(z_gate)
        y = self.out_proj(y)
        return y, h_new


class SelectiveSSMMemory(nn.Module):
    """Anatomical Motion Memory built on Selective SSM."""

    def __init__(self, ch, d_state=16, expand=2):
        super().__init__()
        self.ch = ch
        self.d_state = d_state
        self.d_inner = ch * expand

        self.input_proj = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.GroupNorm(8, ch), nn.GELU(),
        )
        self.ssm = SelectiveSSM(d_model=ch, d_state=d_state, expand=expand)
        self.post_norm = nn.GroupNorm(8, ch)

    def init_memory(self, batch_size, h, w, device):
        output = torch.zeros(batch_size, self.ch, h, w, device=device)
        state = self.ssm.init_state(batch_size * h * w, device)
        return output, state

    def forward(self, f_curr, motion, memory_state):
        B, C, h, w = f_curr.shape
        _, ssm_state = memory_state
        combined = self.input_proj(torch.cat([f_curr, motion], dim=1))
        x_flat = combined.permute(0, 2, 3, 1).reshape(B * h * w, C)
        y_flat, ssm_state_new = self.ssm(x_flat, ssm_state)
        output = y_flat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        output = self.post_norm(output + f_curr)
        return output, (output, ssm_state_new)


# ===================================================================
# 4. Gated Bidirectional Memory Fusion  (unchanged from v2)
# ===================================================================

class GatedMemoryFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 1, bias=False), nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(8, ch), nn.GELU(),
        )

    def forward(self, m_fwd, m_bwd):
        g = self.gate(torch.cat([m_fwd, m_bwd], dim=1))
        return self.out_proj(g * m_fwd + (1.0 - g) * m_bwd)


# ===================================================================
# 5. Memory-guided Cross-Attention  (unchanged from v2)
# ===================================================================

class MemoryCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.sr = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=False),
            nn.GroupNorm(1, dim, eps=1e-6),
        ) if sr_ratio > 1 else nn.Identity()
        self.norm = nn.GroupNorm(8, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_curr, memory, motion):
        B, C, h, w = f_curr.shape
        context_sr = self.sr(memory + motion)
        Q = self.q_proj(f_curr)
        KV = self.kv_proj(context_sr)
        K, V = KV.chunk(2, dim=1)
        _, _, h_sr, w_sr = K.shape
        n_q, n_kv = h * w, h_sr * w_sr
        Q = Q.reshape(B, self.num_heads, self.d_k, n_q).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, self.d_k, n_kv).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, self.d_k, n_kv).permute(0, 1, 3, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V).permute(0, 1, 3, 2).reshape(B, C, h, w)
        return self.norm(self.gamma * self.out_proj(out) + f_curr)


# ===================================================================
# 6. Memory-Guided Boundary-Aware Decoder  (NEW)
# ===================================================================

class MemoryGuidedSkipAttention(nn.Module):
    """
    Memory-Guided Skip Attention (MGSA).

    The anatomical memory from the SSM bottleneck generates dual
    channel-spatial attention gates that selectively filter encoder
    skip features before they enter the decoder.

    Channel gate:  "Which feature channels are relevant given the
                    organs the memory knows are present?"
    Spatial gate:  "Where should the decoder focus given the spatial
                    context accumulated from adjacent slices?"

    Architecture:
        memory → upsample → project
        channel: GAP → MLP → σ → scale skip channels
        spatial: 1×1 conv → σ → scale skip spatially
        output = skip ⊙ channel_gate ⊙ spatial_gate
    """

    def __init__(self, skip_ch: int, mem_ch: int):
        super().__init__()
        # Project memory channels to match skip
        self.mem_proj = nn.Sequential(
            nn.Conv2d(mem_ch, skip_ch, 1, bias=False),
            nn.GroupNorm(8, skip_ch),
            nn.GELU(),
        )

        # Channel attention from memory context
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(skip_ch, skip_ch // 4),
            nn.GELU(),
            nn.Linear(skip_ch // 4, skip_ch),
            nn.Sigmoid(),
        )

        # Spatial attention from memory context
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, skip_ch // 4),
            nn.GELU(),
            nn.Conv2d(skip_ch // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip:   (B, skip_ch, h, w) encoder feature
            memory: (B, mem_ch, h_m, w_m) memory-augmented bottleneck
        Returns:
            (B, skip_ch, h, w) memory-refined skip features
        """
        # Align memory to skip resolution
        mem = F.interpolate(memory, size=skip.shape[2:], mode='bilinear', align_corners=False)
        mem = self.mem_proj(mem)                             # (B, skip_ch, h, w)

        # Channel gate
        ch_gate = self.channel_gate(mem)                     # (B, skip_ch)
        ch_gate = ch_gate.unsqueeze(-1).unsqueeze(-1)        # (B, skip_ch, 1, 1)

        # Spatial gate
        sp_gate = self.spatial_gate(mem)                     # (B, 1, h, w)

        return skip * ch_gate * sp_gate


class LearnableEdgeDetector(nn.Module):
    """
    Learnable edge detection with Sobel-initialised kernels.

    Uses depthwise convolutions initialised with Sobel filters in X and Y
    directions, followed by a learnable 1×1 refinement.  The initialisation
    ensures edge sensitivity from epoch 0, while the learnable weights
    allow the model to adapt to organ-specific boundary patterns.
    """

    def __init__(self, ch: int):
        super().__init__()
        # Depthwise Sobel-initialised conv (X direction)
        self.edge_x = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        # Depthwise Sobel-initialised conv (Y direction)
        self.edge_y = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)

        # Initialise with Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        with torch.no_grad():
            self.edge_x.weight.copy_(sobel_x.view(1, 1, 3, 3).expand(ch, -1, -1, -1))
            self.edge_y.weight.copy_(sobel_y.view(1, 1, 3, 3).expand(ch, -1, -1, -1))

        # Refinement: combine gradient magnitudes
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(8, ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns edge-enhanced features (B, C, H, W)."""
        gx = self.edge_x(x)
        gy = self.edge_y(x)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return self.refine(grad_mag)


class BoundaryAwareDecoderBlock(nn.Module):
    """
    Single decoder level with boundary-aware feature refinement.

    Architecture:
        ┌─────────────────────────────────────────────┐
        │  upsample(in) ─┐                            │
        │                 ├── concat ── main_conv ──┐  │
        │  MGSA(skip) ───┘                         │  │
        │                                          │  │
        │  LearnableEdge(main_out) ── sigmoid ─┐   │  │
        │                                      │   │  │
        │  boundary_gate = 1 + α·edge_map      │   │  │
        │                                      │   │  │
        │  output = main_out ⊙ boundary_gate ──┘   │  │
        │                                          │  │
        │  boundary_pred (for deep supervision) ◄──┘  │
        └─────────────────────────────────────────────┘
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, mem_ch: int):
        super().__init__()
        # Upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Memory-guided skip attention
        self.skip_attn = MemoryGuidedSkipAttention(skip_ch, mem_ch)

        # Main conv path
        self.main_conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

        # Boundary-aware refinement
        self.edge_detector = LearnableEdgeDetector(out_ch)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))   # learnable scaling, init 0

        # Intermediate boundary prediction (for deep supervision)
        self.boundary_pred = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch // 4),
            nn.GELU(),
            nn.Conv2d(out_ch // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,           # from previous decoder level
        skip: torch.Tensor,        # encoder skip connection
        memory: torch.Tensor,      # memory-augmented bottleneck
    ):
        """
        Returns:
            features:     (B, out_ch, h, w) refined decoder features
            boundary_map: (B, 1, h, w) intermediate boundary prediction
        """
        # Upsample and align
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        # Memory-guided skip filtering
        skip_refined = self.skip_attn(skip, memory)

        # Main path
        main_out = self.main_conv(torch.cat([x, skip_refined], dim=1))

        # Boundary-aware refinement
        edge_features = self.edge_detector(main_out)
        edge_map = self.edge_gate(edge_features)
        # Multiplicative sharpening: amplify features near boundaries
        # α starts at 0 → no effect initially, learned during training
        refined = main_out * (1.0 + self.alpha * edge_map)

        # Intermediate boundary prediction for deep supervision
        bnd_pred = self.boundary_pred(main_out)

        return refined, bnd_pred


class MGBADecoder(nn.Module):
    """
    Memory-Guided Boundary-Aware Decoder (MGBA-Decoder).

    Three levels of boundary-aware decoding with memory-guided skip
    attention, plus a final 4× upsample to input resolution.

    Deep boundary supervision: intermediate boundary maps at each
    decoder level are returned for multi-scale boundary loss.
    """

    def __init__(self, dims=(48, 96, 192, 384)):
        super().__init__()
        mem_ch = dims[3]   # memory channel width

        # Three decoder levels with memory-guided skip + boundary awareness
        self.level3 = BoundaryAwareDecoderBlock(dims[3], dims[2], dims[2], mem_ch)   # H/32 → H/16
        self.level2 = BoundaryAwareDecoderBlock(dims[2], dims[1], dims[1], mem_ch)   # H/16 → H/8
        self.level1 = BoundaryAwareDecoderBlock(dims[1], dims[0], dims[0], mem_ch)   # H/8  → H/4

        # Final 4× upsample to input resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(8, dims[0]),
            nn.GELU(),
        )

        # Fuse multi-scale boundary maps for the final boundary head
        self.boundary_fuse = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck: torch.Tensor, skips: list, memory: torch.Tensor):
        """
        Args:
            bottleneck: (B, dims[3], h/32, w/32) — memory-augmented features
            skips:      [stage0, stage1, stage2] encoder features
            memory:     (B, dims[3], h/32, w/32) — fused bidirectional memory

        Returns:
            decoded:       (B, dims[0], H, W) decoded features at input resolution
            boundary_map:  (B, 1, H, W) fused multi-scale boundary map
            deep_bnds:     list of 3 boundary maps at intermediate resolutions
        """
        # Level 3: H/32 → H/16
        x3, bnd3 = self.level3(bottleneck, skips[2], memory)

        # Level 2: H/16 → H/8
        x2, bnd2 = self.level2(x3, skips[1], memory)

        # Level 1: H/8 → H/4
        x1, bnd1 = self.level1(x2, skips[0], memory)

        # Final upsample
        decoded = self.final_up(x1)
        H, W = decoded.shape[2:]

        # Fuse multi-scale boundary maps at output resolution
        bnd3_up = F.interpolate(bnd3, size=(H, W), mode='bilinear', align_corners=False)
        bnd2_up = F.interpolate(bnd2, size=(H, W), mode='bilinear', align_corners=False)
        bnd1_up = F.interpolate(bnd1, size=(H, W), mode='bilinear', align_corners=False)

        boundary_map = self.boundary_fuse(torch.cat([bnd3_up, bnd2_up, bnd1_up], dim=1))

        deep_bnds = [bnd3, bnd2, bnd1]    # for deep supervision loss

        return decoded, boundary_map, deep_bnds


# ===================================================================
# 7. Task Heads  (unchanged)
# ===================================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_ch // 2), nn.GELU(),
            nn.Conv2d(in_ch // 2, n_classes, 1),
        )
    def forward(self, x): return self.head(x)


class SDMHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_ch // 2), nn.GELU(),
            nn.Conv2d(in_ch // 2, n_classes, 1),
        )
    def forward(self, x): return self.head(x)


class PresenceHead(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(in_ch, in_ch // 2), nn.GELU(), nn.Linear(in_ch // 2, n_classes))
    def forward(self, x): return self.head(self.pool(x).flatten(1))


class UncertaintyHead(nn.Module):
    """
    Per-pixel aleatoric uncertainty estimation.

    Predicts log-variance log(σ²) for each pixel from decoded features.
    The output is unconstrained — variance is recovered as exp(log_var).

    Interpretation:
        High log_var  →  model is uncertain  →  flag for clinical review
        Low log_var   →  model is confident  →  trust the contour

    Confidence map:
        conf = σ(−log_var) = 1 / (1 + σ²)
        This gives a smooth 0–1 score where 1 = fully confident.

    Architecture uses a two-layer MLP (as 1×1 convs) with a deeper
    3×3 context layer to capture local spatial uncertainty patterns —
    organ boundaries are inherently uncertain, and this spatial context
    helps the model learn that.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_ch // 2),
            nn.GELU(),
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_ch // 4),
            nn.GELU(),
            nn.Conv2d(in_ch // 4, 1, 1),
        )
        # Initialise final conv bias to -2 so initial σ² ≈ 0.14
        # (moderately confident), preventing unstable early training
        nn.init.constant_(self.head[-1].bias, -2.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) decoded features
        Returns:
            log_var: (B, 1, H, W) — unconstrained log-variance
        """
        return self.head(x)


# ===================================================================
# 8. DAMM-Net++ v5 Main Model
# ===================================================================

class DAMMNetPP(nn.Module):
    """
    DAMM-Net++ v5: full architecture with uncertainty estimation.

    All prior innovations retained:
        v2: Selective SSM Memory, Gated Bidirectional Fusion
        v3: MGBA-Decoder (memory-guided skip attention, boundary-aware refinement)
        v4: Multi-Scale Attention-Enhanced Motion Branch
        v5: UncertaintyHead + confidence-aware output (this version)
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 3,
        enc_dims: tuple = (48, 96, 192, 384),
        enc_depths: tuple = (2, 2, 6, 2),
        num_heads: int = 8,
        drop_path: float = 0.1,
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.bottleneck_ch = enc_dims[3]

        # Shared encoder
        self.encoder = ConvNeXtEncoder(in_channels, enc_dims, enc_depths, drop_path)

        # Sequential memory components
        self.motion_branch = MotionDifferenceBranch(enc_dims[3])
        self.memory_fwd = SelectiveSSMMemory(enc_dims[3], d_state=ssm_d_state, expand=ssm_expand)
        self.memory_bwd = SelectiveSSMMemory(enc_dims[3], d_state=ssm_d_state, expand=ssm_expand)
        self.memory_fuse = GatedMemoryFusion(enc_dims[3])
        self.cross_attn = MemoryCrossAttention(enc_dims[3], num_heads)

        # Memory-Guided Boundary-Aware Decoder
        self.decoder = MGBADecoder(enc_dims)

        # Task heads
        self.seg_head = SegmentationHead(enc_dims[0], n_classes)
        self.sdm_head = SDMHead(enc_dims[0], n_classes)
        self.presence_head = PresenceHead(enc_dims[3], n_classes)
        self.uncertainty_head = UncertaintyHead(enc_dims[0])

    def _encode_sequence(self, x):
        B, T = x.shape[:2]
        all_feats, bottlenecks = [], []
        for t in range(T):
            feats = self.encoder(x[:, t])
            all_feats.append(feats)
            bottlenecks.append(feats[3])
        return all_feats, torch.stack(bottlenecks, dim=1)

    def _bidirectional_memory(self, bottlenecks):
        B, T, C, h, w = bottlenecks.shape
        device = bottlenecks.device

        state_fwd = self.memory_fwd.init_memory(B, h, w, device)
        fwd_outputs, motions = [], []
        for t in range(T):
            f_curr = bottlenecks[:, t]
            f_prev = bottlenecks[:, t - 1] if t > 0 else torch.zeros_like(f_curr)
            motion = self.motion_branch(f_curr, f_prev)
            motions.append(motion)
            output_fwd, state_fwd = self.memory_fwd(f_curr, motion, state_fwd)
            fwd_outputs.append(output_fwd)

        state_bwd = self.memory_bwd.init_memory(B, h, w, device)
        bwd_outputs = [None] * T
        for t in range(T - 1, -1, -1):
            f_curr = bottlenecks[:, t]
            f_next = bottlenecks[:, t + 1] if t < T - 1 else torch.zeros_like(f_curr)
            motion_bwd = self.motion_branch(f_curr, f_next)
            output_bwd, state_bwd = self.memory_bwd(f_curr, motion_bwd, state_bwd)
            bwd_outputs[t] = output_bwd

        fused = [self.memory_fuse(fwd_outputs[t], bwd_outputs[t]) for t in range(T)]
        return fused, motions

    def forward(self, x):
        B, T = x.shape[:2]
        H, W = x.shape[3], x.shape[4]

        all_feats, bottlenecks = self._encode_sequence(x)
        fused_memories, motions = self._bidirectional_memory(bottlenecks)

        seg_list, bnd_list, sdm_list, prs_list = [], [], [], []
        unc_list, conf_list = [], []
        deep_bnd_list = []

        for t in range(T):
            # Memory-guided cross attention at bottleneck
            enhanced = self.cross_attn(
                f_curr=bottlenecks[:, t],
                memory=fused_memories[t],
                motion=motions[t],
            )

            # Presence from bottleneck
            prs_list.append(self.presence_head(enhanced))

            # Decode with memory-guided boundary-aware decoder
            skips = all_feats[t][:3]
            decoded, boundary_map, deep_bnds = self.decoder(
                bottleneck=enhanced,
                skips=skips,
                memory=fused_memories[t],
            )

            # Align to input resolution
            if decoded.shape[2:] != (H, W):
                decoded = F.interpolate(decoded, size=(H, W), mode='bilinear', align_corners=False)
            if boundary_map.shape[2:] != (H, W):
                boundary_map = F.interpolate(boundary_map, size=(H, W), mode='bilinear', align_corners=False)

            seg_list.append(self.seg_head(decoded))
            bnd_list.append(boundary_map)
            sdm_list.append(self.sdm_head(decoded))
            deep_bnd_list.append(deep_bnds)

            # Uncertainty estimation
            log_var = self.uncertainty_head(decoded)          # (B, 1, H, W)
            confidence = torch.sigmoid(-log_var)             # (B, 1, H, W) ∈ [0,1]
            unc_list.append(log_var)
            conf_list.append(confidence)

        return {
            'seg':         torch.stack(seg_list, dim=1),
            'boundary':    torch.stack(bnd_list, dim=1),
            'sdm':         torch.stack(sdm_list, dim=1),
            'presence':    torch.stack(prs_list, dim=1),
            'uncertainty': torch.stack(unc_list, dim=1),     # log-variance
            'confidence':  torch.stack(conf_list, dim=1),    # σ(−log_var)
            'deep_bnd':    deep_bnd_list,
        }


# ===================================================================
# Instantiation helper
# ===================================================================

def build_damm_net(n_classes, device, in_channels=3):
    model = DAMMNetPP(n_classes=n_classes, in_channels=in_channels).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    return model




"""
Loss functions and metrics for DAMM-Net++ v5.

Multi-task loss with deep boundary supervision and uncertainty attenuation:
    L = 1.0·L_dice_ua + 0.5·L_ce_ua + 0.2·L_boundary + 0.2·L_sdm
      + 0.1·L_topology + 0.1·L_presence + 0.15·L_deep_bnd + 0.1·L_unc_reg

Uncertainty-attenuated losses (Kendall & Gal, 2017):
    L_ua = ½ · exp(−log_var) · L_pixel + ½ · log_var

    The predicted per-pixel variance modulates loss contributions:
      • High variance → low loss weight (model learns to be uncertain
        where ground truth is ambiguous, e.g. organ boundaries)
      • The regularisation term ½·log_var prevents the trivial solution
        of predicting infinite variance everywhere

    An additional KL-style regularisation term penalises the mean
    uncertainty to keep the model from becoming over-cautious.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


# ===================================================================
# Utilities
# ===================================================================

def mask_to_one_hot(mask, num_classes):
    return F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()


def compute_gt_boundary(masks):
    m = masks.unsqueeze(1).float()
    dilate = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    erode = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return (dilate - erode).clamp(0, 1)


def compute_gt_sdm_batch(masks, num_classes):
    B, H, W = masks.shape
    sdm = torch.zeros(B, num_classes, H, W, dtype=torch.float32)
    masks_np = masks.cpu().numpy()
    for b in range(B):
        for c in range(num_classes):
            binary = (masks_np[b] == c).astype(np.float64)
            if binary.sum() == 0:
                sdm[b, c] = -5.0
            elif binary.sum() == H * W:
                sdm[b, c] = 5.0
            else:
                pos_dist = ndimage.distance_transform_edt(binary)
                neg_dist = ndimage.distance_transform_edt(1.0 - binary)
                signed = pos_dist - neg_dist
                max_abs = max(np.abs(signed).max(), 1e-6)
                sdm[b, c] = torch.from_numpy(signed / max_abs).float()
    return sdm.to(masks.device)


def compute_gt_presence(masks, num_classes):
    B = masks.shape[0]
    presence = torch.zeros(B, num_classes, device=masks.device)
    for b in range(B):
        for c in torch.unique(masks[b]):
            presence[b, c.long()] = 1.0
    return presence


# ===================================================================
# Individual Losses
# ===================================================================

def dice_loss(y_true_oh, logits, smooth=1.0):
    probs = torch.softmax(logits, dim=1)
    inter = (y_true_oh * probs).sum(dim=(2, 3))
    union = y_true_oh.sum(dim=(2, 3)) + probs.sum(dim=(2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def focal_ce_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def boundary_bce_loss(b_hat, masks):
    gt_bnd = compute_gt_boundary(masks)
    if gt_bnd.shape[2:] != b_hat.shape[2:]:
        gt_bnd = F.interpolate(gt_bnd, size=b_hat.shape[2:], mode='nearest')
    return F.binary_cross_entropy(b_hat, gt_bnd)


def deep_boundary_supervision_loss(deep_bnds, masks):
    """
    Compute boundary BCE at each intermediate decoder scale.

    Args:
        deep_bnds: list of 3 boundary maps at different resolutions
                   [(B,1,h3,w3), (B,1,h2,w2), (B,1,h1,w1)]
        masks:     (B, H, W) integer labels

    Returns:
        scalar loss averaged across all scales
    """
    gt_bnd_full = compute_gt_boundary(masks)   # (B, 1, H, W)
    loss = 0.0
    for bnd_pred in deep_bnds:
        # Downsample GT boundary to match this scale
        gt_bnd_scale = F.interpolate(
            gt_bnd_full, size=bnd_pred.shape[2:], mode='nearest'
        )
        loss += F.binary_cross_entropy(bnd_pred, gt_bnd_scale)
    return loss / len(deep_bnds)


def sdm_loss(sdm_pred, sdm_gt):
    return F.mse_loss(sdm_pred, sdm_gt)


def presence_loss(pres_logits, pres_gt):
    return F.binary_cross_entropy_with_logits(pres_logits, pres_gt)


def topology_connectivity_loss(logits):
    probs = torch.softmax(logits, dim=1)
    dx = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs()
    dy = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs()
    return (dx ** 2).mean() + (dy ** 2).mean()


# ===================================================================
# Uncertainty-Attenuated Losses (NEW in v5)
# ===================================================================

def uncertainty_attenuated_dice(y_true_oh, logits, log_var, smooth=1.0):
    """
    Dice loss with per-pixel uncertainty attenuation.

    L = ½ · exp(−log_var) · dice_map + ½ · log_var

    High-variance pixels contribute less to the dice numerator/denominator,
    allowing the model to be uncertain at ambiguous boundaries without
    being penalised as harshly.

    Args:
        y_true_oh: (B, C, H, W) one-hot
        logits:    (B, C, H, W) raw logits
        log_var:   (B, 1, H, W) predicted log-variance
    """
    probs = torch.softmax(logits, dim=1)

    # Per-pixel precision weight: exp(−log_var), clamped for stability
    precision = torch.exp(-log_var).clamp(max=100.0)          # (B, 1, H, W)

    # Weighted intersection and union
    weighted_inter = (y_true_oh * probs * precision).sum(dim=(2, 3))
    weighted_sum = (y_true_oh * precision).sum(dim=(2, 3)) + (probs * precision).sum(dim=(2, 3))

    dice = (2.0 * weighted_inter + smooth) / (weighted_sum + smooth)
    dice_loss_val = 1.0 - dice.mean()

    # Regularisation: penalise high variance (prevent trivial solution)
    reg = 0.5 * log_var.mean()

    return dice_loss_val + reg


def uncertainty_attenuated_focal_ce(logits, targets, log_var, gamma=2.0):
    """
    Focal cross-entropy with per-pixel uncertainty attenuation.

    L = ½ · exp(−log_var) · focal_ce + ½ · log_var

    Args:
        logits:  (B, C, H, W)
        targets: (B, H, W) integer labels
        log_var: (B, 1, H, W) predicted log-variance
    """
    ce = F.cross_entropy(logits, targets, reduction='none')    # (B, H, W)
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce                             # (B, H, W)

    # Per-pixel attenuation
    precision = torch.exp(-log_var.squeeze(1)).clamp(max=100.0)  # (B, H, W)
    attenuated = 0.5 * precision * focal + 0.5 * log_var.squeeze(1)

    return attenuated.mean()


def uncertainty_regularisation(log_var):
    """
    Prevent over-cautious predictions by penalising the mean uncertainty.

    This is a soft constraint that encourages the model to be confident
    on most pixels and only uncertain where genuinely ambiguous.

    L_reg = max(0, mean(log_var) − τ)²

    where τ = -1.0 corresponds to σ² ≈ 0.37 (moderate confidence).
    Below this threshold no penalty is applied.
    """
    tau = -1.0
    mean_logvar = log_var.mean()
    excess = F.relu(mean_logvar - tau)
    return excess ** 2


# ===================================================================
# Combined Multi-task Loss
# ===================================================================

class DAMMNetLoss(nn.Module):
    """
    Multi-task loss for DAMM-Net++ v5.

    Weights:
        Dice (UA)         1.0   uncertainty-attenuated overlap
        Focal CE (UA)     0.5   uncertainty-attenuated hard-example mining
        Boundary          0.2   final boundary head
        SDM               0.2   signed distance map
        Topology          0.1   connectivity regularisation
        Presence          0.1   organ presence classification
        Deep boundary     0.15  multi-scale boundary supervision
        Uncertainty reg   0.1   prevents over-cautious predictions
    """

    def __init__(
        self,
        n_classes,
        w_dice=1.0,
        w_ce=0.5,
        w_bnd=0.2,
        w_sdm=0.2,
        w_topo=0.1,
        w_pres=0.1,
        w_deep_bnd=0.15,
        w_unc_reg=0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.w_bnd = w_bnd
        self.w_sdm = w_sdm
        self.w_topo = w_topo
        self.w_pres = w_pres
        self.w_deep_bnd = w_deep_bnd
        self.w_unc_reg = w_unc_reg

    def forward(self, outputs, masks, sdm_gt=None, pres_gt=None):
        B, T = masks.shape[:2]
        device = masks.device

        l_dice = l_ce = l_bnd = l_sdm = l_topo = l_pres = l_deep = l_unc = 0.0

        deep_bnd_list = outputs.get('deep_bnd', None)
        has_uncertainty = 'uncertainty' in outputs

        for t in range(T):
            seg_t = outputs['seg'][:, t]
            bnd_t = outputs['boundary'][:, t]
            sdm_t = outputs['sdm'][:, t]
            prs_t = outputs['presence'][:, t]
            mask_t = masks[:, t]

            y_oh = mask_to_one_hot(mask_t, self.n_classes).to(device)

            if has_uncertainty:
                log_var_t = outputs['uncertainty'][:, t]     # (B, 1, H, W)
                l_dice += uncertainty_attenuated_dice(y_oh, seg_t, log_var_t)
                l_ce += uncertainty_attenuated_focal_ce(seg_t, mask_t, log_var_t)
                l_unc += uncertainty_regularisation(log_var_t)
            else:
                l_dice += dice_loss(y_oh, seg_t)
                l_ce += focal_ce_loss(seg_t, mask_t)

            l_bnd += boundary_bce_loss(bnd_t, mask_t)
            l_topo += topology_connectivity_loss(seg_t)

            if deep_bnd_list is not None:
                l_deep += deep_boundary_supervision_loss(deep_bnd_list[t], mask_t)

            if sdm_gt is not None:
                gt_sdm_t = sdm_gt[:, t]
            else:
                gt_sdm_t = compute_gt_sdm_batch(mask_t, self.n_classes)
            l_sdm += sdm_loss(sdm_t, gt_sdm_t)

            if pres_gt is not None:
                gt_prs_t = pres_gt[:, t]
            else:
                gt_prs_t = compute_gt_presence(mask_t, self.n_classes)
            l_pres += presence_loss(prs_t, gt_prs_t)

        # Average over sequence length
        l_dice /= T
        l_ce /= T
        l_bnd /= T
        l_sdm /= T
        l_topo /= T
        l_pres /= T
        l_deep /= T
        l_unc /= T

        total = (
            self.w_dice * l_dice
            + self.w_ce * l_ce
            + self.w_bnd * l_bnd
            + self.w_sdm * l_sdm
            + self.w_topo * l_topo
            + self.w_pres * l_pres
            + self.w_deep_bnd * l_deep
            + self.w_unc_reg * l_unc
        )

        loss_dict = {}
        for name, val in [('dice', l_dice), ('ce', l_ce), ('boundary', l_bnd),
                          ('sdm', l_sdm), ('topology', l_topo), ('presence', l_pres),
                          ('deep_bnd', l_deep), ('unc_reg', l_unc)]:
            loss_dict[name] = val.item() if torch.is_tensor(val) else val

        return total, loss_dict


# ===================================================================
# Metrics
# ===================================================================

def pixel_accuracy(logits, masks):
    return (logits.argmax(1) == masks).float().mean().item()


def mean_iou_score(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        ious.append(1.0 if union == 0 else inter / union)
    return float(np.mean(ious))


def dice_coefficient(y_true_oh, logits, smooth=1.0):
    probs = torch.softmax(logits, dim=1)
    inter = (y_true_oh * probs).sum(dim=(1, 2, 3))
    union = y_true_oh.sum(dim=(1, 2, 3)) + probs.sum(dim=(1, 2, 3))
    return ((2.0 * inter + smooth) / (union + smooth)).mean()


def per_class_iou_score(preds, targets, num_classes):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        ious.append(0.0 if union == 0 else inter / union)
    return np.array(ious)
