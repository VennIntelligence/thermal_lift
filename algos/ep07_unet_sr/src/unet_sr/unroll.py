"""K-step physics-constrained unrolled super-resolution solver.

This is the committed redesign direction (ACL-024 / research_log/network_upgrade_roadmap.md).
It generalizes the V10 "residual-over-drizzle" model (which ACL-024 notes is already a 1-step
unroll) to K steps that ALTERNATE a learned proximal refinement with a data-consistency (DC)
step, and crucially END ON THE DC STEP so the returned estimate is pulled back onto the data:

    x_0 = drizzle / aligned-mean warm-start
    for k in 0..K-1:
        x <- x + prox_k([x, cond])           # learned proximal refinement (UNet, null-space prior)
        x <- x - eta_k * A^T(A x - y)         # DC gradient step (exact A^T via autograd) — LAST
    return x_K                                # ends on DC => forward-consistent by construction

Design history (ACL-025/026): the first implementation ran DC-then-prox, so the *prox had the
last word* — the unconstrained residual UNet could re-inject forward-inconsistent (hallucinated)
high-frequency AFTER the DC step, and a learnable eta let the optimizer shrink the DC step toward
zero. Empirically the DC residual then climbed ABOVE even the smooth warm-start floor (the
hallucination signature) while the structural loss fell. The fix: (1) reorder so each step — and
thus the output x_K — ENDS on the DC step; (2) FREEZE eta by default (learn_eta=False) so the DC
step cannot be bypassed. Anti-hallucination is now architectural (x_K respects the observations),
not a soft forward-model loss term — which was independently falsified (ACL-017/019,
forward_model_weight=0). Caveat: one gradient step is not a full projection; if the terminal DC
residual stays high, escalate eta or add a few CG steps for an exact range projection.

The forward operator A and its exact transpose A^T are certified in forward_torch.py (Gate A).
A^T comes from autograd, so the whole unroll is end-to-end differentiable (double-backward
during training — set create_graph via .training).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward_torch import ScenePSF, data_consistency_grad, prepare_forward_burst_plan
from .losses import gaussian_blur_2d
from .model import ThermalSRUNet

_USE_FORWARD_PLAN_DEFAULT = os.environ.get("TL_SOLVER_FORWARD_PLAN", "1") != "0"


def _inv_softplus(y: float) -> float:
    """raw such that softplus(raw) == y  (for initializing positive step sizes)."""
    return math.log(math.expm1(max(float(y), 1e-6)))


class UnrolledSolver(nn.Module):
    """K-step learned proximal-gradient solver over the certified physical forward operator.

    Args:
        n_steps:       K, number of unroll iterations (3-8 typical).
        cond_channels: channels of the conditioning tensor fed to each prox (5 for
                       solver_no_drizzle, or 9 for hybrid_drizzle2x: 5 fused↑2x
                       + 4 phase-bin drizzle@2x).
        base_channels: UNet width for the prox net(s).
        scale:         SR factor (2).
        share_weights: one prox net reused across steps (data-efficient, default) vs per-step.
        band_highpass_sigma_lr_px: >0 runs the DC term in the high-frequency band (rejects the
                       smooth per-frame drift; matches the SR band, per the data audit).
        huber_delta:   >0 uses a robust Huber DC term (hot/cold defects, stripe noise).
        eta_init:      DC step size (softplus-parameterized, per step).
        learn_eta:     if True, eta is a learnable Parameter; default False freezes it (a learnable
                       eta let the optimizer bypass the DC step — ACL-026).
        prox_use_se:   keep SE channel attention inside the prox UNet. Disable for E3 extent-
                       invariance tests where global average pooling is a suspected coupling path.
        prox_norm:     normalization inside the prox UNet ("group" or "none"). "none" removes
                       GroupNorm for E3 tests of scale/extent distribution drift.
    """

    def __init__(
        self,
        *,
        n_steps: int = 4,
        cond_channels: int = 9,
        base_channels: int = 64,
        scale: int = 2,
        share_weights: bool = True,
        band_highpass_sigma_lr_px: float = 5.0,
        huber_delta: float = 0.0,
        eta_init: float = 0.5,
        learn_eta: bool = False,
        prox_use_se: bool = True,
        prox_norm: str = "group",
        prox_highpass_residual: bool = False,
        prox_highpass_sigma_hr: float = 5.0,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_steps = int(n_steps)
        self.scale = int(scale)
        self.share_weights = bool(share_weights)
        self.cond_channels = int(cond_channels)
        self.band_highpass_sigma_lr_px = float(band_highpass_sigma_lr_px)
        self.huber_delta = float(huber_delta)
        self.prox_use_se = bool(prox_use_se)
        self.prox_norm = str(prox_norm)
        # D-E (ACL-042): high-pass the learned prox correction so it only injects high-frequency
        # detail while the low/mid-frequency field stays anchored to the warm-start + DC. Restores
        # sharpening capacity lost by removing GroupNorm/SE (E3), without letting the prox re-create
        # the low/mid-freq extent drift (tile-grid glow / background lift / flocculence). Off by
        # default -> identical to the plain residual prox.
        self.prox_highpass_residual = bool(prox_highpass_residual)
        self.prox_highpass_sigma_hr = float(prox_highpass_sigma_hr)

        in_ch = 1 + self.cond_channels  # [current estimate x] ++ conditioning
        if self.share_weights:
            self.prox = ThermalSRUNet(
                in_channels=in_ch,
                out_channels=1,
                base_channels=base_channels,
                scale=1,
                norm=self.prox_norm,
                use_se=self.prox_use_se,
            )
        else:
            self.prox = nn.ModuleList(
                ThermalSRUNet(
                    in_channels=in_ch,
                    out_channels=1,
                    base_channels=base_channels,
                    scale=1,
                    norm=self.prox_norm,
                    use_se=self.prox_use_se,
                )
                for _ in range(self.n_steps)
            )
        # per-step DC step sizes (softplus-parameterized, positive). FROZEN by default: a learnable
        # eta let the optimizer drive the DC step toward 0 and bypass the consistency constraint
        # (ACL-026). When frozen it's a buffer (moves with .to()/state_dict, receives no gradient).
        self.learn_eta = bool(learn_eta)
        eta_raw = torch.full((self.n_steps,), _inv_softplus(eta_init))
        if self.learn_eta:
            self.eta_raw = nn.Parameter(eta_raw)
        else:
            self.register_buffer("eta_raw", eta_raw)

    def _prox_net(self, k: int) -> ThermalSRUNet:
        return self.prox if self.share_weights else self.prox[k]

    def _prox(self, k: int, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Residual proximal refinement: x <- x + net([x, cond]) (V10-style, stable).

        With ``prox_highpass_residual`` (D-E, ACL-042) the correction is high-pass filtered before
        it is added, so the prox can only sharpen (inject high-frequency detail) and cannot move the
        low/mid-frequency field, which stays anchored to the warm-start + DC step.
        """
        delta = self._prox_net(k)(torch.cat([x, cond], dim=1))
        if self.prox_highpass_residual and self.prox_highpass_sigma_hr > 0:
            delta = delta - gaussian_blur_2d(delta, self.prox_highpass_sigma_hr)
        return x + delta

    def forward(
        self,
        x0: torch.Tensor,
        y_burst: torch.Tensor,
        shifts: torch.Tensor,
        psf: ScenePSF,
        cond: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
        return_iters: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the K-step solver.

        Args:
            x0:      (B,1,H,W) warm-start HR estimate (aligned mean or hybrid ch5 anchor).
            y_burst: (B,M,h,w) observed LR burst (the SAVED burst; FIXED M frames).
            shifts:  (B,M,2) the SAVED per-frame shifts in LR px (must match the burst).
            psf:     per-sample ScenePSF (from scene metadata: shape/sigma/sigma_y/angle).
            cond:    (B,cond_channels,H,W) conditioning for the prox (e.g. the hybrid obs).
            frame_mask: optional (B,M,1,1) weights (0 for padded/invalid frames).
        Returns: x_K, or (x_K, [x_1..x_K]) if return_iters.
        """
        if x0.shape[1] != 1:
            raise ValueError(f"x0 must be (B,1,H,W); got {tuple(x0.shape)}")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(f"cond must have {self.cond_channels} channels; got {cond.shape[1]}")
        create_graph = self.training
        huber = self.huber_delta if self.huber_delta > 0 else None
        forward_plan = None
        if _USE_FORWARD_PLAN_DEFAULT:
            forward_plan = prepare_forward_burst_plan(
                shifts,
                psf,
                self.scale,
                tuple(map(int, x0.shape[-2:])),
                dtype=x0.dtype,
            )
        x = x0
        iters: list[torch.Tensor] = []
        for k in range(self.n_steps):
            x = self._prox(k, x, cond)   # learned proximal step (null-space prior) — FIRST
            eta = F.softplus(self.eta_raw[k])
            g, _ = data_consistency_grad(
                x, y_burst, shifts, psf, self.scale,
                frame_mask=frame_mask,
                band_highpass_sigma_lr_px=self.band_highpass_sigma_lr_px,
                huber_delta=huber,
                create_graph=create_graph,
                plan=forward_plan,
            )
            x = x - eta * g              # DC gradient step (range constraint) — LAST: x_K ends on DC
            if return_iters:
                iters.append(x)
        return (x, iters) if return_iters else x

    @torch.no_grad()
    def dc_residual_rms(
        self, x: torch.Tensor, y_burst: torch.Tensor, shifts: torch.Tensor, psf: ScenePSF,
        *, band: bool = True,
    ) -> torch.Tensor:
        """Monitoring: RMS of (A x - y) (optionally in the highpass band). Used by Gate B to
        confirm the DC residual drives toward 0 on a clean overfit scene."""
        from .forward_torch import _highpass, forward_burst
        r = forward_burst(x, shifts, psf, self.scale) - y_burst
        if band and self.band_highpass_sigma_lr_px > 0:
            r = _highpass(r, self.band_highpass_sigma_lr_px)
        return torch.sqrt((r * r).mean())
