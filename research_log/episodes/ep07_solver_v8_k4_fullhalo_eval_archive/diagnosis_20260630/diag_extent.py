"""Causal test of the GroupNorm/SE 'extent distribution-shift' hypothesis.

No trained weights needed: extent-invariance is an ARCHITECTURAL property.
We test whether the prox UNet's prediction at a fixed window depends on image
content FAR OUTSIDE that window's convolutional receptive field. If it does,
the network is NOT extent-invariant => switching from 192-tile inference to
full-frame/large-halo inference necessarily changes every pixel.

Variants (identical conv weights; we ablate only the global ops):
  full        : GroupNorm + SE  (the real architecture)
  noSE        : SE -> Identity
  noGN_noSE   : GroupNorm -> Identity AND SE -> Identity (pure conv UNet, finite RF)

Tests:
  D2 far-field : perturb only the outer 64-px frame of a 768^2 field, far beyond
                 the RF of a central 96^2 window. Report normalized RMS change in
                 that window. Pure conv (noGN_noSE) MUST be ~0 (validates margin).
  D1 crop/full : compare net(full)[center192] vs net(192 tile) interior 96^2.
  K-amp        : apply residual prox K times; does the crop/full gap grow with K?
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

def _repo_root():
    p = Path(__file__).resolve()
    for q in [p, *p.parents]:
        if (q / "AGENTS.md").exists():
            return q
    return p.parents[3]
ROOT = _repo_root()
OUT = ROOT / "outputs" / "ep07_solver_diag"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"algos/ep07_unet_sr/src"))
from unet_sr.model import ThermalSRUNet, SEBlock

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
IN_CH = 6  # no-drizzle prox: [x] + 5 cond channels

def build_variants():
    torch.manual_seed(0)
    net = ThermalSRUNet(in_channels=IN_CH, out_channels=1, base_channels=64, scale=1).eval()
    full = net
    def strip(net, gn, se):
        m = copy.deepcopy(net)
        for mod in m.modules():
            for name, child in list(mod.named_children()):
                if se and isinstance(child, SEBlock):
                    setattr(mod, name, nn.Identity())
                if gn and isinstance(child, nn.GroupNorm):
                    setattr(mod, name, nn.Identity())
        return m.eval()
    return {"full": full, "noSE": strip(net, False, True),
            "noGN_noSE": strip(net, True, True)}

def make_field(n, seed=0):
    """Realistic-ish prox input: cold textured background + a hot block + thin warm lines."""
    g = np.random.default_rng(seed)
    bg = 0.2 * g.standard_normal((IN_CH, n, n)).astype(np.float32)
    bg += 22.0
    # hot chip block (off-center) raises all channels
    bg[:, n//3:n//3+n//4, n//3:n//3+n//3] += 3.0
    # a few thin warm lines through the center window
    for k in range(3):
        c = n//2 - 20 + 14*k
        bg[:, c:c+2, n//4:3*n//4] += 2.0
    return torch.from_numpy(bg)[None]

@torch.no_grad()
def run(net, x): return net(x)[0, 0].numpy()

def nrms(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d*d)) / (np.std(a) + 1e-9))

variants = build_variants()

# ---------- D2: far-field perturbation (clean isolation) ----------
N = 768; ws = 96
y0 = x0 = N//2 - ws//2
X = make_field(N, seed=1)
X2 = X.clone()
ring = 64
pert = torch.zeros_like(X2); pert[..., :ring, :] = 3.0; pert[..., -ring:, :] = 3.0
pert[..., :, :ring] = 3.0; pert[..., :, -ring:] = 3.0   # warm the far outer frame only
X2 = X2 + pert
dist = (y0) - ring   # gap between perturbation and window top edge
print(f"D2 field={N} window={ws}@center  perturb=outer {ring}px frame  gap-to-window={dist}px")
print(f"{'variant':12s} {'farfield_nrms(window)':>22s}")
d2 = {}
for name, net in variants.items():
    a = run(net, X)[y0:y0+ws, x0:x0+ws]
    b = run(net, X2)[y0:y0+ws, x0:x0+ws]
    d2[name] = nrms(a, b)
    print(f"{name:12s} {d2[name]:22.4e}")

# ---------- D1: crop-vs-full interior ----------
P = 192; inner = 96
cy = cx = N//2 - P//2
iy = ix = N//2 - inner//2
tile_in = X[..., cy:cy+P, cx:cx+P]
print(f"\nD1 full={N} tile={P}  interior={inner}^2")
print(f"{'variant':12s} {'interior_nrms(full vs tile)':>28s}")
d1 = {}
for name, net in variants.items():
    full_out = run(net, X)[iy:iy+inner, ix:ix+inner]
    tile_out = run(net, tile_in)
    ti = tile_out[P//2-inner//2:P//2+inner//2, P//2-inner//2:P//2+inner//2]
    d1[name] = nrms(full_out, ti)
    print(f"{name:12s} {d1[name]:28.4e}")

# ---------- K-amplification (residual prox loop, shared weights) ----------
# x_{k+1} = x_k + alpha*net([x_k, cond]); measure interior crop/full gap vs K.
print("\nK-amplification (full architecture, alpha=0.1): interior nrms(full vs tile) per step")
net = variants["full"]; alpha = 0.1
Nk = 512; Pk = 192; ik = 96
cyk = Nk//2 - Pk//2; iyk = Nk//2 - ik//2
Xk = make_field(Nk, seed=2)
cond_full = Xk[:, 1:].clone()                       # 5 cond channels (fixed)
cond_tile = cond_full[..., cyk:cyk+Pk, cyk:cyk+Pk].clone()
xf = Xk[:, :1].clone()                               # warm-start x (full)
xt = xf[..., cyk:cyk+Pk, cyk:cyk+Pk].clone()         # warm-start x (tile)
@torch.no_grad()
def prox_step(net, x, cond):
    return x + alpha * net(torch.cat([x, cond], 1))
for k in range(1, 5):
    xf = prox_step(net, xf, cond_full)
    xt = prox_step(net, xt, cond_tile)
    fi = xf[0,0, iyk:iyk+ik, iyk:iyk+ik].numpy()
    tic = xt[0,0, Pk//2-ik//2:Pk//2+ik//2, Pk//2-ik//2:Pk//2+ik//2].numpy()
    print(f"  K={k}:  interior_nrms={nrms(fi, tic):.4e}")

import json
(OUT/"metrics_extent.json").write_text(
    json.dumps({"D2_farfield": d2, "D1_interior": d1}, indent=2))
print("\nInterpretation:")
print(" - noGN_noSE D2 ~ 0  => pure conv UNet is extent-invariant (RF is finite).")
print(" - full/noSE D2 >> 0 => GroupNorm/SE couple far-field content into the window")
print("   => prox prediction depends on SOLVE EXTENT (192 tile vs full frame).")
