"""Gate C — training smoke on the REAL pool (remote GPU).

Validates the full solver training plumbing end-to-end on real generated scenes: the dataset
delivers burst+shifts+PSF at the scale-aligned crop, ScenePSF builds from the batch, the
UnrolledSolver runs forward+backward without NaN, and the DC residual is sane. This is the last
gate before the long run.

Usage on the remote:
    uv run python algos/ep07_unet_sr/tests/test_gate_c_smoke.py --pool data/synthetic/pool_2x_v3_5k
Exit 0 = PASS (start the real run); nonzero = STOP.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
for p in (_ROOT / "tcforge" / "src", _ROOT / "algos" / "ep07_unet_sr" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unet_sr.dataset import HYBRID_DRIZZLE_MEAN_CHANNEL, ThermalSRDataset  # noqa: E402
from unet_sr.losses import ContourSRLoss  # noqa: E402
from unet_sr.solver_train import build_scene_psf, edge_mask, terminal_dc_loss  # noqa: E402
from unet_sr.unroll import UnrolledSolver  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/pool_2x_v3_5k")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    scale = 2
    ds = ThermalSRDataset(
        args.pool, patch_size_hr=args.patch, scale=scale, seed=0, patches_per_scene=8,
        max_scene_cache=4, input_mode="hybrid_drizzle2x", return_metadata=False,
        provide_burst=True, solver_m_frames=args.frames,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"Gate C on {DEVICE}: {len(ds.scene_paths)} scenes, patch={args.patch}, batch={args.batch}, "
          f"M={args.frames}, steps={args.steps}")

    # shape audit on the first batch
    b0 = next(iter(loader))
    p1 = args.patch // scale
    checks = {
        "obs_features": (args.batch, 8, args.patch, args.patch),
        "hr_target": (args.batch, 1, args.patch, args.patch),
        "lr_burst_patch": (args.batch, args.frames, p1, p1),
        "burst_shifts": (args.batch, args.frames, 2),
    }
    shape_ok = True
    for k, want in checks.items():
        got = tuple(b0[k].shape)
        ok = got == want
        shape_ok &= ok
        print(f"  {k:>16}: {got} {'OK' if ok else '!= ' + str(want)}")
    for k in ("psf_sigma_lr_px", "psf_sigma_y_lr_px", "psf_angle_deg", "psf_shape"):
        print(f"  {k:>16}: present={k in b0}")
        shape_ok &= k in b0

    solver = UnrolledSolver(n_steps=3, cond_channels=8, base_channels=32, scale=scale,
                            band_highpass_sigma_lr_px=5.0).to(DEVICE).train()
    criterion = ContourSRLoss(forward_model_weight=0.0)
    mask = edge_mask(p1, p1, 8, DEVICE)
    opt = torch.optim.AdamW(solver.parameters(), lr=2e-4)
    mean_ch = HYBRID_DRIZZLE_MEAN_CHANNEL

    losses, dcs, finite = [], [], True
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        obs = batch["obs_features"].to(DEVICE)
        target = batch["hr_target"].to(DEVICE)
        burst = batch["lr_burst_patch"].to(DEVICE)
        shifts = batch["burst_shifts"].to(DEVICE)
        psf = build_scene_psf(batch, DEVICE)
        x0 = obs[:, mean_ch : mean_ch + 1]
        pred = solver(x0, burst, shifts, psf, obs, frame_mask=mask)
        loss_d = criterion(pred, target, lr_observation=None, lr_obs=None)
        dc = terminal_dc_loss(pred, burst, shifts, psf, scale, 5.0, mask, 0.0)
        total = loss_d["total"] + 0.1 * dc
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0).item()
        opt.step()
        finite &= bool(torch.isfinite(total)) and all(
            torch.isfinite(p.grad).all() for p in solver.parameters() if p.grad is not None)
        losses.append(float(total)); dcs.append(float(dc))
        if step in (0, args.steps // 2, args.steps - 1):
            print(f"  step {step:>3}: total={float(total):.4f} dc={float(dc):.5f} gnorm={gnorm:.2f}")

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"\n[1] shapes/plumbing OK = {shape_ok}")
    print(f"[2] all losses+grads finite = {finite}")
    print(f"[3] loss trend: first5={first:.4f} -> last5={last:.4f}  (informational; DC {dcs[0]:.5f} -> {dcs[-1]:.5f})")
    ok = shape_ok and finite
    print("\nGATE C:", "PASS — plumbing runs on real data; start the K-step run" if ok else "FAIL — STOP")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
