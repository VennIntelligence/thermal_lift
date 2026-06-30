"""Refined array diagnosis + evidence figures (EP07 solver V8/K4)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    for q in [p, *p.parents]:
        if (q / "AGENTS.md").exists():
            return q
    return p.parents[3]
ROOT = _repo_root()
OUT = ROOT / "outputs" / "ep07_solver_diag"; OUT.mkdir(parents=True, exist_ok=True)
def _find_npz():
    c = sorted(ROOT.glob("remote_inbox/**/v8k4_step10000_render_arrays.npz"))
    if not c:
        raise FileNotFoundError("v8k4_step10000_render_arrays.npz not found under remote_inbox/")
    return c[0]
NPZ = _find_npz()
z = np.load(NPZ)
KEYS = ["aligned_mean","tiled_p192_o128","dense_p192_o160","full_halo96",
        "tile_halo32","tile_halo64","tile_halo96"]
R = {k: z[k].astype(np.float64) for k in KEYS}
base = R["aligned_mean"]; H, W = base.shape
hp = lambda a,s: a - gaussian_filter(a,s,mode="nearest")

# ---- autocorrelation seam detector (separates tile step 64 vs 32) ----
def seam_autocorr(diff, axis, lags=(16,32,48,64,96,128)):
    d = hp(diff, 12)
    prof = np.mean(np.abs(d), axis=axis)          # 1D along the seam direction
    prof = prof - prof.mean()
    n = prof.size
    ac = np.correlate(prof, prof, mode="full")[n-1:]
    ac = ac / (ac[0] + 1e-12)
    return {L: float(ac[L]) for L in lags}

print("=== seam autocorrelation (x-direction = vertical seams), normalized ===")
print("%-18s " % "render" + " ".join("ac%-3d" % L for L in (16,32,48,64,96,128)))
seam_ac = {}
for k in KEYS:
    ac = seam_autocorr(R[k]-base, axis=0)
    seam_ac[k] = ac
    print("%-18s " % k + " ".join("%5.2f" % ac[L] for L in (16,32,48,64,96,128)))

# ---- chip-region flocculence (texture OFF strong edges, INSIDE the chip) ----
def grad_mag(a):
    gy,gx = np.gradient(a); return np.hypot(gx,gy)
chip = base > np.percentile(base, 75)             # hot chip pixels
edges = grad_mag(gaussian_filter(base,2,mode="nearest"))
flat_chip = chip & (edges < np.percentile(edges[chip], 40))   # chip interior, off-edge
print("\nchip-interior off-edge pixels:", int(flat_chip.sum()))

# ---- finer-scale flocculence band (periods 2..8 HR px) ----
fy = np.fft.fftfreq(H); fx = np.fft.fftfreq(W)
FX,FY = np.meshgrid(fx,fy); rad = np.hypot(FX,FY)
fine = (rad>=1/8)&(rad<=1/2.2)
metrics = {}
print("\n%-18s %9s %9s %9s %9s %9s" % ("render","chipTexσ","finefloc","P95grad","bgLift","meanΔ"))
for k in KEYS:
    sh = hp(R[k],4)
    chiptex = float(np.std(sh[flat_chip]))
    F=np.fft.fft2(sh); P=(F*np.conj(F)).real/(H*W)
    finefloc=float(P[fine].mean())
    sy0,sx0=367,481
    p95=float(np.percentile(grad_mag(R[k][sy0:sy0+192,sx0:sx0+192]),95))
    bg = R[k] < np.percentile(R[k],40)
    bglift=float(np.mean((R[k]-base)[bg]))
    md=float(np.mean(R[k]-base))
    metrics[k]=dict(chip_tex=chiptex,fine_floc=finefloc,p95=p95,bg_lift=bglift,mean=md,
                    seam_ac=seam_ac[k])
    print("%-18s %9.4f %9.2e %9.4f %9.4f %9.4f"%(k,chiptex,finefloc,p95,bglift,md))
(OUT/"metrics_arrays2.json").write_text(json.dumps(metrics,indent=2))

# ================== FIGURES ==================
def cc(a, f=1/3):
    r,c=a.shape; rr=int(r*f); cc=int(c*f)
    return a[r//2-rr//2:r//2+rr//2, c//2-cc//2:c//2+cc//2]
vmin,vmax=np.percentile(base,1),np.percentile(base,99)

# Fig A: 7 renders center-crop temperature
fig,axes=plt.subplots(2,4,figsize=(16,6.2))
for ax,k in zip(axes.ravel(),KEYS):
    ax.imshow(cc(R[k]),cmap="inferno",vmin=vmin,vmax=vmax); ax.set_title(k,fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
axes.ravel()[-1].axis("off")
fig.suptitle("EP07 V8/K4 step10000 — center 1/3 temperature",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"figA_renders.png",dpi=140); plt.close(fig)

# Fig B: the two key monotonic trends (context axis)
order=["tiled_p192_o128","tile_halo32","tile_halo64","tile_halo96","full_halo96"]
ctx=[0,32,64,96,200]  # nominal extra solve-context (HR px); full_halo ~ whole frame
fig,ax=plt.subplots(1,2,figsize=(11,4))
prom=[max(abs(seam_ac[k][64]),abs(seam_ac[k][32])) for k in order]
ax[0].plot(ctx,prom,"o-"); ax[0].set_title("seam periodicity (autocorr) vs solve context")
ax[0].set_xlabel("extra solve context [HR px] (full_halo≈whole frame)"); ax[0].set_ylabel("max seam autocorr")
bl=[metrics[k]["bg_lift"] for k in order]
ax[1].plot(ctx,bl,"s-",color="C3"); ax[1].set_title("background lift vs solve context")
ax[1].set_xlabel("extra solve context [HR px]"); ax[1].set_ylabel("bg temp lift vs aligned [°C]")
for a in ax:
    for x,k in zip(ctx,order): a.annotate(k.replace("_p192_o128","").replace("tile_",""),(x,a.get_ylim()[0]),fontsize=6,rotation=90,va="bottom")
fig.tight_layout(); fig.savefig(OUT/"figB_context_tradeoff.png",dpi=140); plt.close(fig)

# Fig C: seam axis spectrum tiled vs dense vs full_halo
fig,ax=plt.subplots(figsize=(9,4))
for k,c in [("tiled_p192_o128","C0"),("dense_p192_o160","C1"),("full_halo96","C2")]:
    d=hp(R[k]-base,12); prof=np.mean(np.abs(d),0); prof=prof-prof.mean()
    F=np.abs(np.fft.rfft(prof))**2; f=np.fft.rfftfreq(W)
    per=1/np.maximum(f,1e-9)
    m=(per>=10)&(per<=160)
    ax.plot(per[m],F[m],c,label=k,lw=1.2)
for p in (32,64,96): ax.axvline(p,color="gray",ls=":",lw=.8)
ax.set_xlabel("spatial period [HR px]"); ax.set_ylabel("seam power (x-marginal)")
ax.set_title("Vertical-seam spectrum: peaks at tile step (64) / harmonic (32); full_halo flat")
ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figC_seam_spectrum.png",dpi=140); plt.close(fig)

# Fig D: flat-background ROI zoom (grid vs no-grid)
fy0,fx0,fsz=724,755,96
fig,axs=plt.subplots(1,4,figsize=(14,3.6))
for ax,k in zip(axs,["aligned_mean","tiled_p192_o128","tile_halo64","full_halo96"]):
    roi=R[k][fy0:fy0+fsz,fx0:fx0+fsz]
    ax.imshow(roi-roi.mean(),cmap="RdBu_r",vmin=-0.08,vmax=0.08); ax.set_title(k,fontsize=9)
    ax.set_xticks([]);ax.set_yticks([])
fig.suptitle("Flat-background ROI (mean-removed): tiled shows the grid; halo removes it",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"figD_flatroi.png",dpi=140); plt.close(fig)

# Fig E: line profile across a thin structure (overshoot/swelling)
# pick a row through the sharp ROI with strong gradient
sy0,sx0=367,481
row=sy0+96
xs=slice(sx0,sx0+192)
fig,ax=plt.subplots(figsize=(10,4))
for k,c in [("aligned_mean","k"),("tiled_p192_o128","C0"),("full_halo96","C2"),("tile_halo64","C1")]:
    ax.plot(np.arange(192),R[k][row,xs],c,label=k,lw=1.1)
ax.set_title(f"Line profile across structures (row {row}): full_halo over/under-shoots (swelling)")
ax.set_xlabel("x [HR px]"); ax.set_ylabel("temperature [°C]"); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"figE_lineprofile.png",dpi=140); plt.close(fig)

print(f"\nFigures written to {OUT}")
for p in sorted(OUT.glob("fig*.png")): print(" ",p.name)
