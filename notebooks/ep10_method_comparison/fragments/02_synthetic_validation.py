# %% [markdown]
# ## Synthetic And Run Sanity
#
# 先把三套算法留下的 JSON 产物并排检查。这里不是重新运行合成实验，而是确认每条算法链路是否留下了有限值、坐标或正则行为的基本健康记录。

# %%
import numpy as np

sanity_rows = []

drizzle_syn = artifacts["drizzle"].get("synthetic", {})
if drizzle_syn:
    sanity_rows.append(
        {
            "method": "Drizzle",
            "status": bool(drizzle_syn.get("coordinate_check_pass", False)),
            "metric": "PSNR dB",
            "baseline_value": drizzle_syn.get("baseline_psnr_db"),
            "method_value": drizzle_syn.get("drizzle_psnr_db"),
            "delta_or_ratio": drizzle_syn.get("drizzle_psnr_db") - drizzle_syn.get("baseline_psnr_db"),
            "detail": f"pixfrac={drizzle_syn.get('pixfrac', 'n/a')}",
        }
    )
else:
    sanity_rows.append({"method": "Drizzle", "status": "missing", "detail": "synthetic_validation.json"})

map_syn = artifacts["map_tv"].get("synthetic", {})
if map_syn:
    sanity_rows.append(
        {
            "method": "MAP-TV",
            "status": bool(map_syn.get("map_tv_finite", False)),
            "metric": "PSNR dB",
            "baseline_value": map_syn.get("bicubic_mean_psnr_db"),
            "method_value": map_syn.get("map_tv_psnr_db"),
            "delta_or_ratio": map_syn.get("map_tv_psnr_db") - map_syn.get("bicubic_mean_psnr_db"),
            "detail": f"lambda={map_syn.get('lambda_tv', 'n/a')}, sigma={map_syn.get('psf_sigma', 'n/a')}",
        }
    )
else:
    sanity_rows.append({"method": "MAP-TV", "status": "missing", "detail": "synthetic_validation.json"})

tgv_syn = artifacts["tgv"].get("synthetic", {})
if tgv_syn:
    sanity_rows.append(
        {
            "method": "TGV",
            "status": bool(tgv_syn.get("passed", False)),
            "metric": "MSE vs TV",
            "baseline_value": tgv_syn.get("tv_mse"),
            "method_value": tgv_syn.get("tgv_mse"),
            "delta_or_ratio": tgv_syn.get("tgv_mse") / tgv_syn.get("tv_mse"),
            "detail": f"noisy_mse={tgv_syn.get('noisy_mse', np.nan):.4g}",
        }
    )
else:
    sanity_rows.append({"method": "TGV", "status": "missing", "detail": "synthetic_validation.json"})

sanity_table = pd.DataFrame(sanity_rows)
display(sanity_table.round(4))

# %%
run_rows = []
small_real = artifacts["drizzle"].get("small_real", {})
if small_real:
    run_rows.append(
        {
            "method": "Drizzle",
            "run_note": "small real check",
            "frames_or_rows": small_real.get("n_frames"),
            "best_label": f"pf={small_real.get('pixfrac', 'n/a')}",
            "extra": f"center corr={small_real.get('center_crop_corr_vs_bicubic_mean', np.nan):.4f}",
        }
    )

map_best = artifacts["map_tv"].get("best", {})
if map_best:
    map_tv_best = best_rows.get("MAP-TV")
    run_rows.append(
        {
            "method": "MAP-TV",
            "run_note": "Pareto top",
            "frames_or_rows": len(sweeps["MAP-TV"]),
            "best_label": (
                map_tv_best.get("variant", "missing")
                if map_tv_best is not None
                else "missing"
            ),
            "extra": f"frontier_count={map_best.get('frontier_count', 'n/a')}",
        }
    )

tgv_run = artifacts["tgv"].get("run_summary", {})
if tgv_run:
    run_rows.append(
        {
            "method": "TGV",
            "run_note": "full-session sweep",
            "frames_or_rows": tgv_run.get("n_frames"),
            "best_label": tgv_run.get("best_label"),
            "extra": f"elapsed={tgv_run.get('elapsed_sec', np.nan) / 3600:.2f} h",
        }
    )

if run_rows:
    display(pd.DataFrame(run_rows))

# %% [markdown]
# > **数据说明**: 第一张表读取三算法的 `synthetic_validation.json`，第二张表读取 Drizzle 小真实检查、MAP-TV Pareto 记录和 TGV 运行摘要。Drizzle/MAP-TV 的数值是 toy synthetic PSNR，`delta_or_ratio` 表示方法相对 baseline 增加了多少 dB；TGV 的数值是 toy MSE，`delta_or_ratio` 表示 TGV MSE / TV MSE。
# >
# > **怎么看**: `status=True` 表示对应算法产物通过了最小健康检查。PSNR 越大越好；TGV 的 MSE 和 MSE ratio 越小越好。它们只说明实现链路没有明显崩坏，不是主 session 的真实光学真值。`best_label` 用于后续选择可视化候选。
# >
# > **异常是否正常**: 三个 synthetic 场景并不完全相同，所以不能把 Drizzle 的 synthetic PSNR、MAP-TV 的 synthetic PSNR 和 TGV 的 synthetic MSE 当成公平算法排行榜。缺失 JSON 只代表该 sanity 记录没有生成。
# >
# > **核心发现**: 三算法比较必须回到同一真实 248 clean-frame comparison 的 proxy 指标和中心 ROI highpass 图；synthetic 表只负责排除明显实现/路径问题。
