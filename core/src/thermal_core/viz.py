"""thermal_core.viz — 可复用可视化函数

Notebook 片段只需调用 plot_xxx() 即可产出学术级图表。
所有图表使用 FIGURE_SIZES 标准尺寸，确保字体与图面比例协调。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from thermal_core.plotting import (
    FIGURE_SIZES, METHOD_COLOR_LIST, savefig_academic, make_figure,
)


# ──────────────────────────────────────────────
# 覆盖率热力图
# ──────────────────────────────────────────────

def plot_coverage_heatmap(
    grid: np.ndarray,
    valid_coords: list[int],
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """绘制坐标覆盖率热力图 (repeat 计数)。"""
    vals = sorted(valid_coords)
    fig, ax = make_figure("double_col", height=5.5)
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=3, aspect="equal")

    for i in range(len(vals)):
        for j in range(len(vals)):
            val = grid[i, j]
            color = "white" if val >= 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals)
    ax.set_xlabel(r"X [$\mu$m]")
    ax.set_ylabel(r"Y [$\mu$m]")
    ax.set_title("Coordinate Coverage Map (frame count per coordinate)")
    plt.colorbar(im, ax=ax, label="Number of repeats", fraction=0.046, pad=0.04)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


# ──────────────────────────────────────────────
# 温度分布诊断图
# ──────────────────────────────────────────────

def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x and cumulative probability for an ECDF."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plot_temperature_histograms(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """绘制逐帧温度分布诊断图 (ECDF + rug + robust summary).

    普通直方图在本数据集上会被少量预热/补采帧拉宽横轴，并产生大量空 bin。
    ECDF 不依赖 bin，rug 直接显示每一帧的位置，能更清楚地区分主扫描主体、
    温度段跳变和少量离群帧。
    """
    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=3.2)

    configs = [
        ("T_mean", "(a) Frame Mean Temperature", "Mean"),
        ("T_std",  "(b) Frame Temperature Std",  "Std"),
        ("T_min",  "(c) Frame Min Temperature",  "Min"),
        ("T_max",  "(d) Frame Max Temperature",  "Max"),
    ]

    has_main_flag = "is_main_session" in df.columns
    main_mask = df["is_main_session"].astype(bool).to_numpy() if has_main_flag else None
    colors = {
        "all": METHOD_COLOR_LIST[0],
        "main": METHOD_COLOR_LIST[0],
        "other": METHOD_COLOR_LIST[2],
        "iqr": "#888888",
    }

    for ax, (col, title, label) in zip(axes.flat, configs):
        values = df[col].to_numpy(dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        n_out = int(((values < low_fence) | (values > high_fence)).sum())

        ax.axvspan(q1, q3, color=colors["iqr"], alpha=0.12, linewidth=0, label="IQR")
        ax.axvline(median, color="#333333", linewidth=0.9, linestyle="--", label="Median")

        if has_main_flag and main_mask is not None and main_mask.any() and (~main_mask).any():
            x_main, y_main = _ecdf_xy(values[main_mask])
            x_other, y_other = _ecdf_xy(values[~main_mask])
            ax.step(x_main, y_main, where="post", color=colors["main"],
                    linewidth=1.5, label=f"Main session (n={main_mask.sum()})")
            ax.step(x_other, y_other, where="post", color=colors["other"],
                    linewidth=1.2, label=f"Other frames (n={(~main_mask).sum()})")
            ax.plot(values[main_mask], np.full(main_mask.sum(), -0.035), "|",
                    color=colors["main"], markersize=5, alpha=0.55,
                    transform=ax.get_xaxis_transform(), clip_on=False)
            ax.plot(values[~main_mask], np.full((~main_mask).sum(), -0.075), "|",
                    color=colors["other"], markersize=7, alpha=0.9,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        else:
            x_all, y_all = _ecdf_xy(values)
            ax.step(x_all, y_all, where="post", color=colors["all"],
                    linewidth=1.5, label=f"All frames (n={len(values)})")
            ax.plot(values, np.full(len(values), -0.045), "|",
                    color=colors["all"], markersize=5, alpha=0.55,
                    transform=ax.get_xaxis_transform(), clip_on=False)

        ax.text(
            0.02, 0.96,
            f"median={median:.3f}°C\nIQR={iqr:.3f}°C\nTukey outliers={n_out}",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "#cccccc", "linewidth": 0.5, "alpha": 0.9},
        )
        ax.set_ylim(-0.09, 1.02)
        ax.set_xlabel(f"{label} Temperature [°C]")
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].legend(handles, labels, loc="center", bbox_to_anchor=(0.58, 0.34),
                        fontsize=7,
                        frameon=True, framealpha=0.92, borderpad=0.3,
                        handlelength=1.2, handletextpad=0.4)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


# ──────────────────────────────────────────────
# 温度时间线
# ──────────────────────────────────────────────

def plot_temperature_timeline(
    df_sorted: pd.DataFrame,
    *,
    order_label: str = "filename order",
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """绘制逐帧温度时间线 (均温 + 范围带)。

    使用粗线 + 填充带替代微小散点，确保可读性。
    """
    fig, ax = make_figure("double_col", height=3.5)
    idx = df_sorted.index

    # 温度范围带 (min–max)
    ax.fill_between(idx, df_sorted["T_min"], df_sorted["T_max"],
                    alpha=0.15, color=METHOD_COLOR_LIST[0], label="T range (min–max)")
    # 均温线
    ax.plot(idx, df_sorted["T_mean"], "-",
            linewidth=1.2, color=METHOD_COLOR_LIST[0], label="Mean temperature")

    ax.set_xlabel(f"Frame Index ({order_label})")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Per-Frame Temperature Timeline ({order_label})")
    ax.legend(loc="upper right")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


# ──────────────────────────────────────────────
# Session 检测可视化
# ──────────────────────────────────────────────

def plot_sessions(
    df_sorted: pd.DataFrame,
    session_ids: np.ndarray,
    break_indices: np.ndarray,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """绘制 Session 检测图 — 彩色段 + 断点标注。

    用粗色带替代微小散点，每个 session 显示为连续色段，
    断点处用垂直线 + 温度跳变标注。
    """
    n_sessions = int(session_ids.max()) + 1
    fig, ax = make_figure("double_col", height=3.5)

    # 为每个 session 绘制粗线段（而非散点），颜色循环
    for sid in range(n_sessions):
        mask = session_ids == sid
        subset = df_sorted.loc[mask]
        color = METHOD_COLOR_LIST[sid % len(METHOD_COLOR_LIST)]
        ax.plot(subset.index, subset["T_mean"], "-",
                linewidth=2.0, color=color,
                label=f"S{sid} (n={mask.sum()})")

    # 断点标注：垂直线 + ΔT 标注
    t_means = df_sorted["T_mean"].values
    for brk in break_indices:
        ax.axvline(brk + 0.5, color="#444444", linestyle="--",
                   linewidth=0.8, alpha=0.7)
        dt = t_means[brk + 1] - t_means[brk]
        ax.annotate(
            f"Δ{dt:+.1f}°C",
            xy=(brk + 0.5, t_means[brk]),
            xytext=(0, 12), textcoords="offset points",
            fontsize=7, ha="center", color="#444444",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
        )

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Mean Temperature [°C]")
    ax.set_title(f"Session Detection ({n_sessions} sessions)")

    # Legend: 如果 session 太多，放在图外下方
    if n_sessions > 6:
        ax.legend(fontsize=7, ncol=4,
                  bbox_to_anchor=(0.5, -0.18), loc="upper center")
    else:
        ax.legend(loc="best")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


# ──────────────────────────────────────────────
# EP02 displacement calibration
# ──────────────────────────────────────────────

def plot_displacement_field(
    measurements: pd.DataFrame,
    *,
    theta_deg: float | None = None,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot measured displacement vectors for X/Y scan pairs."""
    fig, ax = make_figure("double_col", height=4.0)

    for axis, color, label in [
        ("x", METHOD_COLOR_LIST[0], "X-scan pairs"),
        ("y", METHOD_COLOR_LIST[1], "Y-scan pairs"),
    ]:
        subset = measurements[measurements["scan_axis"] == axis]
        if subset.empty:
            continue
        ax.scatter(
            subset["dx_px"],
            subset["dy_px"],
            s=16,
            alpha=0.45,
            color=color,
            edgecolor="none",
            label=label,
        )
        means = subset.groupby("delta_um")[["dx_px", "dy_px"]].mean().reset_index()
        for _, row in means.iterrows():
            ax.arrow(
                0,
                0,
                row["dx_px"],
                row["dy_px"],
                width=0.003,
                head_width=0.025,
                length_includes_head=True,
                color=color,
                alpha=0.95,
            )

    if theta_deg is not None:
        from thermal_core.displacement import coordinate_to_shift

        ax.plot([], [], color=METHOD_COLOR_LIST[2], lw=2.0, label=f"Reference model ({theta_deg:.1f}°)")
        for delta_x, delta_y in [
            (2, 0),
            (4, 0),
            (0, 2),
            (0, 4),
        ]:
            dx, dy = coordinate_to_shift(delta_x, delta_y, theta_deg)
            ax.arrow(
                0,
                0,
                dx,
                dy,
                width=0.0018,
                head_width=0.018,
                length_includes_head=True,
                color=METHOD_COLOR_LIST[2],
                alpha=0.75,
            )

    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Measured dx [px]")
    ax.set_ylabel("Measured dy [px]")
    ax.set_title("Measured Displacement Vector Field")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal", adjustable="box")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_theta_bootstrap(
    theta_samples: np.ndarray,
    ci_bounds: tuple[float, float],
    *,
    reference_deg: float = 47.6,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot the bootstrap distribution of fitted theta."""
    fig, ax = make_figure("double_col", height=3.5)
    ax.hist(
        theta_samples,
        bins=50,
        color=METHOD_COLOR_LIST[0],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )
    ax.axvline(ci_bounds[0], color=METHOD_COLOR_LIST[2], ls="--", lw=1.0, label="95% CI")
    ax.axvline(ci_bounds[1], color=METHOD_COLOR_LIST[2], ls="--", lw=1.0)
    ax.axvline(reference_deg, color="#444444", ls=":", lw=1.2, label=f"Reference {reference_deg:.1f}°")
    ax.set_xlabel(r"Fitted $\theta$ [deg]")
    ax.set_ylabel("Bootstrap Count")
    ax.set_title(r"Bootstrap Distribution of Rotation Angle $\theta$")
    ax.legend(loc="best")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_linearity(
    nominal: np.ndarray,
    measured: np.ndarray,
    fit_params: dict,
    *,
    xlabel: str = "Nominal displacement [px]",
    ylabel: str = "Measured displacement [px]",
    title: str = "Stage Linearity Check",
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot linearity regression and residuals."""
    x = np.asarray(nominal, dtype=float)
    y = np.asarray(measured, dtype=float)
    pred = fit_params["slope"] * x + fit_params["intercept"]
    order = np.argsort(x)
    residual = y - pred

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.2)
    ax0, ax1 = axes

    ax0.scatter(x, y, s=18, alpha=0.75, color=METHOD_COLOR_LIST[0], edgecolor="none")
    ax0.plot(x[order], pred[order], color=METHOD_COLOR_LIST[2], lw=1.5)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_title(title)
    ax0.text(
        0.03,
        0.97,
        f"a={fit_params['slope']:.4f}\nb={fit_params['intercept']:.4f}\nR²={fit_params['r2']:.4f}",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

    ax1.scatter(x, residual, s=18, alpha=0.75, color=METHOD_COLOR_LIST[1], edgecolor="none")
    ax1.axhline(0, color="#555555", ls="--", lw=0.8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Residual [px]")
    ax1.set_title("Regression Residuals")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig
