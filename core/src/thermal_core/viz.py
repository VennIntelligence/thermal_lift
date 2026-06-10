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
    FIGURE_SIZES,
    METHOD_COLOR_LIST,
    METHOD_COLORS,
    make_figure,
    savefig_academic,
    setup_academic_style,
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
    fig, ax = make_figure("one_half_col", height=4.2)
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=3, aspect="equal")

    for i in range(len(vals)):
        for j in range(len(vals)):
            val = grid[i, j]
            color = "white" if val >= 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=color)

    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals)
    ax.set_xlabel(r"X [$\mu$m]")
    ax.set_ylabel(r"Y [$\mu$m]")
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


def _plot_ecdf_step(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 1.2,
    fill_alpha: float = 0.10,
) -> None:
    """Draw a single ECDF step curve with optional fill under the curve."""
    x, y = _ecdf_xy(values)
    ax.step(x, y, where="post", color=color, linewidth=linewidth, label=label, zorder=2)
    x_fill = np.concatenate(([x[0]], x))
    y_fill = np.concatenate(([0.0], y))
    ax.fill_between(x_fill, 0.0, y_fill, step="post", color=color, alpha=fill_alpha, linewidth=0)


def plot_temperature_histograms(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """绘制逐帧温度分布诊断图 (ECDF 阶梯曲线 + 底部抖动散点 + 稳健统计值).

    普通直方图会被少量预热/补采帧拉宽横轴；KDE 在早期帧样本过少时不稳定。
    ECDF 对小样本和主 session 大样本都稳定，阶梯曲线可直接读出分位点，
    底部抖动散点保留单帧在温度轴上的位置，便于与 session 分段对照。
    """
    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=3.6)

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
    }

    rng = np.random.default_rng(42)

    for ax, (col, title, label) in zip(axes.flat, configs):
        values = df[col].to_numpy(dtype=float)
        v_min, v_max = values.min(), values.max()
        v_range = v_max - v_min if v_max > v_min else 1.0
        x_pad = 0.08 * v_range

        if has_main_flag and main_mask is not None and main_mask.any() and (~main_mask).any():
            main_vals = values[main_mask]
            other_vals = values[~main_mask]

            _plot_ecdf_step(
                ax, main_vals,
                color=colors["main"],
                label=f"Main session (n={len(main_vals)})",
                linewidth=1.2,
                fill_alpha=0.12,
            )
            _plot_ecdf_step(
                ax, other_vals,
                color=colors["other"],
                label=f"Other frames (n={len(other_vals)})",
                linewidth=1.0,
                fill_alpha=0.08,
            )

            y_main_dots = rng.uniform(-0.07, -0.03, size=len(main_vals))
            ax.scatter(main_vals, y_main_dots, color=colors["main"], alpha=0.45, s=6, edgecolors="none", zorder=3)

            y_other_dots = rng.uniform(-0.15, -0.11, size=len(other_vals))
            ax.scatter(other_vals, y_other_dots, color=colors["other"], alpha=0.8, s=10, edgecolors="none", zorder=3)

            ax.set_ylim(-0.18, 1.05)
        else:
            _plot_ecdf_step(
                ax, values,
                color=colors["all"],
                label=f"All frames (n={len(values)})",
                linewidth=1.2,
                fill_alpha=0.12,
            )
            y_dots = rng.uniform(-0.07, -0.03, size=len(values))
            ax.scatter(values, y_dots, color=colors["all"], alpha=0.45, s=6, edgecolors="none", zorder=3)
            ax.set_ylim(-0.10, 1.05)

        target_name = "Main Session" if (has_main_flag and main_mask is not None and main_mask.any()) else "All"
        target_vals = values[main_mask] if (has_main_flag and main_mask is not None and main_mask.any()) else values

        q1, median, q3 = np.quantile(target_vals, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        n_out = int(((target_vals < low_fence) | (target_vals > high_fence)).sum())

        if "Std" in label:
            stats_text = (
                f"{target_name}:\n"
                f"Median: {median:.4f}\n"
                f"IQR: {iqr:.4f}\n"
                f"Outliers: {n_out}"
            )
            ax.set_xlabel(f"{label}")
        else:
            stats_text = (
                f"{target_name}:\n"
                f"Median: {median:.3f}°C\n"
                f"IQR: {iqr:.3f}°C\n"
                f"Outliers: {n_out}"
            )
            ax.set_xlabel(f"{label} Temperature [°C]")

        ax.text(
            0.5, 0.95,
            stats_text,
            transform=ax.transAxes,
            ha="center", va="top", fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": "#cccccc", "linewidth": 0.5, "alpha": 0.85},
        )

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Cumulative fraction")
        ax.set_xlim(v_min - x_pad, v_max + x_pad)
        ax.set_title(title)
        ax.grid(axis="both", alpha=0.2, linewidth=0.5, linestyle=":")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.02)
    try:
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(rect=(0, 0.05, 1, 1))
    except Exception:
        pass

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
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal", adjustable="box")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_avi_theta_bracket_summary(
    summary: pd.DataFrame,
    *,
    method: str = "gradient",
    reference_deg: float = 47.6,
    xlim: tuple[float, float] = (44.0, 50.0),
    output_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Pooled AVI theta bracket summary for presentation slides.

    Shows Y-scan / Combined / X-scan pooled estimates on a narrowed degree axis,
    with the combined 95% CI band and a bracket spanning the X/Y pooled means.
    """
    setup_academic_style()

    def _row(source: str) -> pd.Series:
        mask = (summary["method"] == method) & (summary["source"] == source)
        if not mask.any():
            raise ValueError(f"missing {method!r} summary row for source={source!r}")
        return summary.loc[mask].iloc[0]

    y_row = _row("y-only")
    x_row = _row("x-only")
    comb_row = _row("combined")

    colors = {
        "y": METHOD_COLORS["accent_1"],
        "x": METHOD_COLORS["primary"],
        "combined": METHOD_COLORS["secondary"],
    }
    markers = {"y": "o", "combined": "D", "x": "s"}
    labels = {
        "y": "Y-scan pooled",
        "combined": "Combined pooled",
        "x": "X-scan pooled",
    }

    fig, ax = make_figure("one_half_col", height=2.75)

    comb_lo = float(comb_row["ci_lower_deg"])
    comb_hi = float(comb_row["ci_upper_deg"])
    ax.axvspan(comb_lo, comb_hi, color=colors["combined"], alpha=0.14, zorder=0)
    ax.axvline(reference_deg, color="#444444", ls=":", lw=1.2, zorder=1)

    y_mark = 1.0
    series = [
        ("y", float(y_row["mean_deg"]), float(y_row["ci_lower_deg"]), float(y_row["ci_upper_deg"])),
        ("combined", float(comb_row["mean_deg"]), comb_lo, comb_hi),
        ("x", float(x_row["mean_deg"]), float(x_row["ci_lower_deg"]), float(x_row["ci_upper_deg"])),
    ]
    for key, mean_deg, lo, hi in series:
        ax.errorbar(
            mean_deg,
            y_mark,
            xerr=[[mean_deg - lo], [hi - mean_deg]],
            fmt=markers[key],
            ms=7.5 if key == "combined" else 6.0,
            mew=0.9,
            capsize=3.5,
            lw=1.1,
            color=colors[key],
            ecolor=colors[key],
            zorder=3,
        )
        ax.text(
            mean_deg,
            y_mark + 0.22,
            f"{mean_deg:.2f}°",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if key == "combined" else "normal",
            color=colors[key],
        )
        ax.text(
            mean_deg,
            y_mark - 0.18,
            labels[key],
            ha="center",
            va="top",
            fontsize=7.0,
            color="#333333",
        )

    bracket_y = 0.58
    y_mean = float(y_row["mean_deg"])
    x_mean = float(x_row["mean_deg"])
    ax.plot(
        [y_mean, x_mean],
        [bracket_y, bracket_y],
        color="#666666",
        lw=0.9,
        solid_capstyle="butt",
        zorder=2,
    )
    for x_pos in (y_mean, x_mean):
        ax.plot(
            [x_pos, x_pos],
            [bracket_y, bracket_y + 0.07],
            color="#666666",
            lw=0.9,
            zorder=2,
        )
    ax.plot(
        reference_deg,
        bracket_y,
        marker="|",
        markersize=11,
        color="#333333",
        mew=1.4,
        zorder=4,
    )
    ax.text(
        reference_deg,
        bracket_y - 0.11,
        f"Ref {reference_deg:.1f}°",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )

    comb_mean = float(comb_row["mean_deg"])
    delta_ref = abs(comb_mean - reference_deg)
    covers_ref = bool(comb_lo <= reference_deg <= comb_hi)
    caption = (
        f"{method.title()} NCC · n={int(comb_row['n'])} AVI · "
        f"Combined {comb_mean:.2f}° · |Δref|={delta_ref:.2f}°"
    )
    if covers_ref:
        caption += f" · 95% CI [{comb_lo:.2f}°, {comb_hi:.2f}°] covers {reference_deg:.1f}°"
    ax.text(
        0.5,
        0.04,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#444444",
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(0.35, 1.45)
    ax.set_xlabel(r"Pooled $\theta$ estimate [deg]")
    ax.set_title("AVI Rotation Angle Validation (Pooled Summary)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.22, linewidth=0.5)

    if save_fn and output_path:
        save_fn(fig, output_path)
    elif output_path is not None:
        savefig_academic(fig, output_path)
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
