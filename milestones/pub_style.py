"""
Publication-quality matplotlib style for IEEE two-column papers (~3.5 in column).

Usage (in any notebook):
    import pub_style          # applies rcParams on import
    pub_style.plot_metric(…)  # upgraded loss-curve helper
"""

import matplotlib.pyplot as plt
import pandas as pd


# ─── IEEE-ready rcParams ────────────────────────────────────────────
def apply():
    """Apply publication-quality rcParams globally.

    Font sizes are calibrated so that axis labels and legends remain
    legible when a figure is shrunk to ~3.5 inches (one IEEE column).
    """
    plt.rcParams.update({
        # fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        # figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        # ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })


# apply on import so notebooks just need `import pub_style`
apply()


# ─── Upgraded plot_metric helper ────────────────────────────────────
def plot_metric(entity, project, run_ids, key, ax=None,
                label=None, color=None, title=None, ylabel=None,
                smooth_window=0, grid=True, *, api=None):
    """Plot a W&B metric with publication-quality styling.

    Parameters
    ----------
    api : wandb.Api, optional
        Pre-constructed API handle.  If *None*, one is created.
    """
    if api is None:
        import wandb
        api = wandb.Api()

    series = _fetch_metric_multi(api, entity, project, run_ids, key)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    plot_label = label or key.split("/")[-1]

    if smooth_window > 0:
        smoothed = series.rolling(smooth_window, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, label=plot_label,
                color=color, linewidth=2.2, zorder=3)
        ax.plot(series.index, series.values, alpha=0.12,
                color=color, linewidth=0.6, zorder=2)
    else:
        ax.plot(series.index, series.values, label=plot_label,
                color=color, linewidth=1.8, zorder=3)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(ylabel or key.split("/")[-1])
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    if grid:
        ax.grid(True, which='major', alpha=0.25, linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.10, linewidth=0.3)
        ax.minorticks_on()
    # remove top & right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, framealpha=0.85, facecolor='white',
              edgecolor='#CCCCCC', loc='upper right')
    return ax


# ─── internal helpers ───────────────────────────────────────────────
def _fetch_metric_multi(api, entity, project, run_ids, key):
    if isinstance(run_ids, str):
        run_ids = [run_ids]

    parts = []
    offset = 0
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        h = run.history(keys=[key], pandas=True)
        series = h[key].dropna().reset_index(drop=True)
        series.index = series.index + offset
        parts.append(series)
        offset += len(series)

    return pd.concat(parts)
