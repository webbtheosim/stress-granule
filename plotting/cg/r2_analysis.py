"""
R^2-vs-feature-count line plot for the CG small-molecule classifier (Fig S1A).

Support module for the CG parameterization figures. Reads a precomputed metrics
table and plots the coefficient of determination R^2 of the parameter
regression as a function of the number of input features ``x``, with one
line+marker series per Wang-Frenkel parameter (epsilon, sigma, mu, r_c) drawn
in the rocket palette. The plot does not fit anything: it only renders the
R^2 values supplied in the CSV.

Input: ``R2_Dataset.csv`` with columns ``x`` (feature count), ``R^2``, and
``Parameter`` (LaTeX parameter label, e.g. ``$\\epsilon$``).
Output: ``{output_dir}/R2_Plot.png`` (400 dpi).

Entry point: ``run_r2_analysis()`` (also runnable as a script via ``main()``
with ``--metrics``/``--out``); called from cg_pipeline/run_full_pipeline.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["run_r2_analysis"]


def _configure_style() -> None:
    """Set the shared Matplotlib/seaborn rc parameters for the figure."""
    sns.set_theme(style="ticks")
    sns.set_style('white')
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.rc('font', size=10)
    plt.rc('axes', linewidth=2)


def run_r2_analysis(metrics_csv: Path | str = 'R2_Dataset.csv', output_dir: Path | str = '.',
                    figsize=None, ax_rect=None) -> None:
    """Plot R^2 vs feature count per parameter and write ``R2_Plot.png``.

    Args:
        metrics_csv: CSV with columns ``x``, ``R^2``, and ``Parameter``.
        output_dir: directory for the output PNG (created if needed).
        figsize: figure size in inches (used with ``ax_rect``); defaults to a
            wide panel.
        ax_rect: explicit axes rectangle [l, b, w, h] in figure fractions, to
            match other square panels; if None, a tight-layout panel is used.
    """
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    _configure_style()
    if ax_rect is not None:
        # explicit axes geometry (e.g. to match other square panels)
        fig = plt.figure(figsize=figsize or (3.2, 2.6))
        axs = fig.add_axes(ax_rect)
    else:
        fig, axs = plt.subplots(figsize=(3.2, 2.6), tight_layout=True)
    axs.set_ylim(0, 1)
    axs.set_xlim(0, 16)
    axs.set_xticks(range(0, 17, 2))

    sns.lineplot(ax=axs, data=df, x="x", y="R^2", linewidth=2, hue="Parameter", legend=False, palette="rocket", zorder=1)
    sns.scatterplot(
        ax=axs,
        data=df,
        x="x",
        y="R^2",
        hue="Parameter",
        palette="rocket",
        legend=False,
        s=20,
        edgecolor="k",
        linewidth=1,
        zorder=2,
        style='Parameter',
        markers={'$\epsilon$': 'o', '$\sigma$': 's', '$\mu$': '^', '$r_{c}$': 'd'}
    )
    axs.tick_params(direction='in', length=4, width=2,
                    top=True, bottom=True, left=True, right=True,
                    labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    axs.set_xlabel(None)
    axs.set_ylabel(None)

    unique_params = df["Parameter"].unique()
    colors = sns.color_palette("rocket", n_colors=len(unique_params))
    markers = {'$\epsilon$': 'o', '$\sigma$': 's', '$\mu$': '^', '$r_{c}$': 'd'}

    handles = []
    labels = []
    for param, color in zip(unique_params, colors):
        line_handle = plt.Line2D([], [], color=color, linewidth=2)
        marker_style = markers.get(param, 'o')
        marker_handle = plt.Line2D([], [], marker=marker_style, markerfacecolor=color, markeredgecolor='k', linewidth=0)
        handles.append((line_handle, marker_handle))
        labels.append(param)

    axs.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.94, 0.07), ncol=1, frameon=False)
    fig.savefig(output_dir / "R2_Plot.png", format="png", dpi=400)
    plt.close(fig)


def main() -> None:
    """Parse ``--metrics``/``--out`` and run the R^2 analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='R2_Dataset.csv')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()
    run_r2_analysis(args.metrics, args.out)


if __name__ == '__main__':
    main()
