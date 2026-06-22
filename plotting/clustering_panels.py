#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render Figure 1 panels C and D: clustering-metric bar plots.

Renderer for the clustering panels of paper Figure 1. Reads the tidy per-frame
clustering tables produced by the compute module
``analysis/cluster_analysis.py`` (imported via the standard plotting->analysis
shim) and draws the phi bar plots:

  - Panel C: phi (fraction of biopolymers in the largest cluster) vs protein mass
    fraction w_p over the 0.0-1.0 RNA-content titration
    (``percent_clustering.csv``). Bars carry the across-frame standard
    error.
  - Panel D: per-species phi at w_p = 0.5 (protein + RNA) compared against
    w_p = 1.0 (pure protein) (``type_clustering.csv``), grouped per
    biopolymer, with across-frame standard error.

This module contains only rendering: the dump parsing and per-frame reduction
live in ``analysis/cluster_analysis.py``. If the per-frame CSVs do not yet
exist they are computed on the fly from ``--data-root``.

The companion N_D and R_g bars (extra columns ``Cluster`` and ``RG`` in the same
CSVs) are also rendered as supporting panels.

Inputs:
  - ``--percent-csv`` : ``percent_clustering.csv`` (panel C)
  - ``--type-csv``    : ``type_clustering.csv``    (panel D)
  Either may be omitted/missing, in which case it is built from ``--data-root``
  (default: ``simulation_inputs/model_systems``; point it at any per-system
  CLUSTER results dump to render from the original results dump).

Outputs (PNG, dpi 400, into ``--out``):
  - panel C : ``Percent_Phi_Bar.png`` (+ ``Percent_ND_Bar.png``, ``Percent_RG_Bar.png``)
  - panel D : ``Type_Phi_Bar.png``    (+ ``Type_ND_Bar.png``,    ``Type_RG_Bar.png``)

CLI:
    python clustering_panels.py [--percent-csv CSV] [--type-csv CSV]
                                [--data-root DIR] [--out DIR]

Adapted (cleaned, de-interactive, processing/plotting split) from the original
``the original ClusteringAnalysis.py``.
"""

import os
import sys
import argparse

# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute module that lives in analysis/ (cluster_analysis).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cluster_analysis as clust


def _apply_style():
    """Apply the shared house style for the Figure-1 clustering panels."""
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_style("white")
    plt.rc("axes", titlesize=10)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("font", size=10)
    plt.rc("axes", linewidth=2)


def _load_or_compute(csv_path, compute_fn, data_root, columns):
    """Return a per-frame DataFrame, computing it from raw dumps if missing.

    Args:
        csv_path: Tidy CSV path to read if present.
        compute_fn: ``analysis/cluster_analysis`` function to call otherwise.
        data_root: Per-system data root passed to ``compute_fn``.
        columns: Expected column list (for the empty-fallback message).

    Returns:
        pandas.DataFrame of per-frame clustering rows.
    """
    if csv_path and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    print("[cluster_analysis.plot] {} not found; computing from {}".format(
        csv_path, data_root))
    df = compute_fn(data_root)
    if df.empty:
        print("[cluster_analysis.plot] WARNING: no data for columns {}".format(columns))
    return df


def _bar(df, x, y, out_path, ylim, yticks=None, palette="Blues",
         hue=None, hue_order=None, figsize=(3.42, 2.3), rotate_x=False):
    """Draw one across-frame mean bar plot with standard-error bars.

    seaborn aggregates the per-frame rows to the mean with ``errorbar='se'``,
    matching the original panels (Blues palette, inward ticks, black caps).

    Args:
        df: Per-frame DataFrame.
        x: Column for the categorical x-axis.
        y: Column to aggregate on the y-axis ("Phi", "Cluster", or "RG").
        out_path: PNG path to write.
        ylim: (lo, hi) y-limits.
        yticks: Optional explicit y-tick array.
        palette: seaborn palette name.
        hue: Optional hue column (panel D groups by state).
        hue_order: Optional hue order.
        figsize: Figure size in inches.
        rotate_x: Rotate x tick labels 45 deg (panel D species names).

    Returns:
        Written PNG path.
    """
    fig, axs = plt.subplots(figsize=figsize, tight_layout=True)
    g1 = sns.barplot(ax=axs, data=df, x=x, y=y, hue=(hue if hue else x),
                     palette=palette, dodge=bool(hue), saturation=100, width=0.5,
                     errorbar="se", capsize=0.05, err_kws={"color": "k", "linewidth": 1},
                     edgecolor="k", hue_order=hue_order)
    if axs.get_legend() is not None:
        axs.get_legend().remove()
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    axs.set_ylim(*ylim)
    if yticks is not None:
        axs.set_yticks(yticks)
    axs.tick_params(left=True, right=True, top=True, bottom=True,
                    labelbottom=True, direction="in", length=4, width=2)
    if rotate_x:
        plt.xticks(rotation=45)
    plt.savefig(out_path, format="png", dpi=400)
    plt.close(fig)
    print("[cluster_analysis.plot] wrote {}".format(out_path))
    return out_path


def plot_percent_panels(df, out_dir):
    """Render panel C (phi vs w_p) and its N_D / R_g companions.

    Args:
        df: ``percent_clustering.csv`` contents.
        out_dir: Output directory for the PNGs.

    Returns:
        Path to the phi panel PNG (panel C), or None if no data.
    """
    if df.empty:
        return None
    # Order bars by ascending protein mass fraction.
    df = df.sort_values("w_p")
    df["w_p"] = df["w_p"].map(lambda v: "{:.1f}".format(float(v)))

    phi_png = _bar(df, x="w_p", y="Phi",
                   out_path=os.path.join(out_dir, "Percent_Phi_Bar.png"),
                   ylim=(0, 1), yticks=np.arange(0, 1.05, 0.2), palette="Blues")
    _bar(df, x="w_p", y="Cluster",
         out_path=os.path.join(out_dir, "Percent_ND_Bar.png"),
         ylim=(0, 80), palette="Blues", figsize=(4.8, 3.2))
    _bar(df, x="w_p", y="RG",
         out_path=os.path.join(out_dir, "Percent_RG_Bar.png"),
         ylim=(0, 8000), palette="Blues", figsize=(4.8, 3.2))
    return phi_png


def plot_type_panels(df, out_dir):
    """Render panel D (per-species phi at w_p=0.5 vs 1.0) and companions.

    Args:
        df: ``type_clustering.csv`` contents.
        out_dir: Output directory for the PNGs.

    Returns:
        Path to the phi panel PNG (panel D), or None if no data.
    """
    if df.empty:
        return None
    df = df.sort_values(["protein", "w_p"]).copy()
    # Legend labels matching the manuscript (w_p annotations).
    df["state_label"] = df["w_p"].map(
        lambda v: r"$\phi_{{P}}={:.1f}$".format(float(v)))
    hue_order = [r"$\phi_{P}=0.5$", r"$\phi_{P}=1.0$"]
    # Two Blues shades (mid + dark) for the two states, as in the original.
    cols = sns.color_palette(palette="Blues", n_colors=11)
    col_pal = [cols[5], cols[10]]

    phi_png = _bar(df, x="protein", y="Phi", hue="state_label", hue_order=hue_order,
                   out_path=os.path.join(out_dir, "Type_Phi_Bar.png"),
                   ylim=(0, 1), yticks=np.arange(0, 1.1, 0.2), palette=col_pal,
                   figsize=(3.42, 2.58), rotate_x=True)
    _bar(df, x="protein", y="Cluster", hue="state_label", hue_order=hue_order,
         out_path=os.path.join(out_dir, "Type_ND_Bar.png"),
         ylim=(0, 120), palette=col_pal, figsize=(4.8, 3.2), rotate_x=True)
    _bar(df, x="protein", y="RG", hue="state_label", hue_order=hue_order,
         out_path=os.path.join(out_dir, "Type_RG_Bar.png"),
         ylim=(0, 10000), palette=col_pal, figsize=(4.8, 3.2), rotate_x=True)
    return phi_png


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render Figure 1 panels C (phi vs w_p) and D (per-species "
                    "phi at w_p=0.5 vs 1.0).")
    parser.add_argument("--percent-csv", default=None,
                        help="percent_clustering.csv (panel C). If "
                             "omitted/missing it is computed from --data-root.")
    parser.add_argument("--type-csv", default=None,
                        help="type_clustering.csv (panel D). If "
                             "omitted/missing it is computed from --data-root.")
    parser.add_argument("--data-root", default=clust.DEFAULT_DATA_ROOT,
                        help="Per-system data root used when CSVs must be "
                             "computed (default: simulation_inputs/model_systems; "
                             "point it at any per-system CLUSTER results dump).")
    parser.add_argument("--out", default="FIGURES",
                        help="Output directory for the panel PNGs "
                             "(default: FIGURES).")
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)
    _apply_style()

    percent_df = _load_or_compute(
        args.percent_csv, clust.compute_percent, args.data_root,
        columns=["percent", "w_p", "Time", "Cluster", "RG", "Phi"])
    plot_percent_panels(percent_df, args.out)

    type_df = _load_or_compute(
        args.type_csv, clust.compute_type, args.data_root,
        columns=["protein", "state", "w_p", "Time", "Cluster", "RG", "Phi"])
    plot_type_panels(type_df, args.out)


if __name__ == "__main__":
    main()
