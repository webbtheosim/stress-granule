#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render Figure 1 panels A and B: sequence-composition histograms.

Renderer for the composition panels of paper Figure 1. Reads the tidy
composition table produced by the compute module ``analysis/sequence_composition.py``
(imported via the standard plotting->analysis shim) and draws:

  - Panel A: one amino-acid composition histogram per protein (TTP, TDP43,
    G3BP1, PABP1, TIA1, FUS). Bars are colored by physicochemical class
    (Electrostatic / Polar / Hydrophobic / Aromatic) and ordered along the fixed
    manuscript x-axis  R H K D E  S T N Q  C G P A V I L M  F Y W.
  - Panel B: the RNA nucleotide-composition bars (A U C G).

This module contains only rendering (no parsing of raw sequences): the heavy
lifting lives in ``analysis/sequence_composition.py``. If the composition CSV does
not yet exist, it is computed on the fly from the shipped sequence data.

Inputs:
  - ``--csv`` : ``sequence_composition.csv`` (tidy table). If absent it is built
    from ``--seq-dir`` (default: the shipped ``analysis/sequences/``).

Outputs (PNG, dpi 400, into ``--out``):
  - ``<out>/<PROTEIN>_Histogram.png`` for each protein  (panel A)
  - ``<out>/RNA_Histogram.png``                         (panel B)

CLI:
    python composition_panels.py [--csv CSV] [--seq-dir DIR] [--out DIR]

Adapted (cleaned, de-interactive, processing/plotting split) from the original
``the original SequenceAnalysis.py``.

Created on Wed May 10 19:27:50 2023
@author: jaykaplan
"""

import os
import sys
import argparse

# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute module that lives in analysis/ (sequence_composition).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sequence_composition as comp


def _apply_style():
    """Apply the shared house style for the Figure-1 composition panels."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("axes", titlesize=10)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("font", size=10)
    plt.rc("axes", linewidth=2)


def _load_composition(csv_path, seq_dir):
    """Return the composition DataFrame, computing the CSV if it is missing.

    Args:
        csv_path: Path to ``sequence_composition.csv`` (read if present).
        seq_dir: Sequence-input directory used to build the CSV if absent.

    Returns:
        pandas.DataFrame in the tidy long form emitted by the compute module.
    """
    if csv_path and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    print("[sequence_composition.plot] {} not found; computing from {}".format(
        csv_path, seq_dir))
    return comp.compute_composition(seq_dir=seq_dir)


def plot_protein_histograms(df, out_dir):
    """Draw the per-protein amino-acid composition histograms (panel A).

    One figure per protein; bars follow the fixed ``AA_ORDER`` and are colored
    by physicochemical class using the four evenly-spaced ``hls`` hues of the
    original figure.

    Args:
        df: Tidy composition DataFrame (protein rows used).
        out_dir: Directory for ``<PROTEIN>_Histogram.png`` files.

    Returns:
        List of written PNG paths.
    """
    written = []
    # Four class colors taken from an 8-color hls wheel (matches the original).
    cols = sns.color_palette(palette="hls", n_colors=8)
    col_pal = [cols[0], cols[2], cols[4], cols[6]]

    proteins = df.loc[df["kind"] == "protein", "molecule"].unique()
    for protein in proteins:
        sub = df.loc[df["molecule"] == protein].copy()
        # Enforce the manuscript residue order on the x-axis.
        sub["residue"] = pd.Categorical(sub["residue"], categories=comp.AA_ORDER,
                                        ordered=True)
        sub = sub.sort_values("residue")

        fig, axs = plt.subplots(figsize=(3.3, 1), tight_layout=True)
        g1 = sns.barplot(ax=axs, data=sub, x="residue", y="count",
                         hue="residue_class", palette=col_pal, saturation=100,
                         width=0.6, dodge=False, edgecolor="k",
                         hue_order=["Electrostatic", "Polar", "Hydrophobic", "Aromatic"])
        if axs.get_legend() is not None:
            axs.get_legend().remove()
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        axs.set_ylim(0, 80)
        plt.yticks(np.arange(0, 81, 20))
        axs.tick_params(left=True, right=True, top=True, bottom=False,
                        labelbottom=True, direction="in", length=4, width=2)

        png_path = os.path.join(out_dir, "{}_Histogram.png".format(protein))
        plt.savefig(png_path, format="png", dpi=400)
        plt.close(fig)
        written.append(png_path)
        print("[sequence_composition.plot] wrote {}".format(png_path))
    return written


def plot_rna_histogram(df, out_dir):
    """Draw the RNA nucleotide-composition bars (panel B).

    Args:
        df: Tidy composition DataFrame (rna rows used).
        out_dir: Directory for ``RNA_Histogram.png``.

    Returns:
        Written PNG path, or None if there are no RNA rows.
    """
    sub = df.loc[df["kind"] == "rna"].copy()
    if sub.empty:
        print("[sequence_composition.plot] no RNA rows; skipping panel B")
        return None
    sub["residue"] = pd.Categorical(sub["residue"], categories=comp.NA_ORDER,
                                    ordered=True)
    sub = sub.sort_values("residue")

    fig, axs = plt.subplots(figsize=(6.72, 1.2), tight_layout=True)
    cols = sns.color_palette(palette="Set2", n_colors=8)
    g1 = sns.barplot(ax=axs, data=sub, x="residue", y="count", color=cols[7],
                     saturation=100, width=0.8, dodge=False, edgecolor="k")
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    axs.set_ylim(0, 400)
    plt.yticks(np.arange(0, 401, 100))
    axs.tick_params(left=True, right=True, top=True, bottom=False,
                    labelbottom=True, direction="in", length=4, width=2)

    png_path = os.path.join(out_dir, "RNA_Histogram.png")
    plt.savefig(png_path, format="png", dpi=400)
    plt.close(fig)
    print("[sequence_composition.plot] wrote {}".format(png_path))
    return png_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render Figure 1 panels A (per-protein amino-acid "
                    "histograms) and B (RNA nucleotide bars).")
    parser.add_argument("--csv", default=None,
                        help="sequence_composition.csv to render. If omitted or "
                             "missing, it is computed from --seq-dir.")
    parser.add_argument("--seq-dir", default=comp.DEFAULT_SEQ_DIR,
                        help="Sequence-input directory used when the CSV must be "
                             "computed (default: shipped analysis/sequences/).")
    parser.add_argument("--out", default="FIGURES",
                        help="Output directory for the panel PNGs "
                             "(default: FIGURES).")
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)
    _apply_style()
    df = _load_composition(args.csv, args.seq_dir)
    plot_protein_histograms(df, args.out)
    plot_rna_histogram(df, args.out)


if __name__ == "__main__":
    main()
