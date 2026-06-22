#!/usr/bin/env python3
"""Per-system publication plots for a single pure-SM solubility control run.

Reads the CSV outputs of one ``sm_analysis.py`` run (cluster size series, SM
number- and mass-density radial profiles, and SM--SM COM RDF) for a single
named system and renders the matching figures. This is the single-system
renderer; the cohort-level aggregation lives in ``sm_plotting.py``.

Inputs (in ``analysis_dir``, for the given ``system`` label):
    Cluster_SM_<system>.csv, Cluster_SM_SUMMARY_<system>.csv (optional),
    Density_Profile_SM_Number_<system>.csv,
    Density_Profile_SM_Mass_<system>.csv, RDF_SM_COM_<system>.csv

Outputs (in ``<analysis_dir>/IMAGES/``):
    Cluster_SM_<system>.png, Density_SM_<system>.png, RDF_SM_<system>.png

CLI:
    python sm_system_analysis.py <analysis_dir> <system>
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _configure_style() -> None:
    """Apply the shared seaborn ticks/white house style and font sizes."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("axes", titlesize=12)
    plt.rc("axes", labelsize=11)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=9)
    plt.rc("font", size=10)


def _time_axis(cluster_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Return the x-axis values and label, preferring ns (from "Time (ps)").

    Falls back to the integer "Frame" column if no finite "Time (ps)" is present.
    """
    if "Time (ps)" in cluster_df.columns:
        times_ps = cluster_df["Time (ps)"].to_numpy(dtype=float)
        if np.isfinite(times_ps).any():
            times_ns = times_ps / 1000.0
            return times_ns, "Time (ns)"
    frames = cluster_df["Frame"].to_numpy(dtype=float)
    return frames, "Frame"


def plot_cluster(cluster_path: str, summary_path: str, outdir: str, system: str) -> None:
    """Plot the largest-cluster fraction vs time for one system, with a 0.1 threshold.

    Writes ``Cluster_SM_<system>.png`` to ``outdir`` and, if ``summary_path``
    exists and is non-empty, prints the summary table to stdout.
    """
    cluster_df = pd.read_csv(cluster_path)
    time_axis, xlabel = _time_axis(cluster_df)

    palette = sns.color_palette("rocket", n_colors=3)
    line_color = palette[1]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    sns.lineplot(
        ax=ax,
        x=time_axis,
        y=cluster_df["LargestClusterFraction"],
        color=line_color,
        linewidth=2.5,
        zorder=2,
    )
    sns.scatterplot(
        ax=ax,
        x=time_axis,
        y=cluster_df["LargestClusterFraction"],
        color=line_color,
        s=30,
        edgecolor="k",
        linewidth=0.8,
        zorder=3,
    )

    ax.axhline(0.1, color="grey", linestyle="--", linewidth=1.5, label="0.1 threshold", zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Largest cluster fraction")
    ax.set_ylim(0, max(0.12, 1.05 * cluster_df["LargestClusterFraction"].max()))
    ax.tick_params(direction="in", top=True, right=True, length=4, width=1.5)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Cluster size – {system}")

    file_path = os.path.join(outdir, f"Cluster_SM_{system}.png")
    fig.tight_layout()
    fig.savefig(file_path, dpi=400)
    plt.close(fig)

    if os.path.isfile(summary_path):
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))


def plot_rdp(number_path: str, mass_path: str, outdir: str, system: str) -> None:
    """Plot side-by-side SM mass- and number-density radial profiles for one system.

    Writes ``Density_SM_<system>.png`` (two panels, with SEM bands) to ``outdir``.
    """
    df_number = pd.read_csv(number_path)
    df_mass = pd.read_csv(mass_path)

    blues = sns.color_palette("Blues", n_colors=9)
    rocket = sns.color_palette("rocket", n_colors=3)

    fig, (ax_mass, ax_number) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)

    # Mass density
    x = df_mass["Distance from center of mass (A)"]
    y = df_mass["SM mass density (mg/mL)"]
    err = df_mass["Standard mean error"]
    ax_mass.plot(x, y, color=blues[5], linewidth=2.5, label="Mass density")
    ax_mass.fill_between(x, y - err, y + err, color=blues[5], alpha=0.25)
    ax_mass.scatter(x, y, color=blues[6], s=20, edgecolor="k", linewidth=0.6)
    ax_mass.set_ylabel("SM mass density (mg/mL)")
    ax_mass.tick_params(direction="in", top=True, right=True, length=4, width=1.5)
    ax_mass.set_xlim(0, x.max())

    # Number density
    x_num = df_number["Distance from center of mass (A)"]
    y_num = df_number["SM number density (1/A^3)"]
    err_num = df_number["Standard mean error"]
    ax_number.plot(x_num, y_num, color=rocket[1], linewidth=2.5, label="Number density")
    ax_number.fill_between(x_num, y_num - err_num, y_num + err_num, color=rocket[1], alpha=0.25)
    ax_number.scatter(x_num, y_num, color=rocket[2], s=20, edgecolor="k", linewidth=0.6)
    ax_number.set_ylabel("SM number density (1/Å³)")
    ax_number.tick_params(direction="in", top=True, right=True, length=4, width=1.5)
    ax_number.set_xlim(0, x_num.max())

    ax_mass.set_xlabel("Radius from box centre (Å)")
    ax_number.set_xlabel("Radius from box centre (Å)")
    fig.suptitle(f"Radial density profiles – {system}")

    fig.tight_layout()
    file_path = os.path.join(outdir, f"Density_SM_{system}.png")
    fig.savefig(file_path, dpi=400)
    plt.close(fig)


def plot_rdf(rdf_path: str, outdir: str, system: str) -> None:
    """Plot the SM--SM COM radial distribution function g(r) for one system.

    Writes ``RDF_SM_<system>.png`` (with the g(r)=1 ideal-gas reference) to ``outdir``.
    """
    rdf_df = pd.read_csv(rdf_path)
    palette = sns.color_palette("crest", n_colors=6)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = rdf_df["Distance (A)"]
    y = rdf_df["g(r)"]
    sns.lineplot(ax=ax, x=x, y=y, color=palette[4], linewidth=2.5)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.3, zorder=1)
    ax.set_xlabel("Pair distance (Å)")
    ax.set_ylabel("g(r)")
    ax.tick_params(direction="in", top=True, right=True, length=4, width=1.5)
    ax.set_ylim(0, max(1.2, y.max() * 1.05))
    ax.set_xlim(0, x.max())
    ax.set_title(f"COM RDF – {system}")

    fig.tight_layout()
    file_path = os.path.join(outdir, f"RDF_SM_{system}.png")
    fig.savefig(file_path, dpi=400)
    plt.close(fig)


def main() -> None:
    """Parse CLI arguments, verify the required CSVs, and render the three figures."""
    parser = argparse.ArgumentParser(
        description="Generate plots from SM_ANALYSIS outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("analysis_dir", help="Directory produced by sm_analysis.py")
    parser.add_argument("system", help="System label (must match CSV filenames)")
    args = parser.parse_args()

    analysis_dir = os.path.abspath(args.analysis_dir)
    images_dir = os.path.join(analysis_dir, "IMAGES")
    os.makedirs(images_dir, exist_ok=True)

    _configure_style()

    cluster_csv = os.path.join(analysis_dir, f"Cluster_SM_{args.system}.csv")
    cluster_summary_csv = os.path.join(
        analysis_dir, f"Cluster_SM_SUMMARY_{args.system}.csv"
    )
    rdp_number_csv = os.path.join(analysis_dir, f"Density_Profile_SM_Number_{args.system}.csv")
    rdp_mass_csv = os.path.join(analysis_dir, f"Density_Profile_SM_Mass_{args.system}.csv")
    rdf_csv = os.path.join(analysis_dir, f"RDF_SM_COM_{args.system}.csv")

    missing = [
        path
        for path in (cluster_csv, rdp_number_csv, rdp_mass_csv, rdf_csv)
        if not os.path.isfile(path)
    ]
    if missing:
        missing_str = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing required CSVs. Run sm_analysis.py first. Missing files:\n  - "
            f"{missing_str}"
        )

    plot_cluster(cluster_csv, cluster_summary_csv, images_dir, args.system)
    plot_rdp(rdp_number_csv, rdp_mass_csv, images_dir, args.system)
    plot_rdf(rdf_csv, images_dir, args.system)

    print(f"[OK] Figures written to {images_dir}")


if __name__ == "__main__":
    main()

