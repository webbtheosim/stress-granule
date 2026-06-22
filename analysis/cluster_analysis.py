#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster-metric extraction for paper Figure 1, panels C and D (compute only).

Compute step for the Figure-1 clustering panels. Reads the per-system LAMMPS
``compute cluster/atom`` radius-of-gyration dumps (``CLUSTER/llps_rg_*.out``) and
reduces each saved frame to three scalars:

  - ``Cluster`` (N_D): number of resolved clusters (entries with Rg > 0),
  - ``RG``           : radius of gyration of the largest cluster in the frame,
  - ``Phi`` (phi)    : fraction of biopolymers in the largest cluster, i.e.
                       (atoms in largest cluster) / (atoms in the FIRST frame).

This module is render-free (no matplotlib, no figure output): it walks the dumps,
applies the equilibration/time window, and writes one tidy CSV per system that
the companion renderer ``plotting/clustering_panels.py`` turns into the figure.

Three systems feed two panels:
  - panel C : ``protein_rna_titration`` -- phi vs protein mass fraction w_p, a
              0-100 % RNA-content titration (system name encodes the percentage).
  - panel D : ``single_protein_rna``  (w_p = 0.5, protein + RNA) compared against
              ``single_protein_pure`` (w_p = 1.0, pure protein), per biopolymer.

Dump format (LAMMPS time-averaged ``fix ave/time`` vector output)::

    # Time-averaged data for fix fxrg
    # TimeStep Number-of-rows
    # Row c_radgyr c_prop
    <TimeStep> <Nclusters>
    1 <Rg_1> <Natoms_1>
    2 <Rg_2> <Natoms_2>
    ...

Frames are delimited by header lines whose first field is a multiple of
``--step-size`` (default 200000 steps). The per-frame reduction reproduces the
original ``ClusteringAnalysis.py`` accumulation exactly (including
normalizing Phi by the first frame's atom total).

Data root (``--data-root``): a directory holding the per-system subfolders, each
with a ``CLUSTER/`` directory of ``llps_rg_*.out`` files. Defaults to the
in-repo ``simulation_inputs/model_systems`` location; for verification against
the original results dump pass ``--data-root <dump>``.

Inputs:
  - ``<data-root>/protein_rna_titration/CLUSTER/llps_rg_<pct>.out``
  - ``<data-root>/single_protein_rna/CLUSTER/llps_rg_<PROT>.out``
  - ``<data-root>/single_protein_pure/CLUSTER/llps_rg_<PROT>.out``

Outputs (in ``--out``):
  - ``percent_clustering.csv`` : columns
        percent, w_p, Time, Cluster, RG, Phi              (panel C)
  - ``type_clustering.csv``    : columns
        protein, state, w_p, Time, Cluster, RG, Phi       (panel D)
    where ``state`` is "RNA" (w_p=0.5) or "Pure" (w_p=1.0).

CLI:
    python cluster_analysis.py [--data-root DIR] [--out DIR]
                                 [--start NS] [--end NS] [--step-size STEPS]

Adapted (cleaned, de-interactive, processing/plotting split) from the original
``ClusteringAnalysis.py`` (``class Analyze`` -> ``cluster_analysis``).
"""

import os
import argparse

import pandas as pd


# Default per-system data root inside the repo (each system has a CLUSTER/ dir).
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_inputs", "model_systems")

# System subfolder names, matching the in-repo model_systems/ layout (each holds
# a per-system CLUSTER/ results dump).
PERCENT_SYSTEM = "protein_rna_titration"
TYPE_RNA_SYSTEM = "single_protein_rna"
TYPE_PURE_SYSTEM = "single_protein_pure"

# Window / sampling defaults (ns), reproducing the original script's constants.
DEFAULT_STEP_SIZE = 200000   # LAMMPS steps between saved cluster frames
DEFAULT_START_NS = 4.0       # discard frames before this time (equilibration)
DEFAULT_END_NS = 120.0       # discard frames after this time


class cluster_analysis:
    """Parse one LAMMPS cluster/Rg dump and reduce each frame to scalars.

    Renamed from the original ``class Analyze``. Holds a single dump file and
    walks it frame-by-frame, appending one tidy row per frame (per-cluster count,
    largest-cluster Rg, and largest-cluster occupancy fraction Phi).
    """

    def __init__(self, seq_file):
        """Store the dump path and reset the first-frame cluster counter.

        Args:
            seq_file: Path to a ``CLUSTER/llps_rg_*.out`` dump.
        """
        self.seq_file = seq_file
        self.initial_cluster = 1

    def read_file(self):
        """Read the dump and drop the 3 header comment lines.

        Returns:
            List of data lines (frame headers + per-cluster rows).
        """
        with open(self.seq_file, "r") as handle:
            contents = handle.readlines()
        return contents[3:]

    def accumulate_windows(self, name, species, effect, step_size):
        """Reduce every saved frame in this dump to one tidy row.

        Faithful re-implementation of the original ``generate_data``: frames are
        delimited by header lines whose first field is a multiple of
        ``step_size``; within a frame each row contributes its Rg and atom count.
        Phi is normalized by the atom total of the FIRST frame (``total_atoms``),
        exactly as in the original. Time advances by ``step_size * 200 / 1e7`` ns
        per frame (= 4.0 ns for the default 200000-step cadence).

        Args:
            name: System-specific label (percentage fraction, or protein name).
            species: One of "protein_rna_titration" (adds percent columns) or any
                other value (adds a "Protein"/"Compound Class" row layout).
            effect: Secondary label stored under "Compound Class" (e.g. the
                w_p annotation for the type-protein systems).
            step_size: LAMMPS steps between frames (frame-delimiter modulus).

        Returns:
            DataFrame with one row per saved frame for this dump.
        """
        lines = self.read_file()
        file_length = len(lines)
        line_num = 0
        biggest = 0
        time_step = 0
        cluster_num = 0
        largest_num = 0
        run_once = True
        total_atoms = 0
        rows = []  # collect per-frame dicts; concat once at the end

        while line_num < file_length:
            atom_sum = 0
            if line_num == file_length:
                break
            if time_step == 0:
                line_num += 1
            while int(lines[line_num].split()[0]) % step_size != 0:
                if line_num == file_length - 1:
                    break
                line = lines[line_num].split()
                rg = float(line[1])

                if rg > 0:
                    cluster_num += 1
                if rg > biggest:
                    biggest = rg

                num_atoms = int(line[2])
                if num_atoms > largest_num:
                    largest_num = num_atoms
                atom_sum += num_atoms
                line_num += 1

            if run_once:
                self.initial_cluster = cluster_num
                total_atoms = atom_sum
                run_once = False

            line_num += 1
            time_step += step_size * 200 / 10000000
            if species == PERCENT_SYSTEM:
                label = r"$\phi_P={}$".format(name)
                row = {"Percent": str(name), "Phi_Label": label, "Time": time_step,
                       "Cluster": cluster_num, "RG": biggest,
                       "Phi": largest_num / total_atoms}
            else:
                row = {"Protein": str(name), "Time": time_step,
                       "Cluster": cluster_num, "RG": biggest,
                       "Phi": largest_num / total_atoms, "Compound Class": effect}
            rows.append(row)

            biggest = 0
            largest_num = 0
            cluster_num = 0

        return pd.DataFrame(rows)


def _cluster_dir(data_root, system):
    """Return ``<data_root>/<system>/CLUSTER`` (the dump directory)."""
    return os.path.join(data_root, system, "CLUSTER")


def _iter_llps_files(cluster_dir):
    """Yield ``(filename, full_path)`` for ``llps_*`` dumps in a directory."""
    if not os.path.isdir(cluster_dir):
        print("[cluster_analysis] WARNING: missing dir {}".format(cluster_dir))
        return
    for entry in sorted(os.scandir(cluster_dir), key=lambda e: e.name):
        if "llps" in entry.name:
            yield entry.name, entry.path


def compute_percent(data_root, start=DEFAULT_START_NS, end=DEFAULT_END_NS,
                    step_size=DEFAULT_STEP_SIZE):
    """Build the tidy per-frame table for the percent-RNA titration (panel C).

    The system name encodes the RNA percentage in ``llps_rg_<pct>.out``; the
    protein mass fraction is ``w_p = 1 - pct/100``. (The original stored the raw
    fraction ``pct/100`` under "Percent"; both are emitted here.)

    Args:
        data_root: Directory holding the ``protein_rna_titration`` system.
        start: Discard frames with Time < start (ns).
        end: Discard frames with Time > end (ns).
        step_size: Frame-delimiter modulus (LAMMPS steps).

    Returns:
        DataFrame with columns percent, w_p, Time, Cluster, RG, Phi.
    """
    frames = []
    cluster_dir = _cluster_dir(data_root, PERCENT_SYSTEM)
    for file_name, file_path in _iter_llps_files(cluster_dir):
        parts = file_name.split(".")[0].split("_")
        percent_fraction = int(parts[2]) / 100  # raw fraction (legacy "Percent")
        analyzer = cluster_analysis(file_path)
        frames.append(analyzer.accumulate_windows(
            name=percent_fraction, species=PERCENT_SYSTEM, effect="",
            step_size=step_size))

    if not frames:
        return pd.DataFrame(columns=["percent", "w_p", "Time", "Cluster", "RG", "Phi"])
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[(df["Time"] >= start) & (df["Time"] <= end)].copy()
    df = df.sort_values(by=["Percent"]).reset_index(drop=True)
    # Tidy renames + protein mass fraction (panel C x-axis is w_p, not RNA frac).
    df = df.rename(columns={"Percent": "percent"})
    df["percent"] = df["percent"].astype(float)
    df["w_p"] = (1.0 - df["percent"]).round(2)
    return df[["percent", "w_p", "Time", "Cluster", "RG", "Phi"]]


def compute_type(data_root, start=DEFAULT_START_NS, end=DEFAULT_END_NS,
                 step_size=DEFAULT_STEP_SIZE):
    """Build the tidy per-frame table for the per-species systems (panel D).

    Combines the protein+RNA system (w_p = 0.5) and the pure-protein system
    (w_p = 1.0), tagging each row with its ``state`` ("RNA"/"Pure") and ``w_p``.

    Args:
        data_root: Directory holding the ``single_protein_rna`` and
            ``single_protein_pure`` systems.
        start: Discard frames with Time < start (ns).
        end: Discard frames with Time > end (ns).
        step_size: Frame-delimiter modulus (LAMMPS steps).

    Returns:
        DataFrame with columns protein, state, w_p, Time, Cluster, RG, Phi.
    """
    frames = []
    for system, state, wp in ((TYPE_RNA_SYSTEM, "RNA", 0.5),
                              (TYPE_PURE_SYSTEM, "Pure", 1.0)):
        cluster_dir = _cluster_dir(data_root, system)
        effect = r"$\phi_{{P}}={}$".format(wp)
        for file_name, file_path in _iter_llps_files(cluster_dir):
            parts = file_name.split(".")[0].split("_")
            protein = parts[2]
            analyzer = cluster_analysis(file_path)
            file_df = analyzer.accumulate_windows(
                name=protein, species=system, effect=effect, step_size=step_size)
            file_df["state"] = state
            file_df["w_p"] = wp
            frames.append(file_df)

    if not frames:
        return pd.DataFrame(
            columns=["protein", "state", "w_p", "Time", "Cluster", "RG", "Phi"])
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[(df["Time"] >= start) & (df["Time"] <= end)].copy()
    df = df.rename(columns={"Protein": "protein"})
    df = df.sort_values(by=["protein", "w_p"]).reset_index(drop=True)
    return df[["protein", "state", "w_p", "Time", "Cluster", "RG", "Phi"]]


def write_clustering_csvs(out_dir, data_root=DEFAULT_DATA_ROOT,
                          start=DEFAULT_START_NS, end=DEFAULT_END_NS,
                          step_size=DEFAULT_STEP_SIZE):
    """Compute and write both per-frame clustering CSVs (panels C and D).

    Args:
        out_dir: Directory to create/use for the CSVs.
        data_root: Per-system data root (see module docstring).
        start, end: Time window in ns.
        step_size: Frame-delimiter modulus (LAMMPS steps).

    Returns:
        Tuple ``(percent_csv_path, type_csv_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)

    percent_df = compute_percent(data_root, start=start, end=end, step_size=step_size)
    percent_csv = os.path.join(out_dir, "percent_clustering.csv")
    percent_df.to_csv(percent_csv, index=False)
    print("[cluster_analysis] wrote {} ({} rows)".format(percent_csv, len(percent_df)))

    type_df = compute_type(data_root, start=start, end=end, step_size=step_size)
    type_csv = os.path.join(out_dir, "type_clustering.csv")
    type_df.to_csv(type_csv, index=False)
    print("[cluster_analysis] wrote {} ({} rows)".format(type_csv, len(type_df)))

    return percent_csv, type_csv


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cluster metrics (phi / RG / N_D) for Figure 1 "
                    "panels C and D (writes tidy CSVs; no plotting).")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="Directory holding the per-system subfolders, each "
                             "with a CLUSTER/ dir of llps_rg_*.out dumps "
                             "(default: simulation_inputs/model_systems; pass "
                             "any per-system CLUSTER results dump).")
    parser.add_argument("--out", default=".",
                        help="Output directory for the clustering CSVs "
                             "(default: current directory).")
    parser.add_argument("--start", type=float, default=DEFAULT_START_NS,
                        help="Discard frames before this time, ns "
                             "(default: %(default)s).")
    parser.add_argument("--end", type=float, default=DEFAULT_END_NS,
                        help="Discard frames after this time, ns "
                             "(default: %(default)s).")
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE,
                        help="LAMMPS steps between cluster frames, used as the "
                             "frame-delimiter modulus (default: %(default)s).")
    return parser.parse_args()


def main():
    args = _parse_args()
    write_clustering_csvs(args.out, data_root=args.data_root,
                          start=args.start, end=args.end,
                          step_size=args.step_size)


if __name__ == "__main__":
    main()
