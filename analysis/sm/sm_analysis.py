#!/usr/bin/env python3
"""Solubility analysis for pure small-molecule (SM) simulations.

Generates clustering metrics, box-centred radial density profiles, and
centre-of-mass radial distribution functions so that we can demonstrate that
isolated small molecules remain dispersed and do not phase-separate.

Outputs (written to ANALYSIS_SM/) for a given system name:
  - Cluster_SM_<name>.csv
  - Cluster_SM_SUMMARY_<name>.csv
  - Density_Profile_SM_Number_<name>.csv
  - Density_Profile_SM_Mass_<name>.csv
  - RDF_SM_COM_<name>.csv

The CSV schemas mirror the existing stress-granule pipeline so that the files
can be ingested by the same downstream tooling.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.transformations import wrap


# ---------------------------------------------------------------------------
# Atom-type masses (g/mol) copied from the coarse-grained parameterisation.
# ---------------------------------------------------------------------------
MASS_DEFAULTS: Dict[int, float] = {
    1: 131.182200,
    2: 57.042200,
    3: 128.182600,
    4: 101.086300,
    5: 156.178800,
    6: 71.070340,
    7: 115.084400,
    8: 129.082500,
    9: 163.177800,
    10: 99.056500,
    11: 113.184600,
    12: 128.082600,
    13: 186.174700,
    14: 147.180000,
    15: 87.068200,
    16: 137.081400,
    17: 114.084500,
    18: 97.106800,
    19: 103.084500,
    20: 113.184600,
    21: 329.200000,
    22: 305.200000,
    23: 345.200000,
    24: 306.200000,
    25: 156.140680,
    26: 180.203840,
    27: 194.236100,
    28: 113.223490,
    29: 110.199580,
    30: 125.104050,
    31: 136.107080,
    32: 104.108280,
    33: 158.287740,
    34: 78.129220,
    35: 104.152440,
    36: 118.176380,
    37: 108.216760,
    38: 102.133500,
    39: 158.177540,
    40: 107.132190,
    41: 134.134620,
    42: 81.050770,
    43: 139.130990,
    44: 173.212450,
    45: 101.125530,
    46: 107.208790,
    47: 145.160890,
    48: 100.140800,
    49: 105.192850,
    50: 101.125530,
    51: 105.192850,
    52: 188.139480,
    53: 141.193410,
    54: 115.155470,
    55: 195.244070,
    56: 92.120520,
    57: 91.112550,
    58: 117.170730,
    59: 174.245980,
    60: 112.538610,
    61: 106.124220,
    62: 181.301770,
}


def _parse_mass_block(data_file: str) -> Dict[int, float]:
    """Parse the ``Masses`` section of a LAMMPS data file if present."""

    mapping: Dict[int, float] = {}
    try:
        with open(data_file, "r") as fh:
            lines = fh.readlines()
    except (OSError, IOError):
        return mapping

    in_block = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if in_block:
                break
            continue
        lower = line.lower()
        if not in_block and lower.startswith("masses"):
            in_block = True
            continue
        if in_block:
            if lower[0].isalpha():
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
                val = float(parts[1])
            except ValueError:
                continue
            mapping[idx] = val
    return mapping


def _has_atoms_section(data_file: str) -> bool:
    """Return True if the LAMMPS data file contains an ``Atoms`` section."""
    try:
        with open(data_file, "r") as fh:
            for line in fh:
                stripped = line.strip().lower()
                if not stripped:
                    continue
                if stripped[0].isdigit():
                    continue
                if stripped.startswith("atoms"):
                    return True
    except (OSError, IOError):
        return False
    return False


@dataclass
class summary_stats:
    """At-a-glance dispersion metrics for one pure-SM system.

    Aggregates the clustering, RDF, and mass-density diagnostics into a single
    record written to ``Cluster_SM_SUMMARY_<name>.csv``. RDF/RDP-derived fields
    are filled in a second pass by :func:`augment_summary`.

    Attributes:
        largest_fraction_mean/max/p95: Largest-cluster fraction statistics over
            the analysed frames (fraction of molecules in the biggest cluster).
        fraction_frames_over_0p1: Fraction of frames where that exceeds 0.1.
        rdf_tail_mean: Mean COM g(r) in the long-range tail (r >= 200 A).
        rdf_peak_max: Maximum COM g(r).
        rdp_mass_cv: Coefficient of variation of the radial mass density.
        frame_dt_ps: Median time spacing between analysed frames (ps).
    """

    largest_fraction_mean: float
    largest_fraction_max: float
    largest_fraction_p95: float
    fraction_frames_over_0p1: float
    rdf_tail_mean: float
    rdf_peak_max: float
    rdp_mass_cv: float
    frame_dt_ps: float


class dsu:
    """Disjoint-set union (union-find) over residues, for cluster labelling.

    Groups molecules connected by within-cutoff bead pairs into clusters. Uses
    path-halving in :meth:`find` and union-by-size in :meth:`union` for near
    linear-time merging.

    Attributes:
        parent: Per-element parent pointer (root encodes the cluster).
        size: Subtree size of each root, used for union-by-size.

    Usage:
        d = dsu(n_residues); d.union(i, j); root = d.find(i)
    """

    def __init__(self, n: int) -> None:
        """Initialise ``n`` singleton clusters (each element its own root)."""
        self.parent = np.arange(n, dtype=int)
        self.size = np.ones(n, dtype=int)

    def find(self, x: int) -> int:
        """Return the cluster root of element ``x`` (with path halving)."""
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Merge the clusters containing ``a`` and ``b`` (union by size)."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full SM analysis driver."""
    parser = argparse.ArgumentParser(
        description="Analyse pure small-molecule simulations for clustering and densities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_file", help="LAMMPS data file (for topology/masses)")
    parser.add_argument("trajectory", help="LAMMPS dump trajectory of pure SM system")
    parser.add_argument("system", help="System name used in output filenames")
    parser.add_argument("--tmin", type=int, default=0, help="Start frame index")
    parser.add_argument("--tmax", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--dt", type=int, default=1, help="Frame stride for analysis")
    parser.add_argument("--cutoff", type=float, default=16.0, help="Clustering cutoff (Å)")
    parser.add_argument(
        "--rmax",
        type=float,
        default=1200.0,
        help="Maximum radius for density/RDF histograms (Å)",
    )
    parser.add_argument("--dr", type=float, default=20.0, help="Radial bin width (Å)")
    parser.add_argument(
        "--output-dir",
        default="ANALYSIS_SM",
        help="Directory where analysis CSVs will be written",
    )
    return parser.parse_args()


def load_universe(data_file: str, trajectory: str) -> Tuple[mda.Universe, Dict[int, float]]:
    """Construct an MDAnalysis universe with residues = molecules.

    Some pure-SM systems are bundled with a convenience ``sys.data`` that lacks
    coordinate columns. In that case MDAnalysis raises a ``ValueError`` about
    missing ``x/y/z`` fields. We handle this gracefully by falling back to
    building the universe from the dump alone.
    """

    read_kwargs = dict(
        atom_style="id mol type q xu yu zu",
        lammps_coordinate_convention="unwrapped",
    )

    mass_map = MASS_DEFAULTS.copy()
    mass_map.update(_parse_mass_block(data_file))

    use_data = _has_atoms_section(data_file)

    if use_data:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = mda.Universe(
                    data_file,
                    trajectory,
                    topology_format="DATA",
                    format="LAMMPSDUMP",
                    **read_kwargs,
                )
        except ValueError as err:
            msg = str(err)
            if "atom_style string missing required field" not in msg:
                raise
            use_data = False
            warnings.warn(
                "Falling back to trajectory-only load because data file is missing "
                "coordinate fields: %s" % msg,
                RuntimeWarning,
            )

    if not use_data:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(
                trajectory,
                format="LAMMPSDUMP",
                **read_kwargs,
            )

    # Guarantee residue information and sensible names.
    # Prefer grouping by LAMMPS 'mol' column from the dump if available.
    u.trajectory[0]
    ts = u.trajectory.ts
    data = getattr(ts, "data", {}) or {}
    mol_ids = None
    for key in ("mol", "molecule", "molecule-ID", "molid"):
        if key in data:
            mol_ids = np.asarray(data[key], dtype=int)
            break

    if mol_ids is not None and mol_ids.size == u.atoms.n_atoms and np.unique(mol_ids).size >= 1:
        if "resids" in u.atoms._topology.attrs:
            u.atoms.resids = mol_ids
        else:
            u.add_TopologyAttr("resids", mol_ids)
    else:
        if "resids" not in u.atoms._topology.attrs:
            u.add_TopologyAttr("resids", np.ones(u.atoms.n_atoms, dtype=int))

    # Ensure resnames exist and are set to 'SM'
    if hasattr(u.residues, "resnames"):
        u.residues.resnames = np.array(["SM"] * u.residues.n_residues, dtype=object)
    else:
        u.add_TopologyAttr("resnames", np.array(["SM"] * u.residues.n_residues, dtype=object))

    # Keep molecules whole when wrapping into the primary cell.
    u.trajectory.add_transformations(wrap(u.atoms, compound="residues"))

    return u, mass_map


def assign_masses(universe: mda.Universe, mass_map: Dict[int, float]) -> None:
    """Attach per-atom masses based on the 'type' column from the dump."""
    types = np.asarray(universe.atoms.types, dtype=int)
    masses = np.array([mass_map[int(t)] for t in types], dtype=float)
    if hasattr(universe.atoms, "hasattr") and universe.atoms.hasattr("masses"):
        universe.atoms.masses = masses
    else:
        universe.add_TopologyAttr("masses", masses)


def _iter_frames(universe: mda.Universe, tmin: int, tmax: int | None, dt: int) -> Iterable:
    """Yield trajectory time steps within the requested window."""

    traj = universe.trajectory
    start = tmin
    stop = len(traj) if tmax is None else min(tmax, len(traj))
    if start >= stop:
        raise ValueError("No frames selected: check tmin/tmax settings")
    traj[start]  # position the cursor
    for ts in traj[start:stop:dt]:
        yield ts


def compute_cluster_metrics(
    universe: mda.Universe,
    tmin: int,
    tmax: int | None,
    dt: int,
    cutoff: float,
) -> Tuple[pd.DataFrame, summary_stats]:
    """Return per-frame cluster statistics and summary metrics."""

    nres = universe.residues.n_residues
    atom_to_res = universe.atoms.resindices.astype(int)

    frames: list[int] = []
    times_ps: list[float] = []
    largest: list[int] = []
    largest_fraction: list[float] = []
    nclusters: list[int] = []

    for ts in _iter_frames(universe, tmin, tmax, dt):
        # FastNS expects float32 coordinates
        coords = universe.atoms.positions.astype(np.float32, copy=False)
        ns = FastNS(float(cutoff), coords, box=ts.dimensions.astype(np.float32))
        pairs = ns.self_search().get_pairs()

        dsu_obj = dsu(nres)
        if pairs.size:
            r1 = atom_to_res[pairs[:, 0]]
            r2 = atom_to_res[pairs[:, 1]]
            mask = r1 != r2
            for a, b in zip(r1[mask], r2[mask]):
                dsu_obj.union(int(a), int(b))

        roots = np.fromiter((dsu_obj.find(i) for i in range(nres)), dtype=int)
        _, counts = np.unique(roots, return_counts=True)

        largest_cluster = int(counts.max()) if counts.size else 1
        n_cluster = int(counts.size if counts.size else nres)

        frames.append(ts.frame)
        time_attr = getattr(ts, "time", math.nan)
        try:
            time_ps = float(time_attr)
        except (TypeError, ValueError):
            time_ps = math.nan
        times_ps.append(time_ps)
        largest.append(largest_cluster)
        largest_fraction.append(largest_cluster / float(nres))
        nclusters.append(n_cluster)

    df = pd.DataFrame(
        {
            "Frame": frames,
            "Time (ps)": times_ps,
            "LargestClusterSize": largest,
            "LargestClusterFraction": largest_fraction,
            "NumClusters": nclusters,
            "NumMolecules": nres,
        }
    )

    finite_times = df["Time (ps)"].replace([np.inf, -np.inf], np.nan).dropna()
    frame_dt_ps = float(np.mean(np.diff(finite_times))) if len(finite_times) > 1 else math.nan

    rdf_tail_mean = math.nan  # placeholder; filled later
    rdf_peak_max = math.nan
    summary = summary_stats(
        largest_fraction_mean=float(np.mean(largest_fraction)),
        largest_fraction_max=float(max(largest_fraction) if largest_fraction else math.nan),
        largest_fraction_p95=float(np.percentile(largest_fraction, 95) if largest_fraction else math.nan),
        fraction_frames_over_0p1=float(
            np.mean([lf > 0.1 for lf in largest_fraction]) if largest_fraction else math.nan
        ),
        rdf_tail_mean=rdf_tail_mean,
        rdf_peak_max=rdf_peak_max,
        rdp_mass_cv=math.nan,
        frame_dt_ps=frame_dt_ps,
    )

    return df, summary


def compute_rdp(
    universe: mda.Universe,
    tmin: int,
    tmax: int | None,
    dt: int,
    rmax: float,
    dr: float,
    *,
    mass_density: bool,
    mass_map: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Radial density profile from the box centre using molecule COMs."""

    r_edges = np.arange(0.0, rmax + dr, dr, dtype=float)
    if r_edges.size < 2:
        raise ValueError("Need at least two radial bins")
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_volume = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    weights = None
    if mass_density:
        assign_masses(universe, mass_map or MASS_DEFAULTS)
        weights = np.array([res.atoms.masses.sum() for res in universe.residues], dtype=float)

    accum: list[np.ndarray] = []
    for ts in _iter_frames(universe, tmin, tmax, dt):
        box = ts.dimensions[:3]
        center = np.asarray(box, dtype=float) / 2.0
        coms = universe.residues.center_of_mass(compound="residues")
        radial = np.linalg.norm(coms - center, axis=1)
        hist, _ = np.histogram(radial, bins=r_edges, weights=weights)
        accum.append(hist)

    if not accum:
        raise RuntimeError("No histogram data collected")

    h = np.vstack(accum)
    mean_counts = h.mean(axis=0)
    std_counts = h.std(axis=0, ddof=1) if h.shape[0] > 1 else np.zeros_like(mean_counts)
    sem_counts = std_counts / math.sqrt(h.shape[0]) if h.shape[0] > 1 else np.zeros_like(mean_counts)

    density = mean_counts / shell_volume
    density_std = std_counts / shell_volume
    density_sem = sem_counts / shell_volume

    if mass_density:
        avogadro = 6.02214076e23
        convert = (1.0 / avogadro) * 1e6 * 1e21  # g/mol per Å^3 -> mg/mL
        density *= convert
        density_std *= convert
        density_sem *= convert
        column = "SM mass density (mg/mL)"
    else:
        column = "SM number density (1/A^3)"

    return pd.DataFrame(
        {
            "Distance from center of mass (A)": r_mid,
            column: density,
            "Standard deviation": density_std,
            "Standard mean error": density_sem,
        }
    )


def compute_rdf(
    universe: mda.Universe,
    tmin: int,
    tmax: int | None,
    dt: int,
    rmax: float,
    dr: float,
) -> pd.DataFrame:
    """Pair radial distribution function using residue COMs."""

    r_edges = np.arange(0.0, rmax + dr, dr, dtype=float)
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    nbins = r_mid.size

    if universe.residues.n_residues < 2:
        return pd.DataFrame({"Distance (A)": r_mid, "g(r)": np.ones_like(r_mid)})

    g_accum = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=float)

    for ts in _iter_frames(universe, tmin, tmax, dt):
        nres = universe.residues.n_residues
        if nres < 2:
            continue
        coms = universe.residues.center_of_mass(compound="residues")
        dist = distances.distance_array(coms, coms, box=ts.dimensions)
        iu = np.triu_indices(nres, k=1)
        dij = dist[iu]

        hist, _ = np.histogram(dij, bins=r_edges)

        volume = np.prod(ts.dimensions[:3])
        rho = nres / volume if volume > 0 else 0.0
        shell_volume = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
        ideal = rho * shell_volume * nres
        ideal[ideal <= 0] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            contribution = np.divide(hist, ideal, out=np.zeros_like(hist, dtype=float), where=~np.isnan(ideal))
        g_accum += contribution
        counts += (~np.isnan(ideal)).astype(float)

    with np.errstate(invalid="ignore"):
        g = np.divide(g_accum, counts, out=np.ones_like(g_accum), where=counts > 0)

    return pd.DataFrame({"Distance (A)": r_mid, "g(r)": g})


def augment_summary(summary: summary_stats, rdp_mass: pd.DataFrame, rdf: pd.DataFrame) -> summary_stats:
    """Fill in RDF and RDP derived metrics for the summary row."""

    mass_col = "SM mass density (mg/mL)"
    if mass_col in rdp_mass:
        values = rdp_mass[mass_col].to_numpy()
        mean = float(np.mean(values)) if values.size else math.nan
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        cv = float(std / mean) if mean else math.nan
    else:
        cv = math.nan

    rdf_values = rdf["g(r)"].to_numpy()
    rdf_peak = float(np.nanmax(rdf_values)) if rdf_values.size else math.nan
    tail_mask = rdf["Distance (A)"] >= 200.0
    tail_vals = rdf.loc[tail_mask, "g(r)"].to_numpy() if tail_mask.any() else np.array([])
    tail_mean = float(np.nanmean(tail_vals)) if tail_vals.size else math.nan

    return summary_stats(
        largest_fraction_mean=summary.largest_fraction_mean,
        largest_fraction_max=summary.largest_fraction_max,
        largest_fraction_p95=summary.largest_fraction_p95,
        fraction_frames_over_0p1=summary.fraction_frames_over_0p1,
        rdf_tail_mean=tail_mean,
        rdf_peak_max=rdf_peak,
        rdp_mass_cv=cv,
        frame_dt_ps=summary.frame_dt_ps,
    )


def write_summary_csv(summary: summary_stats, outdir: str, system: str) -> None:
    """Write the summary record as a one-row ``Cluster_SM_SUMMARY_<system>.csv``."""
    row = {
        "largest_fraction_mean": summary.largest_fraction_mean,
        "largest_fraction_max": summary.largest_fraction_max,
        "largest_fraction_p95": summary.largest_fraction_p95,
        "fraction_frames_over_0p1": summary.fraction_frames_over_0p1,
        "rdf_tail_mean": summary.rdf_tail_mean,
        "rdf_peak_max": summary.rdf_peak_max,
        "rdp_mass_cv": summary.rdp_mass_cv,
        "frame_dt_ps": summary.frame_dt_ps,
    }
    summary_path = os.path.join(outdir, f"Cluster_SM_SUMMARY_{system}.csv")
    pd.DataFrame([row]).to_csv(summary_path, index=False)


def main() -> None:
    """CLI entry point: load the SM trajectory and write all analysis CSVs."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    universe, mass_map = load_universe(args.data_file, args.trajectory)
    assign_masses(universe, mass_map)

    print(
        f"[INFO] Loaded universe with {universe.atoms.n_atoms} atoms and "
        f"{universe.residues.n_residues} SM molecules across {len(universe.trajectory)} frames."
    )

    # ------------------------------------------------------------------
    # Clustering
    cluster_df, summary = compute_cluster_metrics(
        universe, args.tmin, args.tmax, args.dt, args.cutoff
    )
    cluster_path = os.path.join(args.output_dir, f"Cluster_SM_{args.system}.csv")
    cluster_df.to_csv(cluster_path, index=False)
    print(f"[OK] Wrote {cluster_path}")

    # ------------------------------------------------------------------
    # Radial density profiles (number and mass density)
    rdp_number = compute_rdp(
        universe,
        args.tmin,
        args.tmax,
        args.dt,
        args.rmax,
        args.dr,
        mass_density=False,
    )
    rdp_mass = compute_rdp(
        universe,
        args.tmin,
        args.tmax,
        args.dt,
        args.rmax,
        args.dr,
        mass_density=True,
        mass_map=mass_map,
    )

    rdp_number_path = os.path.join(
        args.output_dir, f"Density_Profile_SM_Number_{args.system}.csv"
    )
    rdp_mass_path = os.path.join(
        args.output_dir, f"Density_Profile_SM_Mass_{args.system}.csv"
    )
    rdp_number.to_csv(rdp_number_path, index=False)
    rdp_mass.to_csv(rdp_mass_path, index=False)
    print(f"[OK] Wrote {rdp_number_path}")
    print(f"[OK] Wrote {rdp_mass_path}")

    # ------------------------------------------------------------------
    # Centre-of-mass RDF
    rdf = compute_rdf(
        universe,
        args.tmin,
        args.tmax,
        args.dt,
        rmax=min(args.rmax, 600.0),
        dr=args.dr,
    )
    rdf_path = os.path.join(args.output_dir, f"RDF_SM_COM_{args.system}.csv")
    rdf.to_csv(rdf_path, index=False)
    print(f"[OK] Wrote {rdf_path}")

    # Summary CSV with at-a-glance metrics
    summary = augment_summary(summary, rdp_mass, rdf)
    write_summary_csv(summary, args.output_dir, args.system)
    print(
        f"[OK] Wrote {os.path.join(args.output_dir, f'Cluster_SM_SUMMARY_{args.system}.csv')}"
    )


if __name__ == "__main__":
    main()
