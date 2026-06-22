#!/usr/bin/env python3
"""Pre-segment raw LAMMPS stress dumps into per-window per-residue NPZ files.

Run this ONCE per system BEFORE the parallel RCC_ANALYSIS windows.  It reads
the full raw stress dump (huge), aggregates per-atom stress to per-biopolymer-
residue level, and writes one small NPZ per 50 ns window:

    ANALYSIS_{folder}/Stress_Segmented_{system}_{window_tmin}.npz

Each NPZ contains:
    timesteps   : int64  (M,)       — LAMMPS timestep for each block in this window
    sg_resids   : int32  (R,)       — biopolymer resid axis (shared across windows)
    pxx..pyz    : float32 (M, R)×6  — per-residue stress sums per block

The parallel RCC windows then each read their own NPZ and apply tracked-cluster
membership to produce ``Stress_Tensor_Tracked_*.csv``.

Role: preprocessing step feeding the Green-Kubo viscosity branch of the
RCC pipeline (no figure of its own; upstream of viscosity.py / acf.py viscosity).

Key inputs:
    --path    source data root (e.g. .../PYTHON_ANALYSIS/TEMP_300)
    --folder  system category (SG | DSM | NDSM)
    --temp    temperature label (used only to organize the run)
    --system  optional single system; default processes all in the folder
    --window-ns  window width in ns (default 50)
    Reads <path>/<FOLDER>/GRO/*.data (topology) and
    <path>/<FOLDER>/STRESS/<system>_stress.out.all (raw per-atom stress dump).

Key outputs:
    ANALYSIS_{FOLDER}/Stress_Segmented_{system}_{window_tmin}.npz

Exact CLI invocation:
    python segment_stress.py --path PATH --folder {SG,DSM,NDSM} --temp TEMP \
        [--system NAME] [--window-ns 50]
"""
import argparse
import os
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Topology helpers (minimal, no trajectory needed)
# ---------------------------------------------------------------------------

def _load_topology(data_file):
    """Load LAMMPS data file and return (universe, sg_resids, atom_to_col)."""
    import MDAnalysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = MDAnalysis.Universe(
            data_file,
            topology_format="DATA",
            atom_style="id resid type charge x y z",
        )
    u.add_TopologyAttr('name', [str(t) for t in u.atoms.types])
    u.add_TopologyAttr('resnames', np.array(['UNK'] * u.residues.n_residues, dtype=object))

    # Label residues by type ranges (MPiPi convention)
    types = np.array([int(t) for t in u.atoms.types])
    prot_sel = " or ".join(f"type {i}" for i in range(1, 21))
    for res in u.select_atoms(prot_sel).residues:
        res.resname = "Protein"
    rna_sel = " or ".join(f"type {i}" for i in range(21, 25))
    for res in u.select_atoms(rna_sel).residues:
        res.resname = "RNA"
    sm_sel = " or ".join(f"type {i}" for i in range(25, 63))
    for res in u.select_atoms(sm_sel).residues:
        res.resname = "SM"

    # Build biopolymer resid list and atom→resid_col mapping
    bio = u.select_atoms("resname Protein or resname RNA")
    sg_resids = np.array(
        sorted(set(int(r) for r in bio.residues.resids)),
        dtype=np.int32,
    )
    resid_to_col = {int(r): j for j, r in enumerate(sg_resids)}

    atom_to_col = {}
    for res in bio.residues:
        col = resid_to_col.get(int(res.resid))
        if col is not None:
            for aid in res.atoms.ids:
                atom_to_col[int(aid)] = col

    return u, sg_resids, atom_to_col


def _segment_stress(stress_file, sg_resids, atom_to_col, out_dir, system_name,
                    window_ns=50, timestep_fs=20.0):
    """Read full stress dump and write per-window NPZ files."""
    n_res = len(sg_resids)
    ts_to_ns = timestep_fs * 1e-6  # LAMMPS step → ns

    # Accumulate blocks into per-window buckets
    # Key: window_tmin_ns (int), Value: dict of lists
    windows = {}

    pxx = np.zeros(n_res, dtype=np.float64)
    pyy = np.zeros(n_res, dtype=np.float64)
    pzz = np.zeros(n_res, dtype=np.float64)
    pxy = np.zeros(n_res, dtype=np.float64)
    pxz = np.zeros(n_res, dtype=np.float64)
    pyz = np.zeros(n_res, dtype=np.float64)
    n = 0
    timestep = 0
    n_blocks = 0

    def _flush():
        """Append the current block's per-residue stress to its time window."""
        nonlocal n, n_blocks
        if n == 0:
            return
        t_ns = timestep * ts_to_ns
        win_tmin = int(t_ns // window_ns) * window_ns
        if win_tmin not in windows:
            windows[win_tmin] = {
                "timesteps": [], "pxx": [], "pyy": [], "pzz": [],
                "pxy": [], "pxz": [], "pyz": [],
            }
        w = windows[win_tmin]
        w["timesteps"].append(timestep)
        w["pxx"].append(pxx.copy())
        w["pyy"].append(pyy.copy())
        w["pzz"].append(pzz.copy())
        w["pxy"].append(pxy.copy())
        w["pxz"].append(pxz.copy())
        w["pyz"].append(pyz.copy())
        n_blocks += 1

    print(f"Reading stress dump: {stress_file}", flush=True)
    with open(stress_file, "r") as fh:
        for line in fh:
            if line.strip() == "ITEM: TIMESTEP":
                _flush()
                timestep = int(next(fh))
                pxx[:] = 0.0
                pyy[:] = 0.0
                pzz[:] = 0.0
                pxy[:] = 0.0
                pxz[:] = 0.0
                pyz[:] = 0.0
                n = 0
            elif len(line.strip().split()) == 16:
                atm = line.split()
                col = atom_to_col.get(int(atm[0]))
                if col is not None:
                    pxx[col] += float(atm[6])
                    pyy[col] += float(atm[7])
                    pzz[col] += float(atm[8])
                    pxy[col] += float(atm[9])
                    pxz[col] += float(atm[10])
                    pyz[col] += float(atm[11])
                    n += 1
    _flush()
    print(f"Read {n_blocks} stress blocks across {len(windows)} windows", flush=True)

    # Write per-window NPZ files
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for win_tmin in sorted(windows):
        w = windows[win_tmin]
        out_path = os.path.join(out_dir, f"Stress_Segmented_{system_name}_{win_tmin}.npz")
        np.savez_compressed(
            out_path,
            timesteps=np.array(w["timesteps"], dtype=np.int64),
            sg_resids=sg_resids,
            pxx=np.array(w["pxx"], dtype=np.float32),
            pyy=np.array(w["pyy"], dtype=np.float32),
            pzz=np.array(w["pzz"], dtype=np.float32),
            pxy=np.array(w["pxy"], dtype=np.float32),
            pxz=np.array(w["pxz"], dtype=np.float32),
            pyz=np.array(w["pyz"], dtype=np.float32),
        )
        print(f"  {out_path}  ({len(w['timesteps'])} blocks)", flush=True)
        written += 1
    return written


# ---------------------------------------------------------------------------
# File resolution helpers (mirrors rcc_analysis.py main block)
# ---------------------------------------------------------------------------

def _resolve_data_file(base_path, folder, system_name):
    """Return the LAMMPS data (topology) file path for one system.

    Maps the system category to its GRO/ data-file naming convention
    (SG control vs. DSM ``sim_Y_*`` vs. NDSM ``sim_N_*``). Raises
    ``ValueError`` for an unsupported folder.
    """
    folder = folder.upper()
    if folder == "SG":
        return os.path.join(base_path, "GRO", "sys.data")
    if folder == "NDSM":
        return os.path.join(base_path, "GRO", f"sim_N_{system_name}_1uM.data")
    if folder == "DSM":
        return os.path.join(base_path, "GRO", f"sim_Y_{system_name}_1uM.data")
    raise ValueError(f"Unsupported folder '{folder}'")


SYSTEM_LISTS = {
    "SG": ["sg_X"],
    "DSM": [
        "dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoic_acid",
        "dsm_hydroxyquinoline", "dsm_lipoamide", "dsm_lipoic_acid",
        "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium",
        "dsm_quinicrine",
    ],
    "NDSM": [
        "ndsm_dmso", "ndsm_valeric_acid", "ndsm_ethylenediamine",
        "ndsm_propanedithiol", "ndsm_hexanediol", "ndsm_diethylaminopentane",
        "ndsm_aminoacridine", "ndsm_anthraquinone", "ndsm_acetylenapthacene",
        "ndsm_anacardic",
    ],
}


def main():
    """Parse CLI args and segment stress for one or all systems in a folder.

    Resolves each system's topology and raw stress dump, aggregates per-atom
    stress to per-residue per-window NPZ files, and exits non-zero if any
    system fails or no outputs are produced.
    """
    parser = argparse.ArgumentParser(
        description="Pre-segment raw LAMMPS stress dumps into per-window per-residue NPZ files."
    )
    parser.add_argument("--path", required=True,
                        help="Source data path, e.g. ./PYTHON_ANALYSIS/TEMP_300")
    parser.add_argument("--folder", required=True, choices=["SG", "DSM", "NDSM"],
                        help="System category")
    parser.add_argument("--temp", required=True, help="Temperature label (for output dir)")
    parser.add_argument("--system", default=None,
                        help="Single system name. If omitted, processes all systems in folder.")
    parser.add_argument("--window-ns", type=int, default=50,
                        help="Window width in ns (default: 50)")
    args = parser.parse_args()

    folder = args.folder.upper()
    base_path = os.path.join(args.path, folder)
    systems = [args.system] if args.system else SYSTEM_LISTS[folder]

    # Output directory is ANALYSIS_{FOLDER}/ under the working directory
    out_dir = f"ANALYSIS_{folder}/"
    failures = []
    produced = 0

    for name in systems:
        data_file = _resolve_data_file(base_path, folder, name)
        stress_file = os.path.join(base_path, "STRESS", f"{name}_stress.out.all")

        if not os.path.isfile(data_file):
            msg = f"{name}: data file not found: {data_file}"
            print(f"ERROR {msg}", flush=True)
            failures.append(msg)
            continue
        if not os.path.isfile(stress_file):
            msg = f"{name}: stress file not found: {stress_file}"
            print(f"ERROR {msg}", flush=True)
            failures.append(msg)
            continue

        print(f"\n=== Segmenting stress for {name} ===", flush=True)
        _, sg_resids, atom_to_col = _load_topology(data_file)
        print(f"Biopolymer resids: {len(sg_resids)}, mapped atoms: {len(atom_to_col)}", flush=True)

        written = _segment_stress(
            stress_file, sg_resids, atom_to_col, out_dir, name, window_ns=args.window_ns
        )
        if written <= 0:
            msg = f"{name}: no Stress_Segmented NPZ files were written"
            print(f"ERROR {msg}", flush=True)
            failures.append(msg)
            continue
        produced += written

    if failures:
        print("\nSegmentation completed with failures:", flush=True)
        for msg in failures:
            print(f"  - {msg}", flush=True)
        sys.exit(1)

    if produced <= 0:
        print("\nNo Stress_Segmented outputs were produced.", flush=True)
        sys.exit(1)

    print(f"\nDone. Wrote {produced} Stress_Segmented NPZ files.", flush=True)


if __name__ == "__main__":
    main()
