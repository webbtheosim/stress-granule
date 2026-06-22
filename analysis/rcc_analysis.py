"""Per-frame extraction of structural, dynamic, and contact observables from LAMMPS trajectories.

Pipeline role
-------------
Step 0 (first stage) of the stress-granule (SG) MD analysis pipeline. It is the
heaviest stage: it reads the raw LAMMPS / MDAnalysis trajectory for each system
and writes the per-time-window observables that every later stage
(``average_simulations.py``, ``max_cluster.py``, ``system_analysis.py``, ...)
consumes.

What it computes (per system, per time window)
----------------------------------------------
- The maximal continuous condensate cluster (overlap-tracked across frames) and
  its residue mask.
- Radial density profiles (protein / RNA / per-species / small molecule),
  mass-weighted and converted to mg/mL.
- PCA shape descriptors of the cluster.
- Domain-, residue-, and acid-level contact-count matrices, including
  small-molecule contact variants (16 A cutoff).
- Condensate-comoving mean-squared displacement (unbiased FFT kernel) plus a COM
  sidecar carrying per-chain COM, Rg, and Kirkwood Rh time series for the
  downstream diffusion / viscosity fits.
- Per-residue radius of gyration, and the condensate-only stress tensor for
  Green-Kubo viscosity.

Key inputs
----------
- LAMMPS topology ``GRO/...data`` and trajectory (``TRJ/<system>_whole.xtc`` when
  present, else raw ``TRJ/<system>_traj.lammpsdump``), plus per-system
  ``STRESS/``, ``CLUSTER/``, and ``MSD/`` ``.out.all`` files, all under
  ``<path>/<folder>/``.
- CLI flags: ``--path`` (base dir containing TEMP_* dirs), ``--folder``
  (DSM / NDSM / SG), ``--temp``, ``--tmin``, ``--tmax``, ``--dt`` (ns),
  ``--structure-coordinate-mode`` {auto, compact, historical}, ``--traj-source``
  {auto, whole, raw}, ``--system`` (optional single system). A legacy positional
  form ``<system> <tmin> <tmax> <dt>`` is also accepted.

Key outputs
-----------
Per-window CSV / ``.out.all`` / ``.npz`` products written under
``ANALYSIS_{SG,DSM,NDSM}/`` (density profiles, contact matrices, cluster masks,
PCA, stress tensors, ``<system>_msd_rdp.out.all`` and its
``<system>_msd_rdp_com.npz`` COM sidecar).

CLI invocation
--------------
    python rcc_analysis.py --path ./PYTHON_ANALYSIS/TEMP_300 \
        --folder DSM --temp 300 --tmin 50 --tmax 2000 --dt 50
"""
import csv
import math
import os
import re
import sys
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.decomposition
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance, self_capped_distance
from MDAnalysis.transformations import unwrap, center_in_box, wrap

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN_INTRACHAIN_EXCLUSION = int(os.environ.get("RCC_DOMAIN_INTRACHAIN_EXCLUSION", "5"))


def _is_whole_trajectory_path(traj_file):
    """Return True if ``traj_file`` is a pre-made-whole ``*_whole*.xtc`` trajectory (so MDAnalysis transforms must be disabled)."""
    base = os.path.basename(str(traj_file))
    return base.endswith(".xtc") and "_whole" in base


def _candidate_whole_meta_files(traj_file):
    """Return the ordered, de-duplicated list of candidate ``.meta.txt`` sidecar paths for a whole-trajectory file."""
    traj_file = str(traj_file)
    base, ext = os.path.splitext(traj_file)
    candidates = []
    if traj_file.endswith("_whole.xtc"):
        candidates.append(traj_file[:-len("_whole.xtc")] + "_whole_1ns.meta.txt")
    if traj_file.endswith("_whole_1ns.xtc"):
        candidates.append(traj_file[:-len(".xtc")] + ".meta.txt")
    candidates.append(base + ".meta.txt")
    out = []
    seen = set()
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _read_key_value_file(filepath):
    """Parse a simple ``key = value`` text file into a dict; return an empty dict on any read error."""
    data = {}
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    except Exception:
        return {}
    return data

def _capped_indices(pos_a, pos_b, radius, box):
    """
    Return index arrays (ia, ib) for atom pairs within ``radius`` using capped_distance.
    Compatible with MDAnalysis versions that return either (ia, ib) or an array of shape (n_pairs, 2).
    """
    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    if pos_a.size == 0 or pos_b.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    pos_a = np.atleast_2d(pos_a)
    pos_b = np.atleast_2d(pos_b)

    ret = capped_distance(pos_a, pos_b, max_cutoff=radius, box=box, return_distances=False)

    # Case 1: tuple/list (ia, ib) or (dist, ia, ib)
    if isinstance(ret, (tuple, list)):
        # last two are the index arrays in common MDAnalysis versions
        if len(ret) >= 2 and np.ndim(ret[-2]) == 1 and np.ndim(ret[-1]) == 1:
            ia = np.asarray(ret[-2], dtype=np.intp)
            ib = np.asarray(ret[-1], dtype=np.intp)
            return ia, ib
        # some versions return a single (n_pairs, 2) array
        if len(ret) == 1:
            arr = np.asarray(ret[0])
        else:
            arr = np.asarray([])
    else:
        # Case 2: ndarray of shape (n_pairs, 2)
        arr = np.asarray(ret)

    if arr.ndim == 2 and arr.shape[1] == 2:
        if arr.shape[0] == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        return arr[:, 0].astype(np.intp, copy=False), arr[:, 1].astype(np.intp, copy=False)

    # Fallback: no pairs
    return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)


def _self_capped_pairs(pos, radius, box):
    """
    Return unique index pairs (i < j) within ``radius`` using self_capped_distance.
    Handles both (ia, ib) and (n_pairs, 2) return formats.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    pos = np.atleast_2d(pos)

    ret = self_capped_distance(pos, max_cutoff=radius, box=box, return_distances=False)

    if isinstance(ret, (tuple, list)):
        if len(ret) >= 2 and np.ndim(ret[-2]) == 1 and np.ndim(ret[-1]) == 1:
            ia = np.asarray(ret[-2], dtype=np.intp)
            ib = np.asarray(ret[-1], dtype=np.intp)
        elif len(ret) == 1 and np.asarray(ret[0]).ndim == 2:
            arr = np.asarray(ret[0])
            if arr.shape[0] == 0:
                return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
            ia = arr[:, 0].astype(np.intp, copy=False)
            ib = arr[:, 1].astype(np.intp, copy=False)
        else:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    else:
        arr = np.asarray(ret)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        ia = arr[:, 0].astype(np.intp, copy=False)
        ib = arr[:, 1].astype(np.intp, copy=False)

    # enforce i < j to avoid double counting
    if ia.size:
        mask = ia < ib
        ia = ia[mask]
        ib = ib[mask]

    return ia, ib

class rdp:
    """Per-system trajectory analyzer that drives the full Step 0 extraction.

    Constructing an instance loads the MDAnalysis universe for one system, sets
    up coordinate transforms, identifies the maximal continuous condensate
    cluster, and (as a side effect of ``__init__``) computes and writes every
    per-window observable: radial density profiles, PCA shape descriptors,
    domain/residue/acid contact matrices (and SM variants), condensate-comoving
    MSD with the COM sidecar, per-chain Rg, and the condensate-only stress
    tensor. The methods are the individual extraction steps invoked via
    :meth:`_run_step`, which isolates failures so one bad observable does not
    abort the run.
    """

    def __init__(self, data_file, traj_file, stress_file, cluster_file, msd_file, system_name, tmin, tmax, dt,
                 cutoff, dims, bin_size, structure_coordinate_mode="auto"):
        """Load the universe for ``system_name`` and run the complete per-window extraction.

        Args:
            data_file: LAMMPS topology (``.data``) path.
            traj_file: Trajectory path (whole ``.xtc`` or raw ``.lammpsdump``).
            stress_file, cluster_file, msd_file: Per-system ``.out.all`` inputs.
            system_name: System label (e.g. ``sg_X``, ``dsm_lipoamide``); its
                prefix selects the ``ANALYSIS_<CAT>`` output folder.
            tmin, tmax, dt: Analysis window bounds and stride, interpreted in ns
                and converted to frame indices internally.
            cutoff, dims, bin_size: Cluster cutoff (A), box dimension, and RDP
                bin width.
            structure_coordinate_mode: One of ``auto``/``compact``/``historical``
                controlling the structural-observable coordinate construction
                (``auto`` resolves to ``historical``).
        """

        # Define Saving Directory and File
        # Output base folder determined by system prefix (SG/DSM/NDSM)
        self.system_name = system_name
        self.folder = f"ANALYSIS_{self.system_name.split('_')[0].upper()}/"
        self.stress_file = stress_file
        self.cluster_file = cluster_file
        self.msd_file = msd_file
        self.traj_file = traj_file
        self.using_whole_trajectory = _is_whole_trajectory_path(traj_file)
        requested_structure_mode = str(structure_coordinate_mode).strip().lower()
        if requested_structure_mode not in {"auto", "compact", "historical"}:
            raise ValueError(
                "Unsupported structure_coordinate_mode '{}'. Expected one of: auto, compact, historical".format(
                    requested_structure_mode
                )
            )
        if requested_structure_mode == "auto":
            # Always prefer historical mode: it uses PBC-aware distance_array
            # for RDP and direct cluster positions for PCA, matching V1 behavior.
            self.structure_coordinate_mode = "historical"
        else:
            self.structure_coordinate_mode = requested_structure_mode

        # Define MDAnalysis Universe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Define MDAnalysis Universe", flush=True)
            universe_kwargs = {
                "topology_format": "DATA",
                "atom_style": "id resid type charge x y z",
            }
            if str(traj_file).endswith(".lammpsdump"):
                universe_kwargs["lammps_coordinate_convention"] = "unwrapped"
            self.u = MDAnalysis.Universe(data_file, traj_file, **universe_kwargs)
            print("Trajectory Frames: {}".format(len(self.u.trajectory)), flush=True)
            self.u.add_TopologyAttr('name', [str(t) for t in self.u.atoms.types])
            self.u.add_TopologyAttr('resnames', np.array(['UNK'] * self.u.residues.n_residues, dtype=object))
            self.define_masses()
            assert self.u.residues.n_residues > 1, \
                "No residues were created. Check atom_style: use 'id resid type charge x y z' for 'Atoms # full'."

        # Ensure output directory exists
        os.makedirs(self.folder, exist_ok=True)

        # Generate Component Resnames
        self.proteins = self.get_proteins()
        self.rnas = self.get_rna()
        self.sms = self.get_sm()
        self.sg_residue_groups = self.get_sg()
        self.sg_resids = np.array(
            [int(group.residues.resids[0]) for group in self.sg_residue_groups],
            dtype=np.int32,
        )
        self.max_cluster_occupancy_fraction = 0.95

        print("Proteins: {}".format(self.u.select_atoms("resname Protein*").residues.n_residues), flush=True)
        print("G3BP1: {}".format(self.u.select_atoms("resname ProteinG3BP1").residues.n_residues), flush=True)
        print("TDP43: {}".format(self.u.select_atoms("resname ProteinTDP43").residues.n_residues), flush=True)
        print("PABP1: {}".format(self.u.select_atoms("resname ProteinPABP1").residues.n_residues), flush=True)
        print("TTP: {}".format(self.u.select_atoms("resname ProteinTTP").residues.n_residues), flush=True)
        print("TIA1: {}".format(self.u.select_atoms("resname ProteinTIA1").residues.n_residues), flush=True)
        print("FUS: {}".format(self.u.select_atoms("resname ProteinFUS").residues.n_residues), flush=True)
        print("RNA: {}".format(self.u.select_atoms("resname RNA").residues.n_residues), flush=True)
        print("SM: {}".format(self.u.select_atoms("resname SM").residues.n_residues), flush=True)
        # keep an AtomGroup for transforms; use its residues only for counting
        sg_core = self.u.select_atoms("resname Protein* or resname RNA")
        self.sg_core = sg_core
        print("SG: {}".format(sg_core.residues.n_residues), flush=True)
        print(
            "Trajectory source: {} ({})".format(
                traj_file,
                "whole_xtc" if self.using_whole_trajectory else "raw_dump",
            ),
            flush=True,
        )
        print("Structure coordinate mode: {}".format(self.structure_coordinate_mode), flush=True)

        structure_transforms = self._build_structure_transforms()
        if structure_transforms:
            self.u.trajectory.add_transformations(*structure_transforms)
            print("Applied structural transforms: {}".format(self._describe_transforms(structure_transforms)), flush=True)
        else:
            print("Applied structural transforms: none (using whole trajectory coordinates directly)", flush=True)

        # Generate System Properties
        self.num_total_chains = self.get_total_chain_num()
        self.mass_total_chains = self.get_total_chain_mass()

        # Timing and frame mapping
        # LAMMPS real units: dt = 20 fs; dump stride = 10000 steps → 2e-10 s per saved frame
        self.timestep_fs = 20.0
        self.dump_stride_steps = 10000
        self.raw_frame_dt_s = self.timestep_fs * 1e-15 * self.dump_stride_steps
        self.frame_dt_s = self._infer_frame_dt_seconds()
        print("Frame spacing: {:.6f} ns".format(self.frame_dt_s * 1e9), flush=True)

        # Interpret CLI tmin/tmax/dt as nanoseconds (ns) and convert to frame indices
        def _to_float_ns(val, default=0.0):
            """Coerce ``val`` to a float (ns), returning ``default`` on failure."""
            try:
                return float(val)
            except Exception:
                return float(default)

        def _ns_to_frames(ns):
            """Convert a time in nanoseconds to the nearest saved-frame index."""
            try:
                val_ns = float(ns)
            except Exception:
                val_ns = 0.0
            seconds = val_ns * 1e-9
            return int(round(seconds / self.frame_dt_s))

        self.tmin = _ns_to_frames(tmin)
        tmax_frames = _ns_to_frames(tmax)
        self.tmax = tmax_frames if tmax_frames > 0 else len(self.u.trajectory)
        dt_frames = max(1, _ns_to_frames(dt))
        self.dt = dt_frames
        # Also store the ns values explicitly for RG output (matches RG_FINAL.py)
        self.tmin_ns = _to_float_ns(tmin)
        self.tmax_ns = _to_float_ns(tmax)
        self.dt_ns = _to_float_ns(dt)
        print(f"Start Frame: {self.tmin}", flush=True)
        print(f"End Frame: {self.tmax}", flush=True)
        print(f"Step Frames: {self.dt}", flush=True)
        self.cutoff = cutoff
        self.box_dimensions = dims
        self.bin_size = bin_size
        self.system_num = str(tmin)

        # Instantiate Droplet Properties
        self.cluster_num = {}
        self.cluster_rg = {}
        self.num_inner_chains = {}
        self.mass_inner_chains = {}
        self.num_outer_chains = {}
        self.mass_outer_chains = {}

        # Generate Frame Cluster Groups
        print("Generate Frame Cluster Groups", flush=True)
        self.cluster_group = {}
        self.cluster_array = {}
        self.cluster_bond = {}
        self.cluster_members = {}
        self._compact_cluster_cache = {}
        ok, _ = self._run_step(self.create_clusters, "create_clusters")
        if ok:
            print("Complete", flush=True)

        # Obtain Maximal Continuous Cluster
        print("Obtain Maximal Continuous Cluster", flush=True)
        self.max_selection = ""
        success, result = self._run_step(self.gen_maximal_continuous_cluster,
                                         "gen_maximal_continuous_cluster")
        self.max_continuous_cluster = result if success else None
        if success:
            self._run_step(self._write_tracked_cluster_sidecar,
                           "_write_tracked_cluster_sidecar")
            print("Complete", flush=True)

        # Generate Domain Contact Array
        print("Generate Domain Contact Array", flush=True)
        radius = 16
        ok, _ = self._run_step(self.gen_domain_contacts, "gen_domain_contacts", radius)
        if ok:
            print("Complete", flush=True)

        # Generate SM Domain Contact Array
        if self.u.select_atoms("resname SM").residues.n_residues > 0:
            print("Generate SM Domain Contact Array", flush=True)
            radius = 16
            ok, _ = self._run_step(self.gen_sm_domain_contacts, "gen_sm_domain_contacts", radius)
            if ok:
                print("Complete", flush=True)

        # Generate Density Profiles and Perform PCA
        print("Generate Density Profiles and Perform PCA", flush=True)
        print("SG Density Profile", flush=True)
        self._run_step(self.calc_rdp, "calc_rdp('SG')", "SG")
        self._run_step(self.calc_pca, "calc_pca('SG')", "SG")
        if self.u.select_atoms("resname Protein*").residues.n_residues > 0:
            print("Protein Analysis", flush=True)
            self._run_step(self.calc_rdp, "calc_rdp('Protein')", "Protein")
            self._run_step(self.calc_pca, "calc_pca('Protein')", "Protein")
            print("Protein Species Analysis", flush=True)
            for species in ["G3BP1", "TDP43", "PABP1", "FUS", "TIA1", "TTP"]:
                self._run_step(self.calc_rdp, f"calc_rdp('{species}')", species)
        if self.u.select_atoms("resname RNA").residues.n_residues > 0:
            print("RNA Analysis", flush=True)
            self._run_step(self.calc_rdp, "calc_rdp('RNA')", "RNA")
            self._run_step(self.calc_pca, "calc_pca('RNA')", "RNA")
            print("RNA ADENINE Analysis", flush=True)
            self._run_step(self.calc_rdp, "calc_rdp('ADENINE')", "ADENINE")
            print("RNA UCG Analysis", flush=True)
            self._run_step(self.calc_rdp, "calc_rdp('UCG')", "UCG")
        if self.u.select_atoms("resname SM").residues.n_residues > 0:
            print("SM Analysis", flush=True)
            self._run_step(self.calc_rdp_sm, "calc_rdp_sm")
        print("Complete", flush=True)

        #Generate Residue Contact Array
        print("Generate Residue Contact Array", flush=True)
        radius = 16
        ok, _ = self._run_step(self.gen_residue_contacts, "gen_residue_contacts", radius)
        if ok:
            print("Complete", flush=True)

        # Generate Acid Contact Array
        print("Generate Acid Contact Array", flush=True)
        radius = 16
        ok, _ = self._run_step(self.gen_acid_contacts, "gen_acid_contacts", radius)
        if ok:
            print("Complete", flush=True)

        # Generate SM Residue Contact Array
        if self.u.select_atoms("resname SM").residues.n_residues > 0:
            print("Generate SM Residue Contact Array", flush=True)
            radius = 16
            ok, _ = self._run_step(self.gen_sm_residue_contacts, "gen_sm_residue_contacts", radius)
            if ok:
                print("Complete", flush=True)

        # Generate SM Acid Contact Array
        if self.u.select_atoms("resname SM").residues.n_residues > 0:
            print("Generate SM Acid Contact Array", flush=True)
            radius2 = 16
            ok, _ = self._run_step(self.gen_sm_acid_contacts, "gen_sm_acid_contacts", radius2)
            if ok:
                print("Complete", flush=True)

        # Generate Self-Diffusivity
        print("Calculate Diffusion Coefficient", flush=True)
        for species in [
            "ProteinG3BP1", "ProteinTDP43", "ProteinPABP1",
            "ProteinFUS", "ProteinTTP", "ProteinTIA1", "RNA"
        ]:
            self._run_step(self.calc_diffusivities, f"calc_diffusivities('{species}')", select=species)
        print("Complete", flush=True)

        # LAMMPS-like MSD check file across full trajectory at analysis stride (COM-only)
        if self.tmin == 0:
            print("Generate LAMMPS-like MSD check (COM-only)", flush=True)
            self._run_step(self.calc_lammps_diffusivities, "calc_lammps_diffusivities")
            print("Complete", flush=True)

        # Generate per-chain RG time series (RG_<system>_rg.out.all, as in RG_FINAL.py)
        print("Calculate per-chain RG", flush=True)
        self._run_step(self.calc_rg, "calc_rg")
        print("Complete", flush=True)

        # Generate Stress Tensor
        print("Generate Stress Tensor", flush=True)
        if self.tmin == 0:
            self._run_step(self.parse_stress_file, "parse_stress_file", stress_file)
        # Tracked stress runs at every window (reads per-window segmented NPZ
        # produced by segment_stress.py, which must run before RCC windows)
        self._run_step(self.parse_stress_file_tracked, "parse_stress_file_tracked")
        print("Complete", flush=True)


    def _run_step(self, func, description, *args, **kwargs):
        """Execute a callable and log any exception without aborting the workflow."""
        try:
            return True, func(*args, **kwargs)
        except Exception as exc:
            print(f"{description} failed with error: {exc}", flush=True)
            traceback.print_exc()
            return False, None

    def _infer_frame_dt_seconds(self):
        """Return the seconds-per-saved-frame spacing, reading mdvwhole metadata for whole XTCs and falling back to the raw LAMMPS cadence."""
        if not self.using_whole_trajectory:
            return self.raw_frame_dt_s

        for meta_file in _candidate_whole_meta_files(self.traj_file):
            if not os.path.isfile(meta_file):
                continue
            meta = _read_key_value_file(meta_file)
            stride_val = meta.get("stride_frames")
            try:
                stride = int(stride_val)
            except Exception:
                stride = None
            if stride and stride > 0:
                print(
                    "Detected mdvwhole metadata: {} (stride_frames={})".format(meta_file, stride),
                    flush=True,
                )
                return self.raw_frame_dt_s * float(stride)

        base = os.path.basename(str(self.traj_file))
        if "_whole_1ns" in base or base.endswith("_whole.xtc"):
            print(
                "No mdvwhole metadata found for {}; assuming 1 ns/frame from filename.".format(self.traj_file),
                flush=True,
            )
            return 1e-9

        return self.raw_frame_dt_s

    @staticmethod
    def _describe_transforms(transforms):
        """Return a comma-separated string of the class names of a list of MDAnalysis transformations (for logging)."""
        names = []
        for transform in transforms:
            names.append(getattr(transform, "__class__", type(transform)).__name__)
        return ", ".join(names) if names else "none"

    def _build_structure_transforms(self):
        """Build the legacy structural transform chain (unwrap -> center SG by mass -> wrap residues) used for RDP/PCA/contacts."""
        if len(self.sg_core) == 0:
            return []
        # Restore the legacy V1/V2 structural path for RDP/PCA/contact maps:
        # unwrap -> center SG -> wrap residues into the primary cell.
        # The whole-XTC mdvwhole output is still recentred/wrapped this way in
        # legacy analysis before any PBC-aware structural distances are used.
        return [
            unwrap(self.u.atoms),
            center_in_box(self.sg_core, center='mass'),
            wrap(self.u.atoms, compound='residues'),
        ]

    def _build_diffusion_transforms(self):
        """Build the diffusion transform chain (center SG by mass, plus unwrap for raw dumps) that preserves continuous coordinates for MSD."""
        if len(self.sg_core) == 0:
            return []
        if self.using_whole_trajectory:
            return [center_in_box(self.sg_core, center='mass')]
        return [unwrap(self.u.atoms), center_in_box(self.sg_core, center='mass')]

    # Assign Bead Masses
    def define_masses(self):
        """Assign per-atom MPiPi bead masses (Da) by LAMMPS type, unless nonzero masses are already present on the universe."""
        masses = {
            1: 131.182200,  # 1
            2: 57.042200, # 2
            3: 128.182600,  # 3
            4: 101.086300, # 4
            5: 156.178800,  # 5
            6: 71.070340,  # 6
            7: 115.084400,  # 7
            8: 129.082500,  # 8
            9: 163.177800,  # 9
            10: 99.056500,  # 10
            11: 113.184600,  # 11
            12: 128.082600,  # 12
            13: 186.174700,  # 13
            14: 147.180000,  # 14
            15: 87.068200, # 15
            16: 137.081400,  # 16
            17: 114.084500,  # 17
            18: 97.106800, # 18
            19: 103.084500,  # 19
            20: 113.184600,  # 20
            21: 329.200000,  # 21
            22: 305.200000,  # 22
            23: 345.200000,  # 23
            24: 306.200000,  # 24
            25: 156.140680,  # 25
            26: 180.203840,  # 26
            27: 194.236100,  # 27
            28: 113.223490,  # 28
            29: 110.199580,  # 29
            30: 125.104050,  # 30
            31: 136.107080,  # 31
            32: 104.108280,  # 32
            33: 158.287740,  # 33
            34: 78.129220,  # 34
            35: 104.152440,  # 35
            36: 118.176380,  # 36
            37: 108.216760,  # 37
            38: 102.133500,  # 38
            39: 158.177540,  # 39
            40: 107.132190,  # 40
            41: 134.134620,  # 41
            42: 81.050770,  # 42
            43: 139.130990,  # 43
            44: 173.212450,  # 44
            45: 101.125530,  # 45
            46: 107.208790,  # 46
            47: 145.160890,  # 47
            48: 100.140800,  # 48
            49: 105.192850,  # 49
            50: 101.125530,  # 50
            51: 105.192850,  # 51
            52: 188.139480,  # 52
            53: 141.193410,  # 53
            54: 115.155470,  # 54
            55: 195.244070,  # 55
            56: 92.120520,  # 56
            57: 91.112550,  # 57
            58: 117.170730,  # 58
            59: 174.245980,  # 59
            60: 112.538610,  # 60
            61: 106.124220,  # 61
            62: 181.301770,  # 62
        }

        # If masses already present and nonzero, keep them
        try:
            if np.any(self.u.atoms.masses):
                return
        except Exception:
            # add the attribute if it doesn't exist yet
            self.u.add_TopologyAttr('masses', np.zeros(self.u.atoms.n_atoms, dtype=float))

        # Convert LAMMPS types -> integer IDs
        types = np.array([int(t) for t in self.u.atoms.types])

        # Build and assign per-atom masses
        self.u.atoms.masses = np.array([masses[t] for t in types], dtype=float)

    def get_proteins(self):
        """Tag protein residues (bead types 1-20) generically as ``Protein`` and specialize to ``Protein<SPECIES>`` by chain length; return the residue group."""
        # proteins are types 1..20  (adjust ranges here if needed)
        prot_sel = " or ".join(f"type {i}" for i in range(1, 21))
        prot_res = self.u.select_atoms(prot_sel).residues

        # always give a generic Protein tag; then specialize by length
        for res in prot_res:
            res.resname = "Protein"
            n = len(res.atoms)
            if   n == 466: res.resname = "ProteinG3BP1"
            elif n == 636: res.resname = "ProteinPABP1"
            elif n == 414: res.resname = "ProteinTDP43"
            elif n == 386: res.resname = "ProteinTIA1"
            elif n == 326: res.resname = "ProteinTTP"
            elif n == 526: res.resname = "ProteinFUS"
        return prot_res

    def get_rna(self):
        """Tag RNA residues (bead types 21-24) with resname ``RNA`` and return the residue group."""
        # RNA are types 21..24
        rna_sel = " or ".join(f"type {i}" for i in range(21, 25))
        rna_res = self.u.select_atoms(rna_sel).residues
        for res in rna_res:
            res.resname = "RNA"
        return rna_res

    def get_sm(self):
        """Tag small-molecule residues (bead types 25-62) with resname ``SM`` and return the residue group."""
        # small molecules are types 25..62
        sm_sel = " or ".join(f"type {i}" for i in range(25, 63))
        sm_res = self.u.select_atoms(sm_sel).residues
        for res in sm_res:
            res.resname = "SM"
        return sm_res

    def get_sg(self):
        """Return the stress-granule biopolymer chains (union of RNA + protein residues) as a list of single-residue AtomGroups."""
        # union of RNA + Protein residues (works with non-contiguous resid values)
        resids = np.unique(np.concatenate([self.rnas.resids, self.proteins.resids]))
        resids.sort()
        return [self.u.select_atoms(f"resid {rid}") for rid in resids]

    # Calculate Time Independent System Properties
    def get_total_chain_num(self):
        """Return a dict of total chain counts in the whole system, keyed by species (and SG total / SM)."""
        ret_dict = {"Protein": len(self.u.select_atoms("resname Protein*").residues),
                    "TDP43": len(self.u.select_atoms("resname ProteinTDP43").residues),
                    "FUS": len(self.u.select_atoms("resname ProteinFUS").residues),
                    "TIA1": len(self.u.select_atoms("resname ProteinTIA1").residues),
                    "G3BP1": len(self.u.select_atoms("resname ProteinG3BP1").residues),
                    "PABP1": len(self.u.select_atoms("resname ProteinPABP1").residues),
                    "TTP": len(self.u.select_atoms("resname ProteinTTP").residues),
                    "RNA": len(self.u.select_atoms("resname RNA").residues),
                    "ADENINE": len(self.u.select_atoms("type 21")),
                    "UCG": len(self.u.select_atoms("type 22 or type 23 or type 24")),
                    "SG": len(self.u.select_atoms("resname RNA or resname Protein*").residues),
                    "SM": len(self.u.select_atoms("resname SM").residues)}
        return ret_dict

    def get_total_chain_mass(self):
        """Return a dict of total chain masses (mg) in the whole system, keyed by species (and SG total / SM)."""
        ret_dict = {"Protein": sum(self.u.select_atoms("resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "TDP43": sum(self.u.select_atoms("resname ProteinTDP43").masses) * 1.66 * 10 ** (-21),
                    "FUS": sum(self.u.select_atoms("resname ProteinFUS").masses) * 1.66 * 10 ** (-21),
                    "TIA1": sum(self.u.select_atoms("resname ProteinTIA1").masses) * 1.66 * 10 ** (-21),
                    "G3BP1": sum(self.u.select_atoms("resname ProteinG3BP1").masses) * 1.66 * 10 ** (-21),
                    "PABP1": sum(self.u.select_atoms("resname ProteinPABP1").masses) * 1.66 * 10 ** (-21),
                    "TTP": sum(self.u.select_atoms("resname ProteinTTP").masses) * 1.66 * 10 ** (-21),
                    "RNA": sum(self.u.select_atoms("resname RNA").masses) * 1.66 * 10 ** (-21),
                    "ADENINE": sum(self.u.select_atoms("type 21").masses) * 1.66 * 10 ** (-21),
                    "UCG": sum(self.u.select_atoms("type 22 or type 23 or type 24").masses) * 1.66 * 10 ** (-21),
                    "SG": sum(self.u.select_atoms("resname RNA or resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "SM": sum(self.u.select_atoms("resname SM").masses) * 1.66 * 10 ** (-21)}
        return ret_dict

    # Calculate Time Dependent Cluster Properties
    def get_ave_length(self, cluster):
        """Return the rounded average chain length (atoms per residue) of an AtomGroup ``cluster``."""
        ave_length = int(np.round(len(cluster.atoms) / len(cluster.residues)))
        return ave_length

    def get_ave_mass(self, cluster):
        """Return the average residue mass in Daltons for an AtomGroup ``cluster``."""
        # Return average residue mass in Daltons (amu)
        nres = max(1, len(cluster.residues))
        return float(np.sum(cluster.masses) / nres)

    def get_inner_chain_num(self, cluster):
        """Return a dict of per-species chain counts inside the condensate ``cluster``."""
        ret_dict = {"Protein": len(cluster.select_atoms("resname Protein*").residues),
                    "TDP43": len(cluster.select_atoms("resname ProteinTDP43").residues),
                    "FUS": len(cluster.select_atoms("resname ProteinFUS").residues),
                    "TIA1": len(cluster.select_atoms("resname ProteinTIA1").residues),
                    "G3BP1": len(cluster.select_atoms("resname ProteinG3BP1").residues),
                    "PABP1": len(cluster.select_atoms("resname ProteinPABP1").residues),
                    "TTP": len(cluster.select_atoms("resname ProteinTTP").residues),
                    "RNA": len(cluster.select_atoms("resname RNA").residues),
                    "ADENINE": len(cluster.select_atoms("type 21")),
                    "UCG":     len(cluster.select_atoms("type 22 or type 23 or type 24")),
                    "SG": len(cluster.select_atoms("resname RNA or resname Protein*").residues),
                    "SM": len(cluster.select_atoms("resname SM").residues)}
        return ret_dict

    def get_inner_chain_mass(self, cluster):
        """Return a dict of per-species chain masses (mg) inside the condensate ``cluster``."""
        ret_dict = {"Protein": sum(cluster.select_atoms("resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "TDP43": sum(cluster.select_atoms("resname ProteinTDP43").masses) * 1.66 * 10 ** (-21),
                    "FUS": sum(cluster.select_atoms("resname ProteinFUS").masses) * 1.66 * 10 ** (-21),
                    "TIA1": sum(cluster.select_atoms("resname ProteinTIA1").masses) * 1.66 * 10 ** (-21),
                    "G3BP1": sum(cluster.select_atoms("resname ProteinG3BP1").masses) * 1.66 * 10 ** (-21),
                    "PABP1": sum(cluster.select_atoms("resname ProteinPABP1").masses) * 1.66 * 10 ** (-21),
                    "TTP": sum(cluster.select_atoms("resname ProteinTTP").masses) * 1.66 * 10 ** (-21),
                    "RNA": sum(cluster.select_atoms("resname RNA").masses) * 1.66 * 10 ** (-21),
                    "ADENINE": float(cluster.select_atoms("type 21").masses.sum()) * 1.66e-21,
                    "UCG":     float(cluster.select_atoms("type 22 or type 23 or type 24").masses.sum()) * 1.66e-21,
                    "SG": sum(cluster.select_atoms("resname RNA or resname Protein*").masses) * 1.66 * 10 ** (-21),
                    "SM": sum(cluster.select_atoms("resname SM").masses) * 1.66 * 10 ** (-21)}
        return ret_dict


    def get_outer_chain_num(self, cluster):
        """Return a dict of per-species chain counts outside the condensate ``cluster`` (total minus inner)."""
        return {
            "Protein": self.num_total_chains["Protein"] - self.get_inner_chain_num(cluster)["Protein"],
            "TDP43":   self.num_total_chains["TDP43"]   - self.get_inner_chain_num(cluster)["TDP43"],
            "FUS":     self.num_total_chains["FUS"]     - self.get_inner_chain_num(cluster)["FUS"],
            "TIA1":    self.num_total_chains["TIA1"]    - self.get_inner_chain_num(cluster)["TIA1"],
            "G3BP1":   self.num_total_chains["G3BP1"]   - self.get_inner_chain_num(cluster)["G3BP1"],
            "PABP1":   self.num_total_chains["PABP1"]   - self.get_inner_chain_num(cluster)["PABP1"],
            "TTP":     self.num_total_chains["TTP"]     - self.get_inner_chain_num(cluster)["TTP"],
            "RNA":     self.num_total_chains["RNA"]     - self.get_inner_chain_num(cluster)["RNA"],
            "ADENINE": self.num_total_chains["ADENINE"] - self.get_inner_chain_num(cluster)["ADENINE"],
            "UCG":     self.num_total_chains["UCG"]     - self.get_inner_chain_num(cluster)["UCG"],
            "SG":      self.num_total_chains["SG"]      - self.get_inner_chain_num(cluster)["SG"],
            "SM":      self.num_total_chains["SM"]      - self.get_inner_chain_num(cluster)["SM"],
        }

    def get_outer_chain_mass(self, cluster):
        """Return a dict of per-species chain masses (mg) outside the condensate ``cluster`` (total minus inner)."""
        return {
            "Protein": self.mass_total_chains["Protein"] - self.get_inner_chain_mass(cluster)["Protein"],
            "TDP43":   self.mass_total_chains["TDP43"]   - self.get_inner_chain_mass(cluster)["TDP43"],
            "FUS":     self.mass_total_chains["FUS"]     - self.get_inner_chain_mass(cluster)["FUS"],
            "TIA1":    self.mass_total_chains["TIA1"]    - self.get_inner_chain_mass(cluster)["TIA1"],
            "G3BP1":   self.mass_total_chains["G3BP1"]   - self.get_inner_chain_mass(cluster)["G3BP1"],
            "PABP1":   self.mass_total_chains["PABP1"]   - self.get_inner_chain_mass(cluster)["PABP1"],
            "TTP":     self.mass_total_chains["TTP"]     - self.get_inner_chain_mass(cluster)["TTP"],
            "RNA":     self.mass_total_chains["RNA"]     - self.get_inner_chain_mass(cluster)["RNA"],
            "ADENINE": self.mass_total_chains["ADENINE"] - self.get_inner_chain_mass(cluster)["ADENINE"],
            "UCG":     self.mass_total_chains["UCG"]     - self.get_inner_chain_mass(cluster)["UCG"],
            "SG":      self.mass_total_chains["SG"]      - self.get_inner_chain_mass(cluster)["SG"],
            "SM":      self.mass_total_chains["SM"]      - self.get_inner_chain_mass(cluster)["SM"],
        }

    @staticmethod
    def _kirkwood_rh(positions):
        """Kirkwood hydrodynamic radius from bead positions.

        Rh^{-1} = (2 / N^2) * sum_{i<j} 1 / |r_i - r_j|

        Parameters
        ----------
        positions : (N, 3) array
            Atom positions in Angstrom.

        Returns
        -------
        float
            Rh in Angstrom, or NaN if fewer than 2 atoms.
        """
        N = len(positions)
        if N < 2:
            return np.nan
        # Pairwise distance matrix (upper triangle only via broadcasting)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Upper triangle (i < j) only
        iu = np.triu_indices(N, k=1)
        d_upper = dist[iu]
        d_upper = d_upper[d_upper > 0.0]
        if len(d_upper) == 0:
            return np.nan
        inv_rh = (2.0 / (N * N)) * np.sum(1.0 / d_upper)
        return 1.0 / inv_rh if inv_rh > 0.0 else np.nan

    @staticmethod
    def _msd_fft_total(pos):
        """
        Unbiased total MSD from Cartesian positions using FFT.

        Args:
            pos: (n_time, 3) or (n_time, n_series, 3) positions in Angstrom.
        """
        arr = np.asarray(pos, dtype=float)
        squeeze = False
        if arr.ndim == 2:
            arr = arr[:, None, :]
            squeeze = True
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("pos must have shape (n_time, 3) or (n_time, n_series, 3)")

        T = arr.shape[0]
        n_series = arr.shape[1]
        if T == 1:
            out = np.zeros((1, n_series), dtype=float)
            return out[:, 0] if squeeze else out

        def _autocorr_fft(x):
            """Unbiased autocorrelation of ``x`` (n_time x n_series) along axis 0 via FFT."""
            nfft = 1 << (2 * T - 1).bit_length()
            fx = np.fft.rfft(x, n=nfft, axis=0)
            ac = np.fft.irfft(fx * np.conj(fx), n=nfft, axis=0)[:T]
            norm = (T - np.arange(T)).astype(float)[:, None]
            return ac / norm

        sq = np.sum(arr * arr, axis=2)
        sq_pad = np.vstack([sq, np.zeros((1, n_series), dtype=float)])
        s2 = sum(_autocorr_fft(arr[:, :, d]) for d in range(3))

        q = 2.0 * np.sum(sq, axis=0)
        s1 = np.zeros((T, n_series), dtype=float)
        for m in range(T):
            q = q - sq_pad[m - 1] - sq_pad[T - m]
            s1[m, :] = q / float(T - m)

        msd_tot = np.maximum(s1 - 2.0 * s2, 0.0)
        msd_tot[0, :] = 0.0
        return msd_tot[:, 0] if squeeze else msd_tot


    # Generate Time Dependent Clusters
    def create_clusters(self):
        """Build the per-frame chain connectivity (proximity graph) and cache the largest cluster's group, labels, bond matrix, and member indices for each analyzed frame."""
        nres = len(self.u.select_atoms("resname Protein* or resname RNA").residues)
        res_list = self.sg_residue_groups
        res = np.arange(-0.5, nres + 0.5, 1)

        for t in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            boxa = t.dimensions
            # use compact dtypes to reduce memory footprint
            bond = np.zeros((nres, nres), dtype=np.uint8)
            cluster = np.zeros(nres, dtype=np.int32)
            c = self.gen_cluster(nres, bond, cluster, res, res_list, boxa)
            cluster = c[0]
            clus = c[1]
            member_idx = c[2]
            self.cluster_group[t.frame] = cluster
            self.cluster_array[t.frame] = clus
            self.cluster_bond[t.frame] = bond.copy()
            self.cluster_members[t.frame] = member_idx

    def calc_cluster(self, cluster, n, proximity):
        """Assign each of ``n`` chains to a connected component by iterative label propagation over the ``proximity`` matrix; return the cluster-label array."""
        for i in range(n):  # start with each particle in its own cluster
            cluster[i] = i
        nchange = 1
        while nchange > 0:
            nchange = 0  # change cluster assignment until convergence
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if proximity[i, j] == 1:
                        if cluster[i] != cluster[j]:
                            nchange += 1
                            ii = min(cluster[i], cluster[j])  # reduce cluster assignment to lowest cluster index
                            cluster[i] = ii
                            cluster[j] = ii
        return (cluster)

    def gen_cluster(self, nres, bond, cluster, res, res_list, boxa):
        """Fill the PBC-aware proximity matrix ``bond`` (chains within ``self.cutoff``), find the largest connected cluster, and return its AtomGroup, label array, and member indices."""
        for i in range(nres - 1):  # construct the "proximity" matrix.
            for j in range(i + 1, nres):
                d = distance_array(res_list[i].positions, res_list[j].positions, box=boxa)
                test = np.less(d, [self.cutoff])
                if np.sum(test) >= 1:
                    bond[i, j] = 1
                    bond[j, i] = 1

        clus = self.calc_cluster(cluster, nres, bond)

        hist = np.histogram(clus, bins=res)
        index_bc = np.argmax(hist[0])  # get the index of the biggest cluster.
        cluster = res_list[index_bc]

        for i in range(nres):  # look for the proteins in the biggest cluster
            if clus[i] == index_bc:
                if i != index_bc:
                    cluster = cluster.union(res_list[i])

        member_idx = np.where(clus == index_bc)[0].astype(np.int32)
        return cluster, clus, member_idx

    @staticmethod
    def _minimum_image(delta, box):
        """Apply the orthorhombic minimum-image convention to a displacement vector ``delta`` given box lengths ``box``."""
        boxv = np.asarray(box[:3], dtype=float)
        if boxv.shape[0] != 3:
            raise ValueError("box must provide at least 3 lengths")
        return delta - boxv * np.round(delta / boxv)

    @staticmethod
    def _rg_and_com(positions, masses):
        """Return the mass-weighted radius of gyration and center of mass for ``positions``/``masses`` (NaN if empty)."""
        pos = np.asarray(positions, dtype=float)
        mass = np.asarray(masses, dtype=float)
        if pos.size == 0 or mass.size == 0:
            return math.nan, np.full(3, math.nan, dtype=float)
        com = np.average(pos, axis=0, weights=mass)
        rg2 = np.average(np.sum((pos - com) ** 2, axis=1), weights=mass)
        return float(np.sqrt(max(rg2, 0.0))), com

    def _get_compact_cluster_geometry(self, frame):
        """Reconstruct (and cache) the unwrapped, PBC-stitched cluster geometry for ``frame`` via BFS over the bond graph; return res groups, per-chain shifts, COM, and Rg."""
        cached = self._compact_cluster_cache.get(frame)
        if cached is not None:
            return cached

        member_idx = np.asarray(self.cluster_members.get(frame, []), dtype=int)
        if member_idx.size == 0:
            geom = {
                "res_groups": [],
                "shifts": np.empty((0, 3), dtype=float),
                "cluster_com": np.full(3, math.nan, dtype=float),
                "cluster_rg": math.nan,
            }
            self._compact_cluster_cache[frame] = geom
            return geom

        res_groups = [self.sg_residue_groups[idx] for idx in member_idx]
        bond = self.cluster_bond.get(frame)
        if bond is None:
            sub_bond = np.zeros((len(res_groups), len(res_groups)), dtype=np.uint8)
        else:
            sub_bond = np.asarray(bond[np.ix_(member_idx, member_idx)], dtype=np.uint8)

        com = np.array([ag.center_of_mass() for ag in res_groups], dtype=float)
        shifts = np.zeros_like(com)
        placed = np.zeros(len(res_groups), dtype=bool)
        queue = []

        for seed in range(len(res_groups)):
            if placed[seed]:
                continue
            placed[seed] = True
            queue.append(seed)
            while queue:
                i = queue.pop(0)
                nbrs = np.where(sub_bond[i] > 0)[0]
                for j in nbrs:
                    if placed[j]:
                        continue
                    delta = self._minimum_image(com[j] - com[i], self.u.trajectory.ts.dimensions)
                    shifts[j] = shifts[i] + (delta - (com[j] - com[i]))
                    placed[j] = True
                    queue.append(j)

        cluster_pos = np.concatenate(
            [ag.positions + shifts[i] for i, ag in enumerate(res_groups)], axis=0
        )
        cluster_mass = np.concatenate([ag.masses for ag in res_groups], axis=0)
        cluster_rg, cluster_com = self._rg_and_com(cluster_pos, cluster_mass)
        geom = {
            "res_groups": res_groups,
            "shifts": shifts,
            "cluster_com": cluster_com,
            "cluster_rg": cluster_rg,
        }
        self._compact_cluster_cache[frame] = geom
        return geom

    def _get_compact_selected_positions(self, frame, select):
        """Return PBC-stitched positions, masses, cluster COM, selection Rg, and residue count for atoms matching ``select`` within the compact cluster at ``frame``."""
        geom = self._get_compact_cluster_geometry(frame)
        pos_chunks = []
        mass_chunks = []
        n_residues = 0
        for ag, shift in zip(geom["res_groups"], geom["shifts"]):
            sel = ag.select_atoms(select)
            if len(sel) == 0:
                continue
            pos_chunks.append(sel.positions + shift)
            mass_chunks.append(sel.masses)
            n_residues += sel.residues.n_residues

        if not pos_chunks:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), geom["cluster_com"], math.nan, 0

        positions = np.concatenate(pos_chunks, axis=0)
        masses = np.concatenate(mass_chunks, axis=0)
        selected_rg, _ = self._rg_and_com(positions, masses)
        return positions, masses, geom["cluster_com"], selected_rg, n_residues

    def _get_historical_selected_positions(self, frame, select):
        """Return the legacy (transform-based) positions, masses, cluster COM, selection Rg, and residue count for atoms matching ``select`` at ``frame``."""
        cluster_i = self.cluster_group.get(frame)
        if cluster_i is None or len(cluster_i) == 0:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), np.full(3, math.nan, dtype=float), math.nan, 0

        if select == "all":
            sel = cluster_i
        else:
            sel = cluster_i.select_atoms(select)
        if len(sel) == 0:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), cluster_i.center_of_mass(), math.nan, 0

        positions = np.asarray(sel.positions, dtype=float).copy()
        masses = np.asarray(sel.masses, dtype=float).copy()
        center = np.asarray(cluster_i.center_of_mass(), dtype=float)
        rg = float(sel.radius_of_gyration()) if len(sel) > 1 else math.nan
        n_residues = sel.residues.n_residues
        return positions, masses, center, rg, n_residues

    def _get_selected_positions(self, frame, select):
        """Dispatch to the historical or compact selected-position helper according to ``self.structure_coordinate_mode``."""
        if self.structure_coordinate_mode == "historical":
            return self._get_historical_selected_positions(frame, select)
        return self._get_compact_selected_positions(frame, select)

    def _get_cluster_center_for_sm(self, frame):
        """Return the condensate center of mass at ``frame`` (used as the origin for small-molecule density profiles), per the active coordinate mode."""
        if self.structure_coordinate_mode == "historical":
            cluster_i = self.cluster_group.get(frame)
            if cluster_i is None or len(cluster_i) == 0:
                return np.full(3, math.nan, dtype=float)
            return np.asarray(cluster_i.center_of_mass(), dtype=float)
        return np.asarray(self._get_compact_cluster_geometry(frame)["cluster_com"], dtype=float)





    def _frame_cluster_resid_sets(self, frame):
        """Return the connected clusters at ``frame`` as resid sets, sorted largest-first (then by smallest resid)."""
        labels = np.asarray(self.cluster_array.get(frame, []), dtype=np.int32)
        if labels.size == 0:
            return []

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), set()).add(int(self.sg_resids[idx]))

        return sorted(
            clusters.values(),
            key=lambda resids: (-len(resids), min(resids) if resids else math.inf),
        )

    @staticmethod
    def _select_overlap_tracked_cluster(previous_resids, frame_clusters):
        """Pick the cluster in ``frame_clusters`` that best matches ``previous_resids`` (by overlap then Jaccard) to track the condensate continuously across frames."""
        if not frame_clusters:
            return set()
        if not previous_resids:
            return set(frame_clusters[0])

        prev = set(previous_resids)
        prev_size = len(prev)

        def _score(cluster_resids):
            """Rank a candidate cluster vs ``prev`` by overlap, Jaccard, then size proximity."""
            overlap = len(prev.intersection(cluster_resids))
            union = len(prev.union(cluster_resids))
            jaccard = overlap / union if union else 0.0
            size_delta = -abs(len(cluster_resids) - prev_size)
            tie_break = -min(cluster_resids) if cluster_resids else 0
            return overlap, jaccard, size_delta, len(cluster_resids), tie_break

        best = max(frame_clusters, key=_score)
        return set(best)

    @staticmethod
    def _persistent_resids_from_tracked_clusters(tracked_clusters, occupancy_fraction):
        """Return the resids present in at least ``occupancy_fraction`` of the per-frame tracked clusters (the persistent condensate membership)."""
        if not tracked_clusters:
            return []

        counts = Counter()
        for cluster_resids in tracked_clusters:
            counts.update(int(resid) for resid in cluster_resids)

        threshold = max(1, int(math.ceil(float(occupancy_fraction) * len(tracked_clusters))))
        return sorted(
            resid for resid, count in counts.items()
            if count >= threshold
        )

    # Determine Maximal Continuous Cluster
    def gen_maximal_continuous_cluster(self):
        """Track the condensate across frames by maximal overlap, keep resids present in >=95% of frames, write the cluster mask, and return the persistent-cluster AtomGroup."""
        frame_keys = sorted(self.cluster_group.keys())
        if not frame_keys:
            res_ids = []
            self._tracked_frame_keys = []
            self._tracked_clusters = []
        else:
            # Track the same SG droplet across frames by maximal overlap, then keep
            # residues present in at least 95% of tracked frames. This avoids
            # zeroing the cluster when transient split/merge events change which
            # cluster happens to be largest in a small number of frames.
            tracked_clusters = []
            current_resids = self._select_overlap_tracked_cluster(
                set(),
                self._frame_cluster_resid_sets(frame_keys[0]),
            )
            tracked_clusters.append(current_resids)

            for frame in frame_keys[1:]:
                frame_clusters = self._frame_cluster_resid_sets(frame)
                current_resids = self._select_overlap_tracked_cluster(current_resids, frame_clusters)
                tracked_clusters.append(current_resids)

            res_ids = self._persistent_resids_from_tracked_clusters(
                tracked_clusters,
                self.max_cluster_occupancy_fraction,
            )

            # Store tracked cluster data for downstream consumers (stress, diffusion)
            self._tracked_frame_keys = list(frame_keys)
            self._tracked_clusters = tracked_clusters

        select = " or ".join(f"resid {resid}" for resid in res_ids)
        self.max_selection = select

        out_path = "{}Max_Continuous_Cluster_{}_{}.txt".format(
            self.folder, self.system_name, self.system_num
        )
        file_text = select if select else "# EMPTY_MAX_CONTINUOUS_CLUSTER\n"
        with open(out_path, "w", encoding="utf-8") as fid:
            fid.write(file_text)

        if not res_ids:
            return self.u.atoms[:0]
        return self.u.select_atoms(f"({select}) and (resname Protein* or resname RNA)")

    def _write_tracked_cluster_sidecar(self):
        """Write per-frame tracked cluster membership to NPZ sidecar.

        Output: ``Tracked_Cluster_{system}_{window}.npz`` containing:
            frame_keys   : int array  — trajectory frame indices
            time_ns      : float array — frame times in ns
            sg_resids    : int array  — all SG biopolymer resids (row axis of inside_mask)
            inside_mask  : bool array (n_frames x n_sg_resids) — per-frame membership
            persistent_resids : int array — resids passing 95% occupancy threshold
        """
        frame_keys = getattr(self, "_tracked_frame_keys", [])
        tracked = getattr(self, "_tracked_clusters", [])
        if not frame_keys or not tracked:
            return

        sg_resids = self.sg_resids  # int32 array of all biopolymer resids
        resid_to_col = {int(r): j for j, r in enumerate(sg_resids)}
        n_frames = len(frame_keys)
        n_res = len(sg_resids)

        inside_mask = np.zeros((n_frames, n_res), dtype=bool)
        for i, cluster_resids in enumerate(tracked):
            for resid in cluster_resids:
                col = resid_to_col.get(int(resid))
                if col is not None:
                    inside_mask[i, col] = True

        # Frame times: frame_key * frame_dt_ns
        frame_dt_ns = getattr(self, "frame_dt_s", 200000e-15) * 1e9
        time_ns = np.array([fk * frame_dt_ns for fk in frame_keys], dtype=float)

        # Persistent resids from the max_selection (already computed)
        persistent = np.array(
            sorted(int(r) for r in re.findall(r"resid\s+(\d+)", self.max_selection)),
            dtype=np.int32,
        )

        out_path = "{}Tracked_Cluster_{}_{}.npz".format(
            self.folder, self.system_name, self.system_num
        )
        np.savez_compressed(
            out_path,
            frame_keys=np.asarray(frame_keys, dtype=np.int64),
            time_ns=time_ns,
            sg_resids=sg_resids,
            inside_mask=inside_mask,
            persistent_resids=persistent,
        )
        print(f"Tracked cluster sidecar written to: {out_path}", flush=True)

    # Generate Density Profile of Cluster
    def calc_rdp(self, res_type):
        """Compute and write the time-averaged radial density profile (mg/mL vs distance from cluster COM) for the species/selection ``res_type``."""
        if res_type == "SG":
            file_name = "SG_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname Protein* or resname RNA"
        elif res_type == "Protein":
            file_name = "Protein_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname Protein*"
        elif res_type == "RNA":
            file_name = "RNA_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname RNA"
        elif res_type == "ADENINE":
            file_name = "ADENINE_{}_{}.csv".format(self.system_name, self.system_num)
            select = "type 21"
        elif res_type == "UCG":
            file_name = "UCG_{}_{}.csv".format(self.system_name, self.system_num)
            select = "type 22 or type 23 or type 24"
        elif res_type == "G3BP1":
            file_name = "G3BP1_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinG3BP1"
        elif res_type == "TDP43":
            file_name = "TDP43_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinTDP43"
        elif res_type == "PABP1":
            file_name = "PABP1_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinPABP1"
        elif res_type == "FUS":
            file_name = "FUS_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinFUS"
        elif res_type == "TIA1":
            file_name = "TIA1_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinTIA1"
        elif res_type == "TTP":
            file_name = "TTP_{}_{}.csv".format(self.system_name, self.system_num)
            select = "resname ProteinTTP"
        else:
            raise ValueError(
                "Unsupported res_type '{}' for calc_rdp. Expected one of: SG, Protein, RNA, ADENINE, UCG, G3BP1, TDP43, PABP1, FUS, TIA1, TTP".format(
                    res_type
                )
            )
        # Build radial bins up to half the smallest box length
        box0 = self.u.trajectory[self.tmin].dimensions
        max_r = 0.5 * float(np.min(box0[:3]))
        r = np.arange(0.0, max_r + self.bin_size, self.bin_size)
        nr = len(r)

        density = []
        vol = np.zeros(nr - 1)
        for i in range(nr - 1):
            vol[i] = (4 / 3) * np.pi * (r[i + 1] ** 3 - r[i] ** 3)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            frame = ts.frame
            # physical time in microseconds for labeling
            timestep = frame * self.frame_dt_s * 1e6
            boxa = ts.dimensions
            cluster_i = self.cluster_group[frame]
            cluster_array_i = self.cluster_array[frame]
            sel_pos, sel_mass, cluster_com, sel_rg, nres = self._get_selected_positions(frame, select)
            if len(cluster_i.residues) >= 2 and sel_pos.shape[0] > 0 and nres > 0:
                if self.structure_coordinate_mode == "historical":
                    R = distance_array(
                        np.atleast_2d(cluster_com).astype(np.float32),
                        sel_pos,
                        box=boxa,
                    )
                    radii = R[0]
                else:
                    radii = np.linalg.norm(sel_pos - cluster_com[None, :], axis=1)
                h = np.histogram(radii, bins=r, weights=sel_mass)
                density.append((h[0] / vol) * 1660.5390666)
                try:
                    self.cluster_num[timestep] = len(np.unique(cluster_array_i))
                    self.cluster_rg[timestep] = sel_rg
                    self.num_inner_chains[timestep] = self.get_inner_chain_num(cluster_i)[res_type]
                    self.mass_inner_chains[timestep] = self.get_inner_chain_mass(cluster_i)[res_type]
                    self.num_outer_chains[timestep] = self.get_outer_chain_num(cluster_i)[res_type]
                    self.mass_outer_chains[timestep] = self.get_outer_chain_mass(cluster_i)[res_type]
                except:
                    pass
            else:
                h = np.histogram(1000, bins=r)
                density.append(h[0])

        density_avg = np.zeros(nr - 1)
        density_std = np.zeros(nr - 1)  # standard deviation
        density_se = np.zeros(nr - 1)  # standard error
        le = len(density)

        if len(cluster_i.residues) > 1:
            row_list = [
                ['Timestep', 'Total Mass (mg)', 'Total Chain Number', 'Largest Droplet Radius of Gyration',
                'Number of Droplets', 'Chains in Largest Droplet', 'Mass of Largest Droplet (mg)',
                'Number of External Chains', 'Mass of External Chains']]

            with open("{}Cluster_{}".format(self.folder, file_name), "w") as file:
                writer = csv.writer(file)
                for key in self.cluster_num.keys():
                    row_list.append(
                        [key, self.mass_total_chains[res_type], self.num_total_chains[res_type], self.cluster_rg[key],
                        self.cluster_num[key], self.num_inner_chains[key], self.mass_inner_chains[key],
                        self.num_outer_chains[key], self.mass_outer_chains[key]])
                writer.writerows(row_list)

            for i in range(nr - 1):
                s = 0
                sig = 0
                for ts in range(le):
                    s += density[ts][i]
                density_avg[i] = s / le
                for ts in range(le):
                    sig += (density[ts][i] - density_avg[i]) ** 2
                if le > 1:
                    density_std[i] = (sig / (le - 1)) ** (0.5)
                    density_se[i] = density_std[i] / (le ** 0.5)
                else:
                    density_std[i] = 0
                    density_se[i] = 0
            rplot = 0.5 * (r[1:] + r[:-1])

            row_list = [['Distance from center of mass (A)', 'Protein density (mg/mL)', 'Standard deviation',
                         'Standard mean error']]

            for i in range(len(density_avg)):
                el = [rplot[i], density_avg[i], density_std[i], density_se[i]]
                row_list.append(el)
            with open("{}DensityProfile_{}".format(self.folder, file_name), "w") as file:
                writer = csv.writer(file)
                writer.writerows(row_list)




    # PCA Analysis for Surface Fluctuations
    def calc_pca(self, res_type):
        """Compute and write per-frame PCA shape descriptors (principal eigenvalues / ellipsoid axes) of the ``res_type`` cluster, used for surface-tension shape fluctuations."""
        # Time in microseconds for each stored PCA sample
        frames = np.arange(self.tmin, self.tmax, self.dt)
        Time = frames * self.frame_dt_s * 1e6
        nt = len(frames)
        lambda1 = np.zeros(nt)  # principal value  (related to the length of the principal axes of the ellipsoid)
        lambda2 = np.zeros(nt)
        lambda3 = np.zeros(nt)
        ax1, ax2, ax3 = np.zeros((nt, 3)), np.zeros((nt, 3)), np.zeros((nt, 3))  # principal axis
        ts = 0
        for t in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            clus_atoms = self.cluster_group[t.frame]
            if res_type == "SG":
                file_name = "{}PCA_SG_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "all"
            elif res_type == "Protein":
                file_name = "{}PCA_Protein_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname Protein*"
            elif res_type == "RNA":
                file_name = "{}PCA_RNA_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname RNA"
            elif res_type == "G3BP1":
                file_name = "{}PCA_G3BP1_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinG3BP1"
            elif res_type == "TDP43":
                file_name = "{}PCA_TDP43_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinTDP43"
            elif res_type == "FUS":
                file_name = "{}PCA_FUS_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinFUS"
            elif res_type == "PABP1":
                file_name = "{}PCA_PABP1_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinPABP1"
            elif res_type == "TIA1":
                file_name = "{}PCA_TIA1_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinTIA1"
            elif res_type == "TTP":
                file_name = "{}PCA_TTP_{}_{}.csv".format(self.folder, self.system_name, self.system_num)
                select = "resname ProteinTTP"
            else:
                raise ValueError(
                    "Unsupported res_type '{}' for calc_pca. Expected one of: SG, Protein, RNA, G3BP1, TDP43, PABP1, FUS, TIA1, TTP".format(
                        res_type
                    )
                )

            test_pos, _, cluster_com, _, _ = self._get_selected_positions(t.frame, select)
            if test_pos.shape[0] < 3:
                ts += 1
                continue
            pos = test_pos - cluster_com[None, :]
            X = np.tensordot(pos, [1, 0, 0], axes=1)
            Y = np.tensordot(pos, [0, 1, 0], axes=1)
            Z = np.tensordot(pos, [0, 0, 1], axes=1)
            data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
            features = data.columns[0:3]
            x = data.loc[:, features].values
            pca_dat = sklearn.decomposition.PCA(n_components=3)
            principalComponents_dat = pca_dat.fit_transform(x)
            lambda1[ts] = pca_dat.explained_variance_[0]
            lambda2[ts] = pca_dat.explained_variance_[1]
            lambda3[ts] = pca_dat.explained_variance_[2]
            ax1[ts] = pca_dat.components_[0]
            ax2[ts] = pca_dat.components_[1]
            ax3[ts] = pca_dat.components_[2]
            ts += 1

        row_list = [
            ['Time (us)', 'l1', 'l2', 'l3', 'ax1x', 'ax1y', 'ax1z', 'ax2x', 'ax2y', 'ax2z', 'ax3x', 'ax3y', 'ax3z']]
        for i in range(nt):
            el = [Time[i], lambda1[i], lambda2[i], lambda3[i], ax1[i][0], ax1[i][1], ax1[i][2], ax2[i][0], ax2[i][1],
                  ax2[i][2], ax3[i][0], ax3[i][1], ax3[i][2]]
            row_list.append(el)
        with open(file_name, 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)




    # Generate Small Molecule Density Profile
    def calc_rdp_sm(self):
        """Compute and write the time-averaged small-molecule radial density profile (mg/mL) about the condensate center of mass."""
        # Build radial bins up to half the smallest box length
        box0 = self.u.trajectory[self.tmin].dimensions
        max_r = 0.5 * float(np.min(box0[:3]))
        r = np.arange(0.0, max_r + self.bin_size, self.bin_size)
        nr = len(r)

        density = []
        vol = np.zeros(nr - 1)
        for i in range(nr - 1):
            vol[i] = (4 / 3) * np.pi * (r[i + 1] ** 3 - r[i] ** 3)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            frame = ts.frame
            boxa = ts.dimensions
            sm_atoms = self.u.select_atoms("resname SM")

            boxv = np.asarray(boxa[:3], dtype=float)
            center_ref = np.mod(self._get_cluster_center_for_sm(frame), boxv)
            delta = sm_atoms.positions - center_ref[None, :]
            delta -= boxv * np.round(delta / boxv)
            radii = np.linalg.norm(delta, axis=1)
            h = np.histogram(radii, bins=r, weights=sm_atoms.masses)
            density.append((h[0] / vol) * 1660.5390666)

        density_avg = np.zeros(nr - 1)
        density_std = np.zeros(nr - 1)  # standard deviation
        density_se = np.zeros(nr - 1)  # standard error
        le = len(density)

        if le > 0:
            for i in range(nr - 1):
                s = 0
                sig = 0
                for ts in range(le):
                    s += density[ts][i]
                density_avg[i] = s / le
                for ts in range(le):
                    sig += (density[ts][i] - density_avg[i]) ** 2
                if le > 1:
                    density_std[i] = (sig / (le - 1)) ** (0.5)
                    density_se[i] = density_std[i] / (le ** 0.5)
                else:
                    density_std[i] = 0
                    density_se[i] = 0
            rplot = 0.5 * (r[1:] + r[:-1])

            row_list = [['Distance from center of mass (A)', 'SM density (mg/mL)', 'Standard deviation',
                         'Standard mean error']]

            for i in range(len(density_avg)):
                el = [rplot[i], density_avg[i], density_std[i], density_se[i]]
                row_list.append(el)
            with open("{}DensityProfile_SM_{}_{}.csv".format(self.folder, self.system_name, self.system_num), "w") as file:
                writer = csv.writer(file)
                writer.writerows(row_list)

    # 1) DOMAIN CONTACTS (species × species, per-atom matrix averaged over time)
    def gen_domain_contacts(self, radius):
        """
        Compute species–species contact maps as LA×LB sequence-index matrices averaged over time,
        restricted to the largest SG cluster each frame. Uses capped distances for speed and
        maps global atom indices to per-sequence indices (0..L-1) for each species.
        """
        species = [
            ("ProteinG3BP1", 466),
            ("ProteinTDP43", 414),
            ("ProteinFUS",   526),
            ("ProteinPABP1", 636),
            ("ProteinTIA1",  386),
            ("ProteinTTP",   326),
            ("RNA",           840),
        ]
        name_to_len = {n: L for (n, L) in species}

        for i in range(len(species)):
            for j in range(i, len(species)):
                A, LA = species[i]
                B, LB = species[j]
                M = np.zeros((LA, LB), dtype=np.uint64)
                M_full = np.zeros((LA, LB), dtype=np.uint64)
                M_inter = np.zeros((LA, LB), dtype=np.uint64)
                M_intra_filtered = np.zeros((LA, LB), dtype=np.uint64)
                M_combined_filtered = np.zeros((LA, LB), dtype=np.uint64)
                nt = 0

                def _add_symmetric_contacts(mat, seq_a, seq_b):
                    np.add.at(mat, (seq_a, seq_b), 1)
                    offdiag = seq_a != seq_b
                    if np.any(offdiag):
                        np.add.at(mat, (seq_b[offdiag], seq_a[offdiag]), 1)

                for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
                    cluster_atoms = self.cluster_group[ts.frame]
                    agA = cluster_atoms.select_atoms(f"resname {A}")
                    agB = cluster_atoms.select_atoms(f"resname {B}")
                    if len(agA) == 0 or len(agB) == 0:
                        nt += 1
                        continue

                    # Build Universe-index -> sequence-index maps for both species.
                    # For same-species contacts also track the parent biopolymer chain,
                    # so local intrachain neighbors can be removed without removing
                    # same-index contacts between different chains.
                    g2sA = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
                    g2rA = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
                    for res in agA.residues:
                        order = np.argsort(res.atoms.indices)
                        Lc = len(order)
                        # Use actual residue length if it deviates, else LA
                        Lmap = Lc if Lc != LA else LA
                        atom_idx = np.asarray(res.atoms.indices)[order]
                        g2sA[atom_idx] = np.arange(Lmap, dtype=np.int32)
                        g2rA[atom_idx] = int(res.ix)
                    if A == B:
                        g2sB = g2sA
                        g2rB = g2rA
                    else:
                        g2sB = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
                        g2rB = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
                        for res in agB.residues:
                            order = np.argsort(res.atoms.indices)
                            Lc = len(order)
                            Lmap = Lc if Lc != LB else LB
                            atom_idx = np.asarray(res.atoms.indices)[order]
                            g2sB[atom_idx] = np.arange(Lmap, dtype=np.int32)
                            g2rB[atom_idx] = int(res.ix)

                    if A != B:
                        ia, ib = _capped_indices(agA.positions, agB.positions, radius, ts.dimensions)
                        if ia.size:
                            # Map local vs global indices robustly
                            if (ia.max(initial=-1) >= len(agA)) or (ib.max(initial=-1) >= len(agB)):
                                globA = ia
                                globB = ib
                            else:
                                globA = agA.indices[ia]
                                globB = agB.indices[ib]
                            sA = g2sA[globA]
                            sB = g2sB[globB]
                            good = (sA >= 0) & (sA < LA) & (sB >= 0) & (sB < LB)
                            if np.any(good):
                                np.add.at(M, (sA[good], sB[good]), 1)
                                np.add.at(M_full, (sA[good], sB[good]), 1)
                                np.add.at(M_inter, (sA[good], sB[good]), 1)
                                np.add.at(M_combined_filtered, (sA[good], sB[good]), 1)
                    else:
                        # Use wrapper to consistently return only (ia, ib)
                        ia, ib = _self_capped_pairs(agA.positions, radius, ts.dimensions)
                        if ia.size:
                            # Indices from _self_capped_pairs are local to agA
                            globA = agA.indices[ia]
                            globB = agA.indices[ib]
                            sA = g2sA[globA]
                            sB = g2sA[globB]
                            rA = g2rA[globA]
                            rB = g2rB[globB]
                            good_full = (
                                (sA >= 0) & (sA < LA) &
                                (sB >= 0) & (sB < LB) &
                                (rA >= 0) & (rB >= 0)
                            )
                            if np.any(good_full):
                                sA_good = sA[good_full]
                                sB_good = sB[good_full]
                                rA_good = rA[good_full]
                                rB_good = rB[good_full]

                                # All non-self atom contacts, retained for backward-compatible
                                # diagnostics. self_capped_distance excludes atom i with itself.
                                _add_symmetric_contacts(M_full, sA_good, sB_good)

                                same_chain = rA_good == rB_good
                                local_chain_neighbor = np.abs(sA_good - sB_good) <= DOMAIN_INTRACHAIN_EXCLUSION
                                inter_mask = ~same_chain
                                intra_mask = same_chain & (~local_chain_neighbor)

                                if np.any(inter_mask):
                                    _add_symmetric_contacts(M_inter, sA_good[inter_mask], sB_good[inter_mask])
                                    _add_symmetric_contacts(M_combined_filtered, sA_good[inter_mask], sB_good[inter_mask])
                                if np.any(intra_mask):
                                    _add_symmetric_contacts(M_intra_filtered, sA_good[intra_mask], sB_good[intra_mask])
                                    _add_symmetric_contacts(M_combined_filtered, sA_good[intra_mask], sB_good[intra_mask])

                                # Default/legacy output now matches the scientifically preferred
                                # combined map: interchain plus intrachain with local bonded
                                # neighbors excluded, while preserving interchain same-index
                                # contacts on the diagonal.
                                _add_symmetric_contacts(M, sA_good[inter_mask | intra_mask], sB_good[inter_mask | intra_mask])
                    nt += 1

                def _save_domain_matrix(prefix, matrix):
                    C = (matrix / float(nt)).astype(np.float32) if nt else matrix.astype(np.float32)
                    np.savez_compressed(
                        f"{self.folder}{prefix}{self.system_name}_{A}_{B}_{self.system_num}.csv",
                        C,
                    )

                _save_domain_matrix("Domain_Contacts_Total_", M)
                _save_domain_matrix("Domain_Contacts_TotalCombinedFiltered_", M_combined_filtered)
                _save_domain_matrix("Domain_Contacts_TotalInter_", M_inter)
                if A == B:
                    _save_domain_matrix("Domain_Contacts_TotalIntraFiltered_", M_intra_filtered)
                _save_domain_matrix("Domain_Contacts_TotalFull_", M_full)

    # 2) SM–DOMAIN CONTACTS (species vs all SM atoms, per-atom matrix averaged over time)
    def gen_sm_domain_contacts(self, radius):
        """
        Compute species–SM contact profiles as LA×1 vectors using SM COMs, averaged over time.
        Small molecules are taken from the full simulation box (not only the droplet), while
        proteins/RNA remain restricted to the largest SG cluster each frame.
        """
        species = [
            ("ProteinG3BP1", 466),
            ("ProteinTDP43", 414),
            ("ProteinFUS",   526),
            ("ProteinPABP1", 636),
            ("ProteinTIA1",  386),
            ("ProteinTTP",   326),
            ("RNA",           840),
        ]

        # Use all SM atoms in the box; if none exist, emit zeros and exit early
        sm_all = self.u.select_atoms("resname SM")
        if sm_all.residues.n_residues == 0:
            for A, LA in species:
                np.savetxt(
                    f"{self.folder}SM_Domain_Contacts_Total_{self.system_name}_{A}_{self.system_num}.csv",
                    np.zeros((LA, 1)), delimiter=","
                )
            return

        for A, LA in species:
            v = np.zeros(LA, dtype=np.uint64)
            nt = 0
            for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
                agA = self.u.select_atoms(f"resname {A}")
                if len(agA) == 0:
                    nt += 1
                    continue
                # Universe-index → sequence-index for species A
                g2sA = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
                for res in agA.residues:
                    order = np.argsort(res.atoms.indices)
                    Lc = len(order)
                    Lmap = Lc if Lc != LA else LA
                    g2sA[np.asarray(res.atoms.indices)[order]] = np.arange(Lmap, dtype=np.int32)

                sm_com = sm_all.center_of_mass(compound='residues')
                if sm_com.size == 0:
                    nt += 1
                    continue
                ia, ib = _capped_indices(sm_com, agA.positions, radius, ts.dimensions)
                if ib.size:
                    if ib.max(initial=-1) >= len(agA):
                        globA = ib
                    else:
                        globA = agA.indices[ib]
                    sA = g2sA[globA]
                    mask = (sA >= 0) & (sA < LA)
                    if np.any(mask):
                        np.add.at(v, sA[mask], 1)
                nt += 1

            C = (v / float(nt)) if nt else v.astype(float)
            # Save as LA×1 for matrix shape consistency
            np.savetxt(
                f"{self.folder}SM_Domain_Contacts_Total_{self.system_name}_{A}_{self.system_num}.csv",
                C.reshape(-1, 1), delimiter=","
            )

    def gen_residue_contacts(self, radius):
        """
        Vectorized residue–residue contact analysis.
        Counts one contact per *unordered residue pair* per frame (if any atom–atom pair < radius),
        then records it twice (i->j and j->i) to match legacy outputs.
        Produces the same files:
        - Residue_Contacts_Total_<system>_<tmin>.csv
        - Residue_Contacts_Count_<system>_<tmin>.csv
        - Residue_Contacts_Mean_<system>_<tmin>.csv
        """
        contact_dict = {
            "ProteinG3BP1": 0,
            "ProteinPABP1": 1,
            "ProteinTTP":   2,
            "ProteinTIA1":  3,
            "ProteinTDP43": 4,
            "ProteinFUS":   5,
            "RNA":          6,
        }
        n = len(contact_dict)
        contact_array = np.zeros((n, n), dtype=np.float64)
        total_contacts = 0  # legacy: counts directed i->j + j->i
        nt = 0

        # speed: build a dtype for vectorized pair-uniqueing
        def _unique_unordered_pairs(a, b):
            """Return unique unordered pairs from arrays a,b."""
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            pairs = np.stack((lo, hi), axis=1)
            if pairs.size == 0:
                return pairs
            # use np.unique on rows
            return np.unique(pairs, axis=0)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            ag = self.cluster_group[ts.frame]
            if len(ag) == 0:
                nt += 1
                continue

            # one neighbor search over ALL atoms in the cluster
            ia, ib = _self_capped_pairs(ag.positions, radius, ts.dimensions)
            if ia.size == 0:
                nt += 1
                continue

            # map atoms -> their residue ids (local to Universe, not contiguous)
            atom_resids = ag.atoms.resids
            ra = atom_resids[ia]
            rb = atom_resids[ib]

            # discard same-residue pairs (legacy behavior)
            mask_diff = (ra != rb)
            if not np.any(mask_diff):
                nt += 1
                continue
            ra = ra[mask_diff]
            rb = rb[mask_diff]

            # build resid -> species bucket just once per frame (from residues present in the cluster)
            res_resids = ag.residues.resids
            res_names  = ag.residues.resnames
            resid2bucket = {rid: contact_dict.get(rn, -1) for rid, rn in zip(res_resids, res_names)}

            # filter to residue pairs where both residues are in our 7 species
            sa = np.fromiter((resid2bucket.get(r, -1) for r in ra), dtype=np.int16, count=ra.size)
            sb = np.fromiter((resid2bucket.get(r, -1) for r in rb), dtype=np.int16, count=rb.size)
            valid = (sa >= 0) & (sb >= 0)
            if not np.any(valid):
                nt += 1
                continue
            # Keep only residue pairs whose endpoints are both in the 7 species;
            # buckets (sa/sb) are recomputed from the unique pairs below.
            ra = ra[valid]; rb = rb[valid]

            # reduce atom-level contacts to unique unordered *residue* pairs
            unique_pairs = _unique_unordered_pairs(ra, rb)
            if unique_pairs.size == 0:
                nt += 1
                continue

            # map those residue ids -> species buckets again (vectorized)
            s1 = np.fromiter((resid2bucket.get(r, -1) for r in unique_pairs[:, 0]), dtype=np.int16,
                            count=unique_pairs.shape[0])
            s2 = np.fromiter((resid2bucket.get(r, -1) for r in unique_pairs[:, 1]), dtype=np.int16,
                            count=unique_pairs.shape[0])
            good = (s1 >= 0) & (s2 >= 0)
            if np.any(good):
                s1 = s1[good].astype(np.intp, copy=False)
                s2 = s2[good].astype(np.intp, copy=False)
                # Directed counting: each unordered residue pair (i,j) adds +1
                # to both cell (s1,s2) and (s2,s1), keeping the 7×7 matrix
                # symmetric.  total_contacts is doubled accordingly so that
                # contacts_mean = contact_array / total_contacts gives correct
                # fractions (the 2× cancels in numerator and denominator).
                np.add.at(contact_array, (s1, s2), 1.0)
                np.add.at(contact_array, (s2, s1), 1.0)
                total_contacts += 2 * s1.size
            nt += 1

        # write legacy outputs
        np.savetxt(f"{self.folder}Residue_Contacts_Total_{self.system_name}_{self.system_num}.csv",
                np.array([total_contacts / float(nt)]) if nt else np.array([0.0]), delimiter=",")
        contacts_count = (contact_array / float(nt)) if nt else contact_array
        np.savetxt(f"{self.folder}Residue_Contacts_Count_{self.system_name}_{self.system_num}.csv",
                contacts_count, delimiter=",")
        contacts_mean = (contact_array / float(total_contacts)) if total_contacts > 0 else contacts_count
        np.savetxt(f"{self.folder}Residue_Contacts_Mean_{self.system_name}_{self.system_num}.csv",
                contacts_mean, delimiter=",")


    def gen_acid_contacts(self, radius):
        """
        Vectorized amino/nucleic acid type contact analysis (24x24).
        Performs a single self-capped neighbor search on all acid atoms per frame, then bins by type.
        Matches legacy normalization & mirroring.
        """
        acid_types = list(range(1, 25))  # 1..24
        n = len(acid_types)
        type_to_idx = {t: (t - 1) for t in acid_types}

        M = np.zeros((n, n), dtype=np.uint64)  # accumulate *upper triangle*; mirror after loop
        nt = 0

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            cluster = self.cluster_group[ts.frame]
            # select all acid atoms (types 1..24) within cluster once
            acid_sel = " or ".join(f"type {t}" for t in acid_types)
            A = cluster.select_atoms(acid_sel)
            if len(A) < 2:
                nt += 1
                continue

            ia, ib = _self_capped_pairs(A.positions, radius, ts.dimensions)
            if ia.size == 0:
                nt += 1
                continue

            # per-atom numeric types
            # MDAnalysis types can be strings; make sure we have ints
            A_types = np.asarray([int(t) for t in A.types], dtype=np.int32)

            # convert to (upper-tri) type-pair bin indices
            ta = A_types[ia]
            tb = A_types[ib]
            # keep only those within our dictionary (should already be true)
            mask = np.isin(ta, acid_types) & np.isin(tb, acid_types)
            if not np.any(mask):
                nt += 1
                continue
            ta = ta[mask]
            tb = tb[mask]
            i = np.minimum(ta, tb)
            j = np.maximum(ta, tb)
            # map type numbers -> [0..23] indices
            i = np.vectorize(type_to_idx.get, otypes=[np.int32])(i).astype(np.intp, copy=False)
            j = np.vectorize(type_to_idx.get, otypes=[np.int32])(j).astype(np.intp, copy=False)
            np.add.at(M, (i, j), 1)
            nt += 1

        # mirror lower triangle to match legacy post-processing
        for i in range(n):
            for j in range(i):
                M[i, j] = M[j, i]

        # outputs (identical semantics to legacy)
        np.savetxt(f"{self.folder}Acid_Contacts_Total_{self.system_name}_{self.system_num}.csv",
                np.array([M.sum() / float(nt)]) if nt else np.array([0.0]), delimiter=",")
        count = (M / float(nt)).astype(np.float32) if nt else M.astype(np.float32)
        np.savetxt(f"{self.folder}Acid_Contacts_Count_{self.system_name}_{self.system_num}.csv",
                count, delimiter=",")
        mean = (M / float(M.sum())) if M.sum() > 0 else (M / float(nt) if nt else M.astype(float))
        np.savetxt(f"{self.folder}Acid_Contacts_Mean_{self.system_name}_{self.system_num}.csv",
                mean, delimiter=",")


    def gen_sm_residue_contacts(self, radius):
        """
        Vectorized SM–residue (per-sequence-index) contacts.
        Computes SM COMs once per frame and reuses them across species;
        per-species accumulation is still exact and file-compatible with legacy.
        """
        species = [
            ("ProteinG3BP1", 466),
            ("ProteinTDP43", 414),
            ("ProteinFUS",   526),
            ("ProteinPABP1", 636),
            ("ProteinTIA1",  386),
            ("ProteinTTP",   326),
            ("RNA",          840),
        ]

        SM = self.u.select_atoms("resname SM")
        if len(SM.residues) == 0:
            # emit empty outputs like legacy
            for sp, LA in species:
                np.savetxt(
                    f"{self.folder}SM_Residue_Contacts_Count_{self.system_name}_{sp}_{self.system_num}.csv",
                    np.zeros(LA), delimiter=","
                )
            return

        # Precompute per-species selections and global->sequence maps
        spec_info = []
        for sp, LA in species:
            ag = self.u.select_atoms(f"resname {sp}")
            # map global atom index -> sequence position (0..LA-1), or -1 if not used
            g2s = -np.ones(self.u.atoms.n_atoms, dtype=np.int32)
            for res in ag.residues:
                order = np.argsort(res.atoms.indices)
                Lc = len(order)
                Lmap = Lc if Lc != LA else LA
                g2s[np.asarray(res.atoms.indices)[order]] = np.arange(Lmap, dtype=np.int32)
            spec_info.append({"name": sp, "LA": LA, "ag": ag, "g2s": g2s, "v": np.zeros(LA, dtype=np.uint64)})

        nt = 0
        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            # COM per SM residue (Nsm×3) once per frame
            sm_com = SM.center_of_mass(compound='residues')
            if sm_com.size == 0:
                nt += 1
                continue

            for info in spec_info:
                ag = info["ag"]
                if len(ag) == 0:
                    continue
                ia, ib = _capped_indices(sm_com, ag.positions, radius, ts.dimensions)
                if ib.size == 0:
                    continue
                # Robust to MDAnalysis variants that may return global atom indices
                if ib.max(initial=-1) >= len(ag):
                    glob = ib
                else:
                    glob = ag.indices[ib]
                s = info["g2s"][glob]
                mask = (s >= 0) & (s < info["LA"])
                if np.any(mask):
                    np.add.at(info["v"], s[mask], 1)
            nt += 1

        # write outputs per species
        for info in spec_info:
            C = (info["v"] / float(nt)) if nt else info["v"].astype(float)
            np.savetxt(
                f"{self.folder}SM_Residue_Contacts_Count_{self.system_name}_{info['name']}_{self.system_num}.csv",
                C, delimiter=","
            )


    def gen_sm_acid_contacts(self, radius):
        """
        Vectorized SM–acid-type contacts (24-vector).
        Uses one capped-distance search between all SM COMs and all (type 1..24) acid atoms in the cluster.
        """
        acid_types = list(range(1, 25))  # 1..24
        n = len(acid_types)

        SM = self.u.select_atoms("resname SM")
        if len(SM.residues) == 0:
            np.savetxt(f"{self.folder}Acid_SM_Contacts_Total_{self.system_name}_{self.system_num}.csv",
                    np.array([0.0]), delimiter=",")
            np.savetxt(f"{self.folder}Acid_SM_Contacts_Count_{self.system_name}_{self.system_num}.csv",
                    np.zeros(n), delimiter=",")
            np.savetxt(f"{self.folder}Acid_SM_Contacts_Mean_{self.system_name}_{self.system_num}.csv",
                    np.zeros(n), delimiter=",")
            print("Total Contacts: 0")
            return

        v = np.zeros(n, dtype=np.uint64)
        total = np.uint64(0)
        nt = 0

        sel_acids = " or ".join(f"type {t}" for t in acid_types)

        for ts in self.u.trajectory[self.tmin:self.tmax:self.dt]:
            # acid atoms only from the (largest) cluster this frame
            A = self.cluster_group[ts.frame].select_atoms(sel_acids)
            if len(A) == 0:
                nt += 1
                continue

            sm_com = SM.center_of_mass(compound='residues')
            if sm_com.size == 0:
                nt += 1
                continue

            ia, ib = _capped_indices(sm_com, A.positions, radius, ts.dimensions)
            if ib.size:
                # numeric per-atom acid types
                A_types = np.asarray([int(t) for t in A.types], dtype=np.int32)
                t_hits = A_types[ib]  # type (1..24) for each close SM–acid pair
                # bin directly into 24-vector (index = type-1)
                np.add.at(v, t_hits - 1, 1)
                total += np.uint64(ib.size)
            nt += 1

        np.savetxt(f"{self.folder}Acid_SM_Contacts_Total_{self.system_name}_{self.system_num}.csv",
                np.array([total / float(nt)]) if nt else np.array([0.0]), delimiter=",")
        count = (v / float(nt)) if nt else v.astype(float)
        np.savetxt(f"{self.folder}Acid_SM_Contacts_Count_{self.system_name}_{self.system_num}.csv",
                count, delimiter=",")
        mean = (v / float(total)) if total > 0 else (v / float(nt) if nt else v.astype(float))
        np.savetxt(f"{self.folder}Acid_SM_Contacts_Mean_{self.system_name}_{self.system_num}.csv",
                mean, delimiter=",")

    # Generate Diffusion Coefficients
    def calc_diffusivities(self, select):
        """
        Compute per-residue COM MSD for residues of type `select` that are inside the
        already-determined maximal continuous cluster.

        For diffusion we temporarily switch trajectory transforms to an
        SG-comoving frame with no residue wrap-back:

        - raw LAMMPS dump: ``unwrap + center_in_box``
        - mdvwhole whole XTC: ``center_in_box`` only

        This keeps structural observables on the supplied whole coordinates
        while still removing residual condensate translation for transport.

        Output: one CSV per residue with COM MSD in m^2 vs. time in seconds plus mean Rg.
        """
        # residues of this species within the already-computed maximal cluster
        bio_resids = self.max_continuous_cluster.select_atoms(f"resname {select}*").residues.resids
        if len(bio_resids) == 0:
            return

        traj = self.u.trajectory
        original_transforms = traj._transformations
        try:
            # Diffusion must not wrap molecules back into the primary cell.
            # Remove global SG translation using the full scaffold, not the
            # window-specific max_continuous_cluster subset. Using the
            # persistent subset can imprint artificial drift onto chains that
            # are outside that subset and strongly inflate MSD for some species.
            if len(self.sg_core) == 0:
                print(f"calc_diffusivities('{select}') skipped: center group is empty.", flush=True)
                return
            traj._transformations = self._build_diffusion_transforms()

            # base time between saved frames (seconds). If you already set self.frame_dt_s, we use it.
            # Otherwise fall back to 200000 fs per saved frame (2e-10 s).
            frame_dt_s = getattr(self, "frame_dt_s", 200000e-15)

            for rid in bio_resids:
                # atom group for this single residue (biopolymer)
                ag = self.u.select_atoms(f"resid {rid}")

                com_traj = np.array(
                    [ag.center_of_mass() for _ in self.u.trajectory[self.tmin:self.tmax:self.dt]],
                    dtype=float,
                )
                if com_traj.shape[0] < 2:
                    continue

                msd_sg = self._msd_fft_total(com_traj) * 1e-20
                nframes = msd_sg.size

                # Time axis: every MSD step corresponds to 'self.dt' saved frames
                lagtimes = np.arange(nframes) * (frame_dt_s * self.dt)

                # Mean Rg across the same window
                rg_vals = []
                for _ in self.u.trajectory[self.tmin:self.tmax:self.dt]:
                    rg_vals.append(ag.radius_of_gyration())
                Rg = float(np.mean(rg_vals)) if rg_vals else np.nan

                # Write CSV per-residue
                out = f"{self.folder}{select}_Diffusivity_{self.system_name}_{self.system_num}_{rid}.csv"
                pd.DataFrame(
                    {"MSD (m^2)": msd_sg, "Time (s)": lagtimes, "Rg": Rg}
                ).to_csv(out, index=False)
        finally:
            traj._transformations = original_transforms

    def calc_lammps_diffusivities(self):
        """
        Compute native-grid COM MSD for ALL scaffold residues and write:

        1. ``<system>_msd_rdp.out.all`` with one row per sampled lag time
        2. ``<system>_msd_rdp_com.npz`` sidecar with the underlying COM/Rg
           trajectory sampled at 1 ns for downstream segmented diffusion

        Behaviour:
        - Only runs when self.tmin == 0 (start from frame 0).
        - Ignores the CLI tmax for this diagnostic and goes to the end of the trajectory.
        - Samples the diffusion trajectory at a fixed native 1 ns cadence.

        Output structure:
            <time_ns> <n_residues>
            <row_idx> <resid> <resname> <COMMSD_total> <Rg_mean> <Rg_sem>

        - COM MSD values are totals (x+y+z) in Å^2.
        - Rg_mean and Rg_sem are global per-residue statistics over the sampled
          post-equilibration trajectory and are repeated on every time row for
          file compatibility with existing downstream readers.
        - File name: <TEMP root>/<system_name>_msd_rdp.out.all
        """
        if self.tmin != 0:
            return

        traj = self.u.trajectory
        n_saved = len(traj)
        if n_saved < 2:
            return

        original_transforms = traj._transformations
        traj._transformations = self._build_diffusion_transforms()

        all_residues = self.u.select_atoms("resname Protein* or resname RNA").residues
        n_residues = len(all_residues)

        # Base time between saved frames
        frame_dt_s = getattr(self, "frame_dt_s", 200000e-15)
        # Use fixed 1 ns sampling (converted in frames) as a base time series
        frames_per_ns = max(1, int(round(1e-9 / frame_dt_s)))
        start_fr = 0
        stop_fr = n_saved
        frame_idx = list(range(start_fr, stop_fr, frames_per_ns))
        nF = len(frame_idx)
        times_ns = np.array([fr * frame_dt_s * 1e9 for fr in frame_idx], dtype=float)

        # Output alongside the original LAMMPS msd file if provided; otherwise under ANALYSIS_* folder
        base_dir = os.path.dirname(self.msd_file) if hasattr(self, "msd_file") and self.msd_file else self.folder
        outfile = os.path.join(base_dir, f"{self.system_name}_msd_rdp.out.all")

        print(f"Computing MSD check (COM-only) for {n_residues} residues → {outfile}", flush=True)

        sidecar = os.path.join(base_dir, f"{self.system_name}_msd_rdp_com.npz")

        with open(outfile, 'w') as f:
            f.write("# Time-averaged MSD data (RDP_FINAL.py check_diffusivities)\n")
            f.write("# Time (ns) Number-of-rows\n")
            f.write("# Row resid resname COMMSD_total(A^2) Rg(A) Rg_SEM(A)\n")

            # Pass 1: collect COM, Rg, and Rh over sampled frames (single trajectory sweep)
            com = np.zeros((nF, n_residues, 3), dtype=float)
            rg_samples = np.zeros((nF, n_residues), dtype=float)
            rh_samples = np.zeros((nF, n_residues), dtype=float)
            for k, fr in enumerate(frame_idx):
                traj[fr]
                for ridx, residue in enumerate(all_residues):
                    ag = residue.atoms
                    com[k, ridx, :] = ag.center_of_mass()
                    rg_samples[k, ridx] = ag.radius_of_gyration()
                    rh_samples[k, ridx] = self._kirkwood_rh(ag.positions)

            com_msd = self._msd_fft_total(com)  # (nF, n_residues) in A^2
            global_rg_mean = np.nanmean(rg_samples, axis=0)
            if nF > 1:
                global_rg_sem = np.nanstd(rg_samples, axis=0, ddof=1) / math.sqrt(nF)
            else:
                global_rg_sem = np.full((n_residues,), np.nan, dtype=float)

            for lag_idx, time_ns in enumerate(times_ns):
                f.write(f"{time_ns:.6f} {n_residues}\n")
                for row_idx, residue in enumerate(all_residues, start=1):
                    ridx = row_idx - 1
                    f.write(
                        f"{row_idx} {residue.resid} {residue.resname} "
                        f"{com_msd[lag_idx, ridx]:.6f} {global_rg_mean[ridx]:.6f} {global_rg_sem[ridx]:.6e}\n"
                    )

        np.savez_compressed(
            sidecar,
            times_ns=np.asarray(times_ns, dtype=float),
            resids=np.asarray([res.resid for res in all_residues], dtype=np.int32),
            resnames=np.asarray([str(res.resname) for res in all_residues], dtype="U32"),
            com_A=np.asarray(com, dtype=np.float32),
            rg_A=np.asarray(rg_samples, dtype=np.float32),
            rh_A=np.asarray(rh_samples, dtype=np.float32),
        )

        # Restore transforms
        traj._transformations = original_transforms
        print(f"✓ MSD check written to: {outfile}")
        print(f"✓ RCC COM trajectory sidecar written to: {sidecar}")

    # Generate per-dt RG snapshots (RG_<system>_rg.out.all)
    def calc_rg(self):
        """
        Compute radius of gyration per residue at every dt interval (in ns).

        Logic is copied from RG_FINAL.calc_diffusivities(), but exposed here as
        `calc_rg` so that RDP_FINAL generates the same RG_<system>_rg.out.all
        file used by the diffusion post-processing (diffusion.py).

        - Only runs when self.tmin == 0.
        - Walks from the start of the trajectory and samples frames spaced by ``dt`` (CLI, in ns).
        - For each sampled frame, writes a block:
              <time_ns> <n_residues>
              <row> <resid> <resname> <Rg(A)>
        """
        if self.tmin != 0:
            return

        traj = self.u.trajectory
        n_saved = len(traj)
        if n_saved < 2:
            return

        all_residues = self.u.select_atoms("resname Protein* or resname RNA").residues
        n_residues = len(all_residues)

        # Time per saved frame (seconds)
        frame_dt_s = getattr(self, "frame_dt_s", 200000e-15)
        frame_dt_ns = frame_dt_s * 1e9

        # CLI dt is in ns; require positive
        dt_ns = float(self.dt_ns)
        if dt_ns <= 0.0:
            raise ValueError("RDP_FINAL.calc_rg: dt must be > 0 ns")

        # Convert dt (ns) to a frame stride
        dt_frames = max(1, int(round(dt_ns / frame_dt_ns)))

        # Build list of frames starting at 0 with stride dt_frames
        frame_idx = list(range(0, n_saved, dt_frames))
        if not frame_idx:
            return

        # Corresponding times in ns
        times_ns = np.array([fr * frame_dt_ns for fr in frame_idx], dtype=float)

        # Respect tmax (ns) if provided (>0); otherwise use full trajectory
        if self.tmax_ns > 0.0:
            mask = times_ns <= (float(self.tmax_ns) + 1e-9)
            if not np.any(mask):
                return
            times_ns = times_ns[mask]
            frame_idx = [fr for fr, keep in zip(frame_idx, mask) if keep]

        outfile = f"{self.folder}RG_{self.system_name}_rg.out.all"
        tmp_outfile = f"{outfile}.tmp"
        print(f"Computing per-dt RG snapshots for {n_residues} residues → {outfile}", flush=True)

        try:
            with open(tmp_outfile, 'w') as f:
                f.write("# Per-dt RG snapshots (RDP_FINAL.py)\n")
                f.write("# Time (ns) Number-of-rows\n")
                f.write("# Row resid resname Rg(A)\n")

                for t_ns, fr in zip(times_ns, frame_idx):
                    # Move trajectory to the requested frame
                    traj[fr]
                    f.write(f"{t_ns:.6f} {n_residues}\n")
                    for row_idx, residue in enumerate(all_residues, start=1):
                        rg_val = residue.atoms.radius_of_gyration()
                        f.write(
                            f"{row_idx} {residue.resid} {residue.resname} "
                            f"{rg_val:.6f}\n"
                        )
            os.replace(tmp_outfile, outfile)
        except Exception:
            if os.path.exists(tmp_outfile):
                try:
                    os.remove(tmp_outfile)
                except OSError:
                    pass
            raise

        print(f"RG time series written to: {outfile}")

    # Generate Stress Tensors
    def parse_stress_file(self, stress_file):
        """Sum the per-atom virial stress over the maximal-cluster atoms in each timestep of ``stress_file`` and write the condensate stress-tensor time series CSV (for Green-Kubo viscosity)."""
        write_lines = [["Timestep", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]]
        cluster_list = self.max_continuous_cluster.atoms.ids
        pxx = 0.0
        pyy = 0.0
        pzz = 0.0
        pxy = 0.0
        pxz = 0.0
        pyz = 0.0
        n = 0
        timestep = 0
        with open(stress_file, "r+") as file:
            for line in file:
                if line.strip() == "ITEM: TIMESTEP":
                    # on new block, flush previous block if it had content
                    if n > 0:
                        write_lines.append([timestep, pxx, pyy, pzz, pxy, pxz, pyz])
                    timestep = int(next(file))
                    pxx = 0.0
                    pyy = 0.0
                    pzz = 0.0
                    pxy = 0.0
                    pxz = 0.0
                    pyz = 0.0
                    n = 0
                elif len(line.strip().split()) == 16:
                    atm = line.split()
                    atom_id = int(atm[0])
                    if atom_id in cluster_list:
                        pxx += float(atm[6])
                        pyy += float(atm[7])
                        pzz += float(atm[8])
                        pxy += float(atm[9])
                        pxz += float(atm[10])
                        pyz += float(atm[11])
                        n += 1
        # flush last block
        if n > 0:
            write_lines.append([timestep, pxx, pyy, pzz, pxy, pxz, pyz])
        self.end_time = timestep
        with open("{}Stress_Tensor_{}_{}.csv".format(self.folder, self.system_name, self.system_num), "w") as file:
            writer = csv.writer(file)
            writer.writerows(write_lines)

    def parse_stress_file_tracked(self):
        """Apply frame-wise tracked cluster membership to pre-segmented stress.

        Loads ``Stress_Segmented_{system}_{window}.npz`` (written by
        ``segment_stress.py`` before the parallel RCC windows), applies this
        window's tracked cluster membership per frame, and writes
        ``Stress_Tensor_Tracked_{system}_{window}.csv``.

        Runs at every window.
        """
        # Try integer then float label (matches _time_labels convention)
        npz_path = None
        for label in [str(int(self.tmin_ns)), str(self.tmin_ns)]:
            candidate = "{}Stress_Segmented_{}_{}.npz".format(
                self.folder, self.system_name, label
            )
            if os.path.isfile(candidate):
                npz_path = candidate
                break
        if npz_path is None:
            print(f"parse_stress_file_tracked: segmented NPZ not found for "
                  f"{self.system_name} window {self.tmin_ns}", flush=True)
            return

        frame_keys = getattr(self, "_tracked_frame_keys", [])
        tracked = getattr(self, "_tracked_clusters", [])
        if not frame_keys or not tracked:
            print("parse_stress_file_tracked: no tracked clusters for this window", flush=True)
            return

        data = np.load(npz_path, allow_pickle=False)
        timesteps = data["timesteps"]
        sg_resids = data["sg_resids"]
        pxx_win = data["pxx"].astype(np.float64)
        pyy_win = data["pyy"].astype(np.float64)
        pzz_win = data["pzz"].astype(np.float64)
        pxy_win = data["pxy"].astype(np.float64)
        pxz_win = data["pxz"].astype(np.float64)
        pyz_win = data["pyz"].astype(np.float64)
        ts_win = timesteps

        if len(ts_win) == 0:
            print(f"parse_stress_file_tracked: empty segmented NPZ for window "
                  f"{self.tmin_ns} ns", flush=True)
            return

        # Build per-tracked-frame residue masks
        resid_to_col = {int(r): j for j, r in enumerate(sg_resids)}
        n_tracked = len(frame_keys)
        tracked_masks = []
        for cluster_resids in tracked:
            mask_r = np.zeros(len(sg_resids), dtype=bool)
            for resid in cluster_resids:
                col = resid_to_col.get(int(resid))
                if col is not None:
                    mask_r[col] = True
            tracked_masks.append(mask_r)

        # Map stress blocks to nearest tracked frame via time
        frame_dt_ns = self.frame_dt_s * 1e9
        tracked_times = np.array([fk * frame_dt_ns for fk in frame_keys], dtype=np.float64)

        write_lines = [["Timestep", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]]
        for i in range(len(ts_win)):
            t_ns = float(ts_win[i]) * (self.timestep_fs * 1e-6)
            pos = int(np.searchsorted(tracked_times, t_ns))
            if pos >= n_tracked:
                pos = n_tracked - 1
            elif pos > 0:
                if (t_ns - tracked_times[pos - 1]) < (tracked_times[pos] - t_ns):
                    pos = pos - 1
            rmask = tracked_masks[pos]
            write_lines.append([
                int(ts_win[i]),
                float(np.sum(pxx_win[i, rmask])),
                float(np.sum(pyy_win[i, rmask])),
                float(np.sum(pzz_win[i, rmask])),
                float(np.sum(pxy_win[i, rmask])),
                float(np.sum(pxz_win[i, rmask])),
                float(np.sum(pyz_win[i, rmask])),
            ])

        out_path = "{}Stress_Tensor_Tracked_{}_{}.csv".format(
            self.folder, self.system_name, self.system_num
        )
        with open(out_path, "w") as fh:
            writer = csv.writer(fh)
            writer.writerows(write_lines)
        print(f"Tracked stress tensor ({len(write_lines)-1} blocks, window "
              f"{self.tmin_ns}-{self.tmax_ns} ns) written to: {out_path}", flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run RDP/RCC Analysis on LAMMPS trajectories.")

    # Optional parsing logic to maintain pipeline compatibility
    parser.add_argument('--path', type=str, required=True, help='Base absolute path containing temperature directories')
    parser.add_argument('--folder', type=str, required=True, help='Folder name (e.g., DSM, NDSM, SG)')
    parser.add_argument('--temp', type=str, help='Temperature (e.g., 285)')
    parser.add_argument('--tmin', type=float, required=True, help='Start time in ns')
    parser.add_argument('--tmax', type=float, required=True, help='End time in ns')
    parser.add_argument('--dt', type=float, required=True, help='Time interval in ns')
    parser.add_argument(
        '--structure-coordinate-mode',
        type=str,
        default='auto',
        choices=['auto', 'compact', 'historical'],
        help='Structural observable coordinate mode. auto=historical for whole XTC, compact for raw dump.'
    )
    parser.add_argument(
        '--traj-source',
        type=str,
        default='auto',
        choices=['auto', 'whole', 'raw'],
        help='Trajectory source selection. auto prefers <system>_whole.xtc when present, else raw <system>_traj.lammpsdump.'
    )

    # Check if run with positional arguments directly (legacy run_sg style) or via argparse pipeline
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        name_legacy = sys.argv[1]
        tmin = float(sys.argv[2])
        tmax = float(sys.argv[3])
        dt = float(sys.argv[4])
        # Temporarily infer path based on script runtime or hardcode fallback if invoked via legacy
        path = os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS/TEMP_285/DSM/")
        folder_name = os.path.basename(os.path.normpath(path)).upper()
        systems_to_run = [name_legacy]
    else:
        parser.add_argument('--system', type=str, default=None, help='Specific system name. If omitted, loops all in folder.')
        args = parser.parse_args()
        folder_name = args.folder.upper()
        path = os.path.join(args.path, args.folder)
        tmin = args.tmin
        tmax = args.tmax
        dt = args.dt
        structure_coordinate_mode = args.structure_coordinate_mode
        traj_source = args.traj_source

        if args.system:
            systems_to_run = [args.system]
        else:
            if args.folder == "SG":
                systems_to_run = ["sg_X"]
            elif args.folder == "DSM":
                systems_to_run = ["dsm_hydroxyquinoline", "dsm_lipoamide", "dsm_lipoic_acid", "dsm_dihydrolipoic_acid",
                                  "dsm_anisomycin", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine",
                                  "dsm_mitoxantrone", "dsm_daunorubicin"]
            elif args.folder == "NDSM":
                systems_to_run = ["ndsm_dmso", "ndsm_valeric_acid", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                                  "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine",
                                  "ndsm_anthraquinone", "ndsm_acetylenapthacene", "ndsm_anacardic"]
            else:
                systems_to_run = []

    cutoff = 20
    dims = 1200
    bin_size = 20
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        structure_coordinate_mode = "auto"
        traj_source = "auto"

    def resolve_data_file(base_path, folder, system_name):
        folder = folder.upper()
        if folder == "SG":
            return os.path.join(base_path, "GRO", "sys.data")
        if folder == "NDSM":
            return os.path.join(base_path, "GRO", f"sim_N_{system_name}_1uM.data")
        if folder == "DSM":
            return os.path.join(base_path, "GRO", f"sim_Y_{system_name}_1uM.data")
        raise ValueError(f"Unsupported folder '{folder}'. Expected one of: DSM, NDSM, SG.")

    def resolve_traj_file(base_path, system_name, source_mode):
        trj_dir = os.path.join(base_path, "TRJ")
        whole_link = os.path.join(trj_dir, f"{system_name}_whole.xtc")
        whole_xtc = os.path.join(trj_dir, f"{system_name}_whole_1ns.xtc")
        raw_dump = os.path.join(trj_dir, f"{system_name}_traj.lammpsdump")

        if source_mode == "whole":
            for candidate in (whole_link, whole_xtc):
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(
                f"Whole trajectory requested but not found for {system_name}. Checked: {whole_link}, {whole_xtc}"
            )

        if source_mode == "raw":
            return raw_dump

        for candidate in (whole_link, whole_xtc, raw_dump):
            if os.path.isfile(candidate):
                return candidate
        return raw_dump

    any_failed = False
    for name in systems_to_run:
        data_file = resolve_data_file(path, folder_name, name)
        traj_file = resolve_traj_file(path, name, traj_source)
        stress_file = os.path.join(path, "STRESS", f"{name}_stress.out.all")
        cluster_file = os.path.join(path, "CLUSTER", f"{name}_cluster.out.all")
        msd_file = os.path.join(path, "MSD", f"{name}_msd.out.all")

        try:
            required_files = {
                "data_file": data_file,
                "traj_file": traj_file,
                "stress_file": stress_file,
                "cluster_file": cluster_file,
                "msd_file": msd_file,
            }
            missing = [f"{label}: {filepath}" for label, filepath in required_files.items() if not os.path.isfile(filepath)]
            if missing:
                raise FileNotFoundError(
                    "Missing required input files:\n  - " + "\n  - ".join(missing)
                )

            rdp_obj = rdp(data_file=data_file, traj_file=traj_file, stress_file=stress_file, cluster_file=cluster_file,
                      msd_file=msd_file, system_name=name, tmin=tmin, tmax=tmax,
                      dt=dt, cutoff=cutoff, dims=dims, bin_size=bin_size,
                      structure_coordinate_mode=structure_coordinate_mode)
        except Exception as e:
            any_failed = True
            print(f"Error running {name}: {e}")
            traceback.print_exc()

    if any_failed:
        sys.exit(1)
