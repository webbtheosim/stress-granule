"""Time-window averaging of per-frame MD observables across blocks and replicates.

Pipeline role
-------------
Step 1 of the stress-granule (SG) MD analysis pipeline. Runs after
``rcc_analysis.py`` (Step 0) and before ``max_cluster.py`` (Step 2). It
averages the per-window CSV products written by Step 0 into per-system block
averages (with SEM), then into per-class (DSM / NDSM) aggregates, and stages /
synthesizes the diffusion artifacts the post-RCC stages require.

What it averages
----------------
For each small molecule / control system, over the requested time windows:
- radial density profiles (protein, RNA, SG, SM),
- domain/residue/acid contact-count matrices (incl. SM-contact variants),
- cluster geometry (Rg, droplet counts, masses), PCA shape descriptors,
- G3BP1 diffusivity (MSD) curves time-aligned across chains/windows,
- per-class stress-tensor traces (for Green-Kubo viscosity downstream).
It also materializes the aggregate RCC diffusion inputs
(``<CAT>_msd_rdp.out.all`` + COM sidecar + cluster masks) that the DSM/NDSM
average rows need but no earlier stage produces.

Key inputs
----------
- ``ANALYSIS_{SG,DSM,NDSM}/`` per-window CSV/out products from Step 0, plus the
  external RCC MSD artifacts under
  ``./PYTHON_ANALYSIS/<TEMP>/<CAT>/MSD/``.
- CLI flags: ``--path`` (TEMP_XXX dir), ``--folder`` (output prefix, default
  ``CLASSIFY``), ``--temp``, ``--tmin``, ``--dt``, ``--tmax``, ``--use-lists``,
  ``--plot-only``.

Key outputs
-----------
- Per-system / per-class averaged CSVs under ``ANALYSIS_{SG,DSM,NDSM}_AVE/``
  (``Density_Profile_*``, ``*_Contacts_Mean/SEM_*``, ``Cluster_*``, ``PCA_*``,
  ``G3BP1_Diffusivity_*``, ``Stress_Tensor_*``, aggregate RCC diffusion files).
- The empty ``FIGURES/``, ``IMAGES/``, ``RESULTS/`` directory scaffold consumed
  by later plotting stages.

CLI invocation
--------------
    python average_simulations.py --path TEMP_300 --folder CLASSIFY --temp 300 \
        --tmin 50 --dt 50 --tmax 2000
"""
import argparse
import math
import os
import re
import shutil
import sys
from glob import glob

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _time_labels(t):
    """Return candidate timestamp strings matching both legacy and RCC outputs."""
    labels = []
    try:
        tf = float(t)
        if tf.is_integer():
            labels.append(str(int(tf)))
        labels.append(str(tf))
    except Exception:
        labels.append(str(t))

    seen = set()
    ordered = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _resolve_time_file(path_builder, t):
    """Resolve a raw analysis file written with either integer or float window labels."""
    for label in _time_labels(t):
        path = path_builder(label)
        if os.path.isfile(path):
            return path
    return None


def _format_window_label(t):
    """Write aggregate raw files using the floating-point label style from RCC outputs."""
    try:
        tf = float(t)
    except Exception:
        return str(t)
    return f"{tf:.1f}" if tf.is_integer() else str(tf)


def _matrix_sem(mats):
    """Return the element-wise standard error of the mean across a list of equally shaped matrices (NaN if <2 inputs)."""
    arr = np.stack([np.asarray(m, dtype=float) for m in mats], axis=0)
    if arr.shape[0] <= 1:
        return np.full(arr.shape[1:], np.nan, dtype=float)
    return np.nanstd(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])


def _parse_cluster_resids(path):
    """Read 'resid X or resid Y ...' cluster masks into a sorted integer list."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    nums = re.findall(r"resid\s+(\d+)", text)
    if not nums:
        nums = re.findall(r"\b(\d+)\b", text)
    return sorted({int(x) for x in nums})


def _load_rdp_msd_table(msd_path):
    """Parse <system>_msd_rdp.out.all into (times_ns, msd_mat_A2, resids, resnames)."""
    times = []
    blocks = []
    resid_list = None
    resname_list = None

    with open(msd_path, "r", encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2:
            i += 1
            continue
        try:
            time_ns = float(parts[0])
            nrows = int(parts[1])
        except ValueError:
            i += 1
            continue
        i += 1
        rows_msd = []
        rows_resid = []
        rows_resname = []
        for _ in range(nrows):
            if i >= len(lines):
                break
            cols = lines[i].split()
            if len(cols) < 4:
                i += 1
                continue
            try:
                rows_resid.append(int(cols[1]))
                rows_resname.append(cols[2])
                rows_msd.append(float(cols[3]))
            except ValueError:
                i += 1
                continue
            i += 1
        if rows_msd:
            times.append(time_ns)
            blocks.append(rows_msd)
            if resid_list is None:
                resid_list = rows_resid
                resname_list = rows_resname

    if not blocks or resid_list is None or resname_list is None:
        raise RuntimeError(f"No RCC MSD data parsed from {msd_path}")

    nmin = min(len(row) for row in blocks)
    msd_mat_A2 = np.array([row[:nmin] for row in blocks], dtype=float)
    times_ns = np.array(times, dtype=float)
    resids = np.array(resid_list[:nmin], dtype=int)
    resnames = np.array(resname_list[:nmin], dtype=object)
    return times_ns, msd_mat_A2, resids, resnames


def generate_aggregate_rdp_inputs(temp_root, category, system_names, aggregate_tag=None):
    """
    Build class-level RCC diffusion artifacts expected by system_analysis.py.

    The raw RCC stage only writes per-system `<system>_msd_rdp.out.all` and
    `<system>_msd_rdp_com.npz` files under `/projects/.../<CAT>/MSD/`. The
    aggregate DSM/NDSM rows in the post-RCC pipeline later ask for
    `ANALYSIS_<CAT>/<CAT>_msd_rdp.out.all`, but no earlier stage created them.

    This helper materializes those missing aggregate inputs by concatenating the
    member-system RCC trajectories on a unique resid axis and by writing matching
    aggregate maximal-cluster masks in the raw `ANALYSIS_<CAT>` folder.
    """
    systems = list(system_names)
    if not systems:
        print(f"[aggregate_rdp] No member systems supplied for {category}; skipping.")
        return

    aggregate_tag = aggregate_tag or category
    temp_root = os.path.abspath(temp_root)
    raw_dir = os.path.join(temp_root, f"ANALYSIS_{category}")
    if not os.path.isdir(raw_dir):
        print(f"[aggregate_rdp] Raw analysis folder missing for {category}: {raw_dir}")
        return

    real_raw_dir = os.path.realpath(raw_dir)
    temp_label = os.path.basename(os.path.dirname(real_raw_dir))
    external_msd_dir = os.path.join(
        _REPO_ROOT, "PYTHON_ANALYSIS", temp_label, category, "MSD",
    )

    def resolve_external(system_name, suffix):
        """Return the local-then-external path to ``<system_name><suffix>``; raise if neither exists."""
        local_path = os.path.join(raw_dir, f"{system_name}{suffix}")
        if os.path.isfile(local_path):
            return local_path
        external_path = os.path.join(external_msd_dir, f"{system_name}{suffix}")
        if os.path.isfile(external_path):
            return external_path
        raise FileNotFoundError(
            f"Missing RCC diffusion artifact for {system_name}: "
            f"checked {local_path} and {external_path}"
        )

    base_times = None
    base_resids = None
    base_resnames = None
    resid_block = None
    msd_blocks = []
    com_blocks = []
    rg_sample_blocks = []
    rh_sample_blocks = []
    agg_resids = []
    agg_resnames = []

    for sys_idx, system_name in enumerate(systems):
        msd_path = resolve_external(system_name, "_msd_rdp.out.all")
        sidecar_path = resolve_external(system_name, "_msd_rdp_com.npz")

        times_ns, msd_mat_A2, resids, resnames = _load_rdp_msd_table(msd_path)
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            side_times = np.asarray(sidecar["times_ns"], dtype=float)
            side_resids = np.asarray(sidecar["resids"], dtype=int)
            side_resnames = np.asarray(sidecar["resnames"]).astype(str)
            com_A = np.asarray(sidecar["com_A"], dtype=np.float32)
            rg_A = np.asarray(sidecar["rg_A"], dtype=np.float32)
            if "rh_A" in sidecar:
                rh_A = np.asarray(sidecar["rh_A"], dtype=np.float32)
            else:
                rh_A = np.full(rg_A.shape, np.nan, dtype=np.float32)

        if base_times is None:
            base_times = times_ns
            base_resids = resids
            base_resnames = np.asarray(resnames).astype(str)
            resid_block = int(np.max(base_resids))
        else:
            if not np.allclose(times_ns, base_times):
                raise ValueError(f"RCC MSD time grid mismatch for {system_name}")
            if not np.array_equal(resids, base_resids):
                raise ValueError(f"RCC MSD resid layout mismatch for {system_name}")
            if not np.array_equal(np.asarray(resnames).astype(str), base_resnames):
                raise ValueError(f"RCC MSD resname layout mismatch for {system_name}")

        if not np.allclose(side_times, base_times):
            raise ValueError(f"RCC COM sidecar time grid mismatch for {system_name}")
        if not np.array_equal(side_resids, base_resids):
            raise ValueError(f"RCC COM sidecar resid layout mismatch for {system_name}")
        if not np.array_equal(side_resnames, base_resnames):
            raise ValueError(f"RCC COM sidecar resname layout mismatch for {system_name}")

        offset = sys_idx * resid_block
        agg_resids.append(base_resids + offset)
        agg_resnames.append(base_resnames.copy())
        msd_blocks.append(msd_mat_A2)
        com_blocks.append(com_A)
        rg_sample_blocks.append(rg_A)
        rh_sample_blocks.append(rh_A)

    agg_times = np.asarray(base_times, dtype=float)
    agg_resids = np.concatenate(agg_resids).astype(np.int32)
    agg_resnames = np.concatenate(agg_resnames).astype(str)
    agg_msd = np.concatenate(msd_blocks, axis=1).astype(float)
    agg_com = np.concatenate(com_blocks, axis=1).astype(np.float32)
    agg_rg_samples = np.concatenate(rg_sample_blocks, axis=1).astype(np.float32)
    agg_rh_samples = np.concatenate(rh_sample_blocks, axis=1).astype(np.float32)
    agg_rg_mean = np.nanmean(agg_rg_samples, axis=0)
    if agg_rg_samples.shape[0] > 1:
        agg_rg_sem = np.nanstd(agg_rg_samples, axis=0, ddof=1) / math.sqrt(agg_rg_samples.shape[0])
    else:
        agg_rg_sem = np.full(agg_rg_mean.shape, np.nan, dtype=float)

    msd_out = os.path.join(raw_dir, f"{aggregate_tag}_msd_rdp.out.all")
    with open(msd_out, "w", encoding="utf-8") as handle:
        handle.write("# Aggregate RCC MSD data generated by average_simulations.py\n")
        handle.write("# Time (ns) Number-of-rows\n")
        handle.write("# Row resid resname COMMSD_total(A^2) Rg(A) Rg_SEM(A)\n")
        for lag_idx, time_ns in enumerate(agg_times):
            handle.write(f"{time_ns:.6f} {agg_resids.size}\n")
            for row_idx, (resid, resname) in enumerate(zip(agg_resids, agg_resnames), start=1):
                j = row_idx - 1
                handle.write(
                    f"{row_idx} {int(resid)} {resname} "
                    f"{agg_msd[lag_idx, j]:.6f} {agg_rg_mean[j]:.6f} {agg_rg_sem[j]:.6e}\n"
                )

    sidecar_out = os.path.join(raw_dir, f"{aggregate_tag}_msd_rdp_com.npz")
    np.savez_compressed(
        sidecar_out,
        times_ns=agg_times,
        resids=agg_resids,
        resnames=agg_resnames,
        com_A=agg_com,
        rg_A=agg_rg_samples,
        rh_A=agg_rh_samples,
    )

    rg_out = os.path.join(raw_dir, f"RG_{aggregate_tag}_rg.out.all")
    with open(rg_out, "w", encoding="utf-8") as handle:
        handle.write("# Aggregate per-dt RG snapshots generated by average_simulations.py\n")
        handle.write("# Time (ns) Number-of-rows\n")
        handle.write("# Row resid resname Rg(A)\n")
        for frame_idx, time_ns in enumerate(agg_times):
            handle.write(f"{time_ns:.6f} {agg_resids.size}\n")
            for row_idx, (resid, resname) in enumerate(zip(agg_resids, agg_resnames), start=1):
                handle.write(
                    f"{row_idx} {int(resid)} {resname} "
                    f"{float(agg_rg_samples[frame_idx, row_idx - 1]):.6f}\n"
                )

    pattern = os.path.join(raw_dir, f"Max_Continuous_Cluster_{systems[0]}_*.txt")
    base_window_paths = sorted(glob(pattern))
    if not base_window_paths:
        raise FileNotFoundError(f"No cluster masks found for aggregate source system: {systems[0]}")

    window_times = []
    for path in base_window_paths:
        m = re.match(rf"Max_Continuous_Cluster_{re.escape(systems[0])}_([0-9.]+)\.txt$", os.path.basename(path))
        if m:
            window_times.append(float(m.group(1)))
    window_times = sorted(window_times)

    for time_ns in window_times:
        aggregate_resids = []
        for sys_idx, system_name in enumerate(systems):
            cluster_path = _resolve_time_file(
                lambda label: os.path.join(raw_dir, f"Max_Continuous_Cluster_{system_name}_{label}.txt"),
                time_ns,
            )
            if cluster_path is None:
                raise FileNotFoundError(
                    f"Missing cluster mask for {system_name} at t={time_ns} ns in {raw_dir}"
                )
            offset = sys_idx * resid_block
            aggregate_resids.extend(r + offset for r in _parse_cluster_resids(cluster_path))
        cluster_out = os.path.join(
            raw_dir,
            f"Max_Continuous_Cluster_{aggregate_tag}_{_format_window_label(time_ns)}.txt",
        )
        with open(cluster_out, "w", encoding="utf-8") as handle:
            handle.write(" or ".join(f"resid {resid}" for resid in sorted(set(aggregate_resids))))

    print(
        f"[aggregate_rdp] Wrote aggregate RCC diffusion inputs for {aggregate_tag}: "
        f"{msd_out}, {sidecar_out}, RG_{aggregate_tag}_rg.out.all, and {len(window_times)} cluster masks"
    )

class average:
    """Per-system time-window averager: averages one system's per-window observables across blocks.

    Reads the per-window CSVs in a raw ``ANALYSIS_<CAT>`` folder for a single
    small molecule / control system and writes block-averaged products (mean and
    SEM) into the matching ``<CAT>_AVE`` folder.
    """

    def __init__(self, path, category, tmin, tmax, dt):
        """Store the dataset root, raw input category, and the [tmin, tmax) window with stride ``dt``."""
        self.path=path
        self.df = pd.DataFrame()
        self.category = category
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        """Average the radial density profile of ``biopolymer`` for system ``sm`` over all windows, writing mean density + SEM."""
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        distance_col = None
        for i in range(self.tmin,self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(category,biopolymer, sm, str(i)))
                continue
            try:
                cur_df = pd.read_csv(input_path)
                distance_col = cur_df["Distance from center of mass (A)"]
                if biopolymer == "SM":
                    density_col = list(cur_df["SM density (mg/mL)"])
                else:
                    density_col = list(cur_df["Protein density (mg/mL)"])
                df_temp[str(i)] = density_col
            except Exception:
                print("{}/DensityProfile_{}_{}_{}.csv unreadable".format(category,biopolymer, sm, str(i)))
                continue

        if distance_col is None or df_temp.empty:
            print("No DensityProfile data found for {} {} in {}".format(biopolymer, sm, category))
            return

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def contact_ave(self, biopolymer, sm):
        """Average ``biopolymer`` contact-count matrices for system ``sm`` over all windows, writing mean and SEM matrices."""
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        first_path = _resolve_time_file(
            lambda label: "{}/{}_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, label),
            self.tmin,
        )
        if first_path is None:
            print("{}_Contacts_Count_{}_{}.csv File Missing".format(biopolymer, sm, str(self.tmin)))
            return
        df = pd.read_csv(first_path, header=None)
        mats = [df.to_numpy(dtype=float)]
        base_shape = mats[0].shape
        count = 1
        for i in range(int(self.tmin+self.dt),self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/{}_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("{}_Contacts_Count_{}_{}.csv.csv File Missing".format(biopolymer, sm, str(i)))
                continue
            try:
                df_temp = pd.read_csv(input_path, header=None)
                arr_temp = df_temp.to_numpy(dtype=float)
                if arr_temp.shape != base_shape:
                    print("{}_Contacts_Count_{}_{} shape mismatch (skipping)".format(biopolymer, sm, str(i)))
                    continue
                df = df.add(df_temp, fill_value=0)
                mats.append(arr_temp)
                count += 1
            except Exception:
                print("{}_Contacts_Count_{}_{}.csv.csv unreadable".format(biopolymer, sm, str(i)))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=False)
        pd.DataFrame(_matrix_sem(mats)).to_csv(
            "{}/{}_AVE/{}_Contacts_SEM_{}.csv".format(self.path, self.category, biopolymer, sm),
            index=False,
            header=False,
        )

    def contact_sm_ave(self, biopolymer, sm):
        """
        Average SM contact matrices across time.

        - For Acid–SM contacts, uses legacy
          `<biopolymer>_SM_Contacts_Count_<sm>_<t>.csv` files.
        - For Residue–SM contacts, first prefers the legacy
          `Residue_SM_Contacts_Count_<sm>_<t>.csv` files; if these are
          missing, it reconstructs a 7-element vector
          [G3BP1, PABP1, TTP, TIA1, TDP43, FUS, RNA]
          from the newer per-species
          `SM_Residue_Contacts_Count_<sm>_<species>_<t>.csv` outputs
          produced by `RDP_FINAL.py`.
        """
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())

        # Special handling for residue–SM contacts, where RDP_FINAL now writes
        # per-species sequence-index vectors instead of the old 7-vector.
        if biopolymer == "Residue":
            # Order must match SYSTEM_ANALYSIS.py: G3BP1, PABP1, TTP, TIA1, TDP43, FUS, RNA
            species = [
                "ProteinG3BP1",
                "ProteinPABP1",
                "ProteinTTP",
                "ProteinTIA1",
                "ProteinTDP43",
                "ProteinFUS",
                "RNA",
            ]

            sum_vec = None
            vectors = []
            n_used = 0

            for t in range(self.tmin, self.tmax, self.dt):
                # 1) Legacy aggregate file: Residue_SM_Contacts_Count_<sm>_<t>.csv
                legacy = _resolve_time_file(
                    lambda label: os.path.join(
                        category,
                        "Residue_SM_Contacts_Count_{}_{}.csv".format(sm, label),
                    ),
                    t,
                )
                if legacy is not None and os.path.isfile(legacy):
                    try:
                        vec = np.loadtxt(legacy, delimiter=",", dtype=float)
                    except Exception:
                        print(
                            "Residue_SM_Contacts_Count_{}_{} unreadable (skipping)".format(
                                sm, t
                            )
                        )
                        continue
                    vec = np.atleast_1d(vec)
                else:
                    # 2) New per-species sequence-index files:
                    #    SM_Residue_Contacts_Count_<sm>_<species>_<t>.csv
                    per_species = []
                    missing = False
                    for sp in species:
                        fn = _resolve_time_file(
                            lambda label: os.path.join(
                                category,
                                "SM_Residue_Contacts_Count_{}_{}_{}.csv".format(
                                    sm, sp, label
                                ),
                            ),
                            t,
                        )
                        if fn is None:
                            missing = True
                            break
                        try:
                            v = np.loadtxt(fn, delimiter=",", dtype=float)
                        except Exception:
                            missing = True
                            break
                        v = np.atleast_1d(v)
                        # Sum over sequence indices → total contacts for that species
                        per_species.append(np.nansum(v))

                    if missing or not per_species:
                        print(
                            "Residue_SM contacts file missing for {} at t={} (skipping)".format(
                                sm, t
                            )
                        )
                        continue

                    vec = np.asarray(per_species, dtype=float)

                if sum_vec is None:
                    sum_vec = vec
                else:
                    if sum_vec.shape != vec.shape:
                        print(
                            "Residue_SM contacts shape mismatch for {} at t={} (skipping)".format(
                                sm, t
                            )
                        )
                        continue
                    sum_vec = sum_vec + vec
                vectors.append(vec.astype(float))
                n_used += 1

            if sum_vec is None or n_used == 0:
                print("No Residue–SM contacts found for {}; skipping average".format(sm))
                return

            mean_vec = sum_vec / float(n_used)
            # Write as a 7×1 column to match legacy layout
            df_out = pd.DataFrame(mean_vec.reshape(-1, 1))
            df_out.to_csv(
                "{}/{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(
                    self.path, self.category, biopolymer, sm
                ),
                index=False,
                header=False,
            )
            sem_vec = _matrix_sem([v.reshape(-1, 1) for v in vectors])
            pd.DataFrame(sem_vec).to_csv(
                "{}/{}_AVE/{}_SM_Contacts_SEM_{}.csv".format(
                    self.path, self.category, biopolymer, sm
                ),
                index=False,
                header=False,
            )
            return

        # Default path (e.g. Acid–SM): legacy *_SM_Contacts_Count_<sm>_<t>.csv
        def _load_file(t):
            """Resolve the ``{biopolymer}_SM_Contacts_Count_{sm}_{t}.csv`` path for window ``t`` (or None)."""
            return _resolve_time_file(
                lambda label: "{}/{}_SM_Contacts_Count_{}_{}.csv".format(category, biopolymer, sm, label),
                t,
            )

        first_path = _load_file(self.tmin)
        if first_path is None:
            print("{}_SM contacts file missing for {} at t={} (skipping)".format(biopolymer, sm, self.tmin))
            return

        df = pd.read_csv(first_path, header=None)
        mats = [df.to_numpy(dtype=float)]
        base_shape = mats[0].shape
        count = 1
        for i in range(int(self.tmin + self.dt), self.tmax, self.dt):
            path_i = _load_file(i)
            if path_i is None:
                print("{}_SM contacts file missing for {} at t={} (skipping)".format(biopolymer, sm, i))
                continue
            try:
                df_temp = pd.read_csv(path_i, header=None)
                arr_temp = df_temp.to_numpy(dtype=float)
                if arr_temp.shape != base_shape:
                    print("{}_SM contacts shape mismatch for {} at t={} (skipping)".format(biopolymer, sm, i))
                    continue
                df = df.add(df_temp, fill_value=0)
                mats.append(arr_temp)
                count += 1
            except Exception:
                print("{}_SM_Contacts_Count_{}_{} missing or unreadable".format(biopolymer, sm, i))

        df = df.div(count, fill_value=0)
        df.to_csv("{}/{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm), index=False, header=False)
        pd.DataFrame(_matrix_sem(mats)).to_csv(
            "{}/{}_AVE/{}_SM_Contacts_SEM_{}.csv".format(self.path, self.category, biopolymer, sm),
            index=False,
            header=False,
        )

    def cluster_ave(self, biopolymer, sm):
        """Average per-window cluster geometry (Rg, droplet counts, masses) for ``biopolymer``/``sm`` and append explicit SEM columns."""
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        first_path = _resolve_time_file(
            lambda label: "{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, label),
            self.tmin,
        )
        if first_path is None:
            print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(self.tmin)))
            return
        df_cols = pd.read_csv(first_path).columns.tolist()
        for i in range(self.tmin,self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))
                continue
            try:
                mean_values = pd.read_csv(input_path).mean()
                df_list.append(mean_values)
                index_list.append(i)
            except Exception:
                print("Cluster_{}_{}_{}.csv unreadable".format(biopolymer, sm, str(i)))

        if not df_list:
            print("No Cluster data found for {} {} in {}".format(biopolymer, sm, category))
            return

        df = pd.DataFrame(columns=df_cols,data=df_list, index=index_list)

        rg_sem = df["Largest Droplet Radius of Gyration"].sem()
        nd_sem = df["Number of Droplets"].sem()
        chains_largest_sem = df["Chains in Largest Droplet"].sem()
        mass_largest_sem = df["Mass of Largest Droplet (mg)"].sem()
        number_external_sem = df["Number of External Chains"].sem()
        mass_external_sem = df["Mass of External Chains"].sem()

        df_mean = df.mean()
        df = pd.DataFrame(df_mean).transpose()
        df["RG SEM"] = rg_sem
        df["ND SEM"] = nd_sem
        df["Chains Largest SEM"] = chains_largest_sem
        df["Mass Largest SEM"] = mass_largest_sem
        df["NE SEM"] = number_external_sem
        df["ME SEM"] = mass_external_sem
        df.to_csv("{}/{}_AVE/Cluster_{}_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=True)

    def diffusivity_ave_time_aligned(self, sm: str):
        """Average G3BP1 diffusivity curves across chains and windows by aligning on time.

        - Accepts missing chains and non-consecutive windows.
        - Supports files where MSD is in um^2 ("MSD (um)") or m^2 ("MSD (m^2)").
        - Preferred input location is the staged copies in `<base>/ANALYSIS_*_AVE/DIFFUSIVITY`,
          but will fall back to the raw `ANALYSIS_*` directory if needed.
        """
        cat = sm.split("_")[0].upper()
        # Candidate sources (prefer collected copies)
        staged = os.path.join(self.path, f"ANALYSIS_{cat}_AVE", "DIFFUSIVITY")
        patterns = []
        if os.path.isdir(staged):
            patterns.extend([
                os.path.join(staged, f"G3BP1_Diffusivity_{sm}_*_*.csv"),
                os.path.join(staged, f"ProteinG3BP1_Diffusivity_{sm}_*_*.csv"),
            ])
        raw_cat = f"ANALYSIS_{cat}"
        patterns.extend([
            os.path.join(raw_cat, f"G3BP1_Diffusivity_{sm}_*_*.csv"),
            os.path.join(raw_cat, f"ProteinG3BP1_Diffusivity_{sm}_*_*.csv"),
        ])

        files = []
        for pat in patterns:
            files.extend(glob(pat))
        files = sorted(set(files))
        if not files:
            print(f"No diffusivity files found for {sm}; looked in {staged} and {raw_cat}")
            return

        frames = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            cols = df.columns
            # Time in seconds
            if "Time (s)" in cols:
                t = pd.to_numeric(df["Time (s)"], errors="coerce")
            elif "Time (ns)" in cols:
                t = pd.to_numeric(df["Time (ns)"], errors="coerce") * 1e-9
            else:
                # If no recognizable time, skip
                continue
            # MSD in m^2 (keep SI units throughout pipeline)
            if "MSD (m^2)" in cols:
                msd = pd.to_numeric(df["MSD (m^2)"], errors="coerce")
            elif "MSD (um^2)" in cols:
                # Legacy: convert um^2 -> m^2 (1 m = 1e6 um; 1 m^2 = 1e12 um^2)
                msd = pd.to_numeric(df["MSD (um^2)"], errors="coerce") * 1e-12
            elif "MSD (um)" in cols:
                # Legacy: assume um^2 despite ambiguous header
                msd = pd.to_numeric(df["MSD (um)"], errors="coerce") * 1e-12
            else:
                continue
            # Optional Rg
            rg = pd.to_numeric(df["Rg"], errors="coerce") if "Rg" in cols else pd.Series(index=df.index, dtype=float)

            cur = pd.DataFrame({"Time (s)": t, "MSD (m^2)": msd, "Rg": rg})
            # Exclude zeros (missing) from averages later
            cur.loc[cur["MSD (m^2)"] == 0, "MSD (m^2)"] = np.nan
            cur.loc[cur["Rg"] == 0, "Rg"] = np.nan
            frames.append(cur)

        if not frames:
            print(f"All diffusivity files unreadable for {sm}")
            return

        all_df = pd.concat(frames, ignore_index=True)
        # Round time to avoid floating drift when grouping; keep sub-ns resolution
        all_df["Time_round"] = all_df["Time (s)"].round(12)
        grouped = all_df.groupby("Time_round")
        mean = grouped.mean(numeric_only=True)
        sem = grouped.sem(numeric_only=True, ddof=1)
        count = grouped.count()

        out = pd.DataFrame({
            "Time (s)": mean.index.values,
            "MSD (m^2)": mean["MSD (m^2)"].values,
            "MSD_SEM": sem["MSD (m^2)"].values,
            "Rg": mean.get("Rg", pd.Series(index=mean.index, dtype=float)).values,
            "Rg_SEM": sem.get("Rg", pd.Series(index=sem.index, dtype=float)).values,
            "N": count["MSD (m^2)"].values,
        }).sort_values("Time (s)")

        out_path = "{}/{}_AVE/G3BP1_Diffusivity_{}.csv".format(self.path, self.category, sm)
        out.to_csv(out_path, index=False)

    def pca_ave(self, biopolymer, sm):
        """Average per-window PCA shape descriptors for ``biopolymer``/``sm`` over all windows into a single row."""
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        for i in range(self.tmin,self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/PCA_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("PCA_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))
                continue
            mean_values = pd.read_csv(input_path).mean()
            df_list.append(mean_values)
            index_list.append(i)
        if not df_list:
            print("No PCA data found for {} {} in {}".format(biopolymer, sm, category))
            return
        df = pd.DataFrame(df_list, index=index_list)
        df.to_csv("{}/{}_AVE/PCA_{}_{}.csv".format(self.path,self.category, biopolymer, sm), index=False, header=True)

    def collect_diffusion(self, prefix, sm):
        """Copy the in-window G3BP1 diffusivity CSVs for system ``sm`` from the raw folder into the ``_AVE/DIFFUSIVITY`` staging folder."""
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        if self.tmin > 0:
            tmin = self.tmin - self.dt
        else:
            tmin = self.tmin
        if self.tmax < 2000:
            tmax = self.tmax + self.dt
        else:
            tmax = self.tmax

        diff_list = list(np.arange(tmin,tmax,self.dt))

        for filename in os.listdir(category):
            # Only consider G3BP1 diffusivity CSV files for this SM
            if "Diffusivity" not in filename or "G3BP1" not in filename or sm not in filename:
                continue

            source_file = os.path.join(category, filename)
            if not os.path.isfile(source_file):
                continue

            # Robustly extract the timestep from any integer token in the filename.
            # This avoids relying on a fixed position, which breaks for SM names with
            # extra underscores (e.g., dsm_dihydrolipoic_acid).
            base = os.path.splitext(filename)[0]
            timestep = None
            for token in base.split("_"):
                try:
                    val = int(token)
                except ValueError:
                    continue
                if val in diff_list:
                    timestep = val
                    break

            # Skip files whose timestep is outside the requested window
            if timestep is None:
                continue

            dest_file = os.path.join("{}/{}_AVE/DIFFUSIVITY".format(self.path,self.category), filename)
            shutil.copy2(source_file, dest_file)


class aggregate:
    """Per-class aggregator: averages the per-system ``_AVE`` products into one DSM / NDSM class result.

    Consumes the per-system block averages written by :class:`average` and
    combines them across the small molecules in a class, writing a single
    aggregate file (and SEM across systems) per observable.
    """

    def __init__(self, path, category, tmin=None):
        """Store the dataset root and class ``category`` (e.g. ``DSM``); ``tmin`` is an optional hint for locating stress files."""
        self.df = pd.DataFrame()
        self.path=path
        self.category = category
        # Optional time window start; used only as a hint
        # when looking for per-SM Stress_Tensor files.
        self.tmin = tmin

    def rdp_ave(self, biopolymer, sm_list):
        """Average the per-system density profiles of ``biopolymer`` across ``sm_list`` into one class profile with cross-system SEM."""
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            distance_col = (pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "Protein density (mg/mL)"])
            df_temp[str(sm)]=density_col

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False)

    def contact_ave(self, biopolymer, sm_list):
        """Average the per-system mean ``biopolymer`` contact matrices across ``sm_list`` into one class matrix with cross-system SEM."""
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm_list[0]), header=None)
        mats = [df.to_numpy(dtype=float)]
        base_shape = mats[0].shape
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm), header=None)
                arr_temp = df_temp.to_numpy(dtype=float)
                if arr_temp.shape != base_shape:
                    print("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv shape mismatch (skipping)".format(self.path, self.category, biopolymer, sm))
                    continue
                df = df.add(df_temp, fill_value=0)
                mats.append(arr_temp)
                count += 1
            except:
                print("{}/ANALYSIS_{}_Contacts_Mean_{}.csv File Missing".format(self.path, biopolymer, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/{}_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=False)
        pd.DataFrame(_matrix_sem(mats)).to_csv(
            "{}/ANALYSIS_{}_AVE/{}_Contacts_SEM_{}.csv".format(self.path, self.category, biopolymer, self.category),
            index=False,
            header=False,
        )

    def contact_sm_ave(self, biopolymer, sm_list):
        """Average the per-system mean ``biopolymer``-SM contact matrices across ``sm_list`` into one class matrix with cross-system SEM."""
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm_list[0]), header=None)
        mats = [df.to_numpy(dtype=float)]
        base_shape = mats[0].shape
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, sm), header=None)
                arr_temp = df_temp.to_numpy(dtype=float)
                if arr_temp.shape != base_shape:
                    print("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv shape mismatch (skipping)".format(self.path, self.category, biopolymer, sm))
                    continue
                df = df.add(df_temp, fill_value=0)
                mats.append(arr_temp)
                count += 1
            except:
                print("{}/ANALYSIS_{}_SM_Contacts_Mean_{}_{}.csv File Missing".format(self.path, self.category, biopolymer, sm))

        df = df.div(count, fill_value=0)
        df.to_csv("{}/ANALYSIS_{}_AVE/{}_SM_Contacts_Mean_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=False)
        pd.DataFrame(_matrix_sem(mats)).to_csv(
            "{}/ANALYSIS_{}_AVE/{}_SM_Contacts_SEM_{}.csv".format(self.path, self.category, biopolymer, self.category),
            index=False,
            header=False,
        )

    def cluster_ave(self, biopolymer, sm_list):
        """Average the per-system cluster-geometry rows for ``biopolymer`` across ``sm_list`` into one class row with cross-system SEM columns."""
        df_list = []
        index_list = []
        df_cols = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm_list[0])).columns.tolist()
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)

        df = pd.DataFrame(columns=df_cols, data=df_list, index=index_list)
        rg_sem = df["Largest Droplet Radius of Gyration"].sem()
        nd_sem = df["Number of Droplets"].sem()
        chains_largest_sem = df["Chains in Largest Droplet"].sem()
        mass_largest_sem = df["Mass of Largest Droplet (mg)"].sem()
        number_external_sem = df["Number of External Chains"].sem()
        mass_external_sem = df["Mass of External Chains"].sem()

        df_mean = df.mean()
        df = pd.DataFrame(df_mean).transpose()
        df["RG SEM"] = rg_sem
        df["ND SEM"] = nd_sem
        df["Chains Largest SEM"] = chains_largest_sem
        df["Mass Largest SEM"] = mass_largest_sem
        df["NE SEM"] = number_external_sem
        df["ME SEM"] = mass_external_sem
        df.to_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)

    def pca_ave(self, biopolymer, sm_list):
        """Average the per-system PCA shape-descriptor rows for ``biopolymer`` across ``sm_list`` into one class table."""
        df_list = []
        index_list = []
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/PCA_{}_{}.csv".format(self.path, self.category, biopolymer, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)
        df = pd.DataFrame(df_list, index=index_list)
        df.to_csv("{}/ANALYSIS_{}_AVE/PCA_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)

    def stress_ave(self, sm_list):
        """
        Average Stress_Tensor_* across small molecules, reading from ANALYSIS_<CAT>
        (do not copy into _AVE). Writes aggregated result to ANALYSIS_<CAT>_AVE/Stress_Tensor_<CAT>.csv.
        """
        df_xx = pd.DataFrame()
        df_yy = pd.DataFrame()
        df_zz = pd.DataFrame()
        df_xy = pd.DataFrame()
        df_xz = pd.DataFrame()
        df_yz = pd.DataFrame()
        df = pd.DataFrame()
        time_col = None
        category_base = f"ANALYSIS_{self.category}"

        for sm in sm_list:
            src = None
            # Prefer an explicit tmin-stamped file if present, but accept either
            # integer or floating-point labels written by RCC analysis.
            if self.tmin is not None:
                src = _resolve_time_file(
                    lambda label: os.path.join(category_base, f"Stress_Tensor_{sm}_{label}.csv"),
                    self.tmin,
                )
            if src is None:
                src = _resolve_time_file(
                    lambda label: os.path.join(category_base, f"Stress_Tensor_{sm}_{label}.csv"),
                    0,
                )
            if src is None:
                legacy_plain = os.path.join(category_base, f"Stress_Tensor_{sm}.csv")
                src = legacy_plain if os.path.isfile(legacy_plain) else None
            if src is None:
                print(f"Stress_Tensor file missing for {sm} (skipping)")
                continue

            df_src = pd.read_csv(src)
            if time_col is None and "Timestep" in df_src.columns:
                time_col = df_src["Timestep"]

            for col, store in [
                ("Pxx", df_xx),
                ("Pyy", df_yy),
                ("Pzz", df_zz),
                ("Pxy", df_xy),
                ("Pxz", df_xz),
                ("Pyz", df_yz),
            ]:
                if col in df_src.columns:
                    store[f"{col}_{sm}"] = df_src[col]

        if time_col is None or df_xx.empty:
            print(f"No stress data averaged for category {self.category}")
            return

        pxx_avg, pxx_sem = df_xx.mean(axis=1), df_xx.sem(axis=1)
        pyy_avg, pyy_sem = df_yy.mean(axis=1), df_yy.sem(axis=1)
        pzz_avg, pzz_sem = df_zz.mean(axis=1), df_zz.sem(axis=1)
        pxy_avg, pxy_sem = df_xy.mean(axis=1), df_xy.sem(axis=1)
        pxz_avg, pxz_sem = df_xz.mean(axis=1), df_xz.sem(axis=1)
        pyz_avg, pyz_sem = df_yz.mean(axis=1), df_yz.sem(axis=1)

        df["Timestep"] = time_col
        df["Pxx"] = pxx_avg
        df["Pyy"] = pyy_avg
        df["Pzz"] = pzz_avg
        df["Pxy"] = pxy_avg
        df["Pxz"] = pxz_avg
        df["Pyz"] = pyz_avg
        df["Pxx Sem"] = pxx_sem
        df["Pyy Sem"] = pyy_sem
        df["Pzz Sem"] = pzz_sem
        df["Pxy Sem"] = pxy_sem
        df["Pxz Sem"] = pxz_sem
        df["Pyz Sem"] = pyz_sem

        df.to_csv("{}/ANALYSIS_{}_AVE/Stress_Tensor_{}.csv".format(self.path, self.category, self.category),
                  index=False)

    def collect_diffusion(self, sm_list):
        """
        Write aggregate RCC diffusion inputs under the raw ANALYSIS_<CAT> folder.

        DSM/NDSM aggregate rows later request these raw RCC artifacts during the
        confined-diffusion stage, so they must exist before SYSTEM_ANALYSIS runs.
        """
        if not sm_list:
            print(f"[aggregate_rdp] No systems available for {self.category}; skipping.")
            return
        temp_root = os.path.abspath(os.getcwd())
        generate_aggregate_rdp_inputs(temp_root, self.category, sm_list, aggregate_tag=self.category)




class average_biopolymers():
    """Per-species density-profile averager: builds the per-biopolymer and composite RDP products.

    Averages each individual biopolymer species' radial density profile over
    time windows, averages those across the systems in a class, and reconstructs
    the composite Protein / RNA / total-SG profiles by additive combination.
    """

    def __init__(self, path, category, tmin, tmax, dt):
        """Store the dataset root, the ``_AVE`` output category, and the [tmin, tmax) window with stride ``dt``."""
        self.path = path
        self.category=category
        self.df = pd.DataFrame()
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        """Average the per-window density profile of a single species ``biopolymer`` for system ``sm`` into the ``_AVE`` folder."""
        category = sm.split("_")[0].upper()
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        distance_col = None
        for i in range(self.tmin, self.tmax, self.dt):
            input_path = _resolve_time_file(
                lambda label: "ANALYSIS_{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("ANALYSIS_{}/DensityProfile_{}_{}_{}.csv File Missing".format(category, biopolymer, sm, str(i)))
                continue
            cur_df = pd.read_csv(input_path)
            distance_col = cur_df["Distance from center of mass (A)"]
            if biopolymer == "SM":
                density_col = list(cur_df["SM density (mg/mL)"])
            else:
                density_col = list(cur_df["Protein density (mg/mL)"])
            df_temp[str(i)] = density_col

        if distance_col is None or df_temp.empty:
            print("No DensityProfile data found for {} {} in ANALYSIS_{}".format(biopolymer, sm, category))
            return

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        if sm == "sg_X":
            sm = "SG"
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def rdp_sm_ave(self, biopolymer, sm_list, sm_type):
        """Average the per-system density profiles of species ``biopolymer`` across ``sm_list`` into a ``sm_type`` class profile."""
        df_temp = pd.DataFrame()
        df_sem_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            input_df = pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))
            distance_col = input_df["Distance from center of mass (A)"]
            if biopolymer == "SM":
                density_col = list(
                    input_df["SM density (mg/mL)"])
            else:
                density_col = list(
                    input_df["Protein density (mg/mL)"])
            df_temp[str(sm)] = density_col
            if "Standard mean error" in input_df.columns:
                df_sem_temp[str(sm)] = pd.to_numeric(input_df["Standard mean error"], errors="coerce")
        row_avg = df_temp.mean(axis=1)
        if len(sm_list) == 1 and not df_sem_temp.empty:
            row_sem = df_sem_temp.iloc[:, 0]
        else:
            row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm_type), index=False)

    def gen_ave(self, sm_list, sm_type):
        """Run :meth:`rdp_ave` for every species over every system in ``sm_list`` (per-system per-species window averaging)."""
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for sm in sm_list:
            for biopolymer in biopolymer_list:
                self.rdp_ave(biopolymer, sm)

    def gen_agg(self, sm_list, sm_type):
        """Run :meth:`rdp_sm_ave` for every species across ``sm_list`` to build ``sm_type`` class profiles (skipped for the SG control)."""
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for biopolymer in biopolymer_list:
            if sm_type != "SG":
                self.rdp_sm_ave(biopolymer,sm_list,sm_type)

    def additive_conc(self, sm):
        """Reconstruct composite Protein, RNA, and total-SG density profiles for ``sm`` by summing the per-species profiles (errors added in quadrature)."""
        category = sm.split("_")[0].upper()
        # Read from ANALYSIS_*_AVE instead of BIOPOLYMER_ANALYSIS_*
        ave_folder = "ANALYSIS_{}_AVE".format(category)
        
        distances = np.array((pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, ave_folder, "RNA", sm))[
            "Distance from center of mass (A)"]))

        rna_density = np.array(
            pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, ave_folder, "RNA", sm))[
                "Protein density (mg/mL)"])
        rna_sme = np.array(
            pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, ave_folder, "RNA", sm))[
                "Standard mean error"])

        protein_density = np.zeros(rna_density.shape)
        protein_sme = np.zeros(rna_sme.shape)

        df = pd.DataFrame()

        protein_list = ["G3BP1", "FUS", "PABP1", "TDP43", "TIA1", "TTP"]
        for protein in protein_list:
            try:
                density_col = np.array(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, ave_folder, protein, sm))[
                        "Protein density (mg/mL)"])
                error_col = np.array(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, ave_folder, protein, sm))[
                        "Standard mean error"])
                protein_density = protein_density + density_col
                protein_sme = protein_sme + error_col ** 2
            except:
                print("Density_Profile_{}_{}.csv File Missing".format(protein, sm))

        protein_sme = np.sqrt(protein_sme)

        sg_density = protein_density + rna_density
        sg_sme = np.sqrt(protein_sme ** 2 + rna_sme ** 2)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = sg_density
        df_sg["Standard mean error"] = sg_sme
        df_sg.to_csv("{}/{}/Density_Profile_SG_{}.csv".format(self.path, self.category, sm), index=False)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = protein_density
        df_sg["Standard mean error"] = protein_sme
        df_sg.to_csv("{}/{}/Density_Profile_Protein_{}.csv".format(self.path, self.category, sm), index=False)

        df_sg = pd.DataFrame()
        df_sg["Distance from center of mass (A)"] = distances
        df_sg["Protein density (mg/mL)"] = rna_density
        df_sg["Standard mean error"] = rna_sme
        df_sg.to_csv("{}/{}/Density_Profile_RNA_{}.csv".format(self.path, self.category, sm), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Average MD simulation data across time windows and replicates')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', default='CLASSIFY', help='Output folder prefix (default: CLASSIFY)')
    parser.add_argument('--temp', type=int, required=True, help='Temperature in Kelvin (e.g., 300)')
    parser.add_argument('--tmin', type=int, required=True, help='Start frame for time averaging')
    parser.add_argument('--dt', type=int, required=True, help='Frame stride for time averaging')
    parser.add_argument('--tmax', type=int, required=True, help='End frame for time averaging')
    parser.add_argument('--use-lists', action='store_true',
                        help='Use dsm_list.txt and ndsm_list.txt from output directory instead of default SM lists')
    parser.add_argument('--plot-only', action='store_true', help='No-op for this averaging script; exits after verifying the existing output root')

    args = parser.parse_args()

    # Validate path exists
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    args.path = os.path.abspath(args.path)
    if args.plot_only:
        output_root = os.path.join(args.path, f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}")
        if not os.path.isdir(output_root):
            print(f"Error: Analysis root {output_root} does not exist; run full averaging first.")
            sys.exit(1)
        print(f"[plot-only] AVERAGE_SIMULATIONS has no plot products; existing root verified: {output_root}")
        sys.exit(0)

    # Check for input directories and warn if missing
    for cat in ['SG', 'DSM', 'NDSM']:
        input_dir = os.path.join(args.path, f"ANALYSIS_{cat}")
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found - {cat} analysis will be skipped")

    # Start working from the dataset root so all relative reads resolve
    os.chdir(args.path)

    def gen_path(path, folder):
        """Create a fresh ``path/folder`` directory, removing and recreating it if it already exists."""
        full_path = "{}/{}".format(path,folder)
        if not os.path.exists(full_path):
            # Create the folder
            os.makedirs(full_path)
        else:
            shutil.rmtree(full_path)
            os.makedirs(full_path)

    def gen_avg(path, dt, start, end, dsm_names, ndsm_names):
        """Top-level driver: run all per-system and per-class averages for SG, DSM, and NDSM and create the figure/results scaffold.

        ``path`` is the analysis output root, ``start``/``end``/``dt`` define the
        time window, and ``dsm_names``/``ndsm_names`` are the small-molecule
        system lists; missing input categories are skipped with a warning.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)

        # SG
        print("SG BLOCK AVERAGES")
        folder = "ANALYSIS_SG_AVE"
        gen_path(path, folder)

        ave = average(path,"ANALYSIS_SG", start, end, dt)

        ave.rdp_ave("Protein", "sg_X")
        ave.rdp_ave("RNA", "sg_X")
        ave.rdp_ave("SG", "sg_X")

        ave.contact_ave("Acid", "sg_X")
        ave.contact_ave("Residue", "sg_X")

        ave.cluster_ave("RNA", "sg_X")
        ave.cluster_ave("Protein", "sg_X")
        ave.cluster_ave("SG", "sg_X")

        ave.pca_ave("SG", "sg_X")
        ave.pca_ave("Protein", "sg_X")
        ave.pca_ave("RNA", "sg_X")

        gen_path(path, "{}/DIFFUSIVITY".format(folder))

        ave.collect_diffusion("G3BP1", "sg_X")
        ave.diffusivity_ave_time_aligned("sg_X")
        print("SG BLOCK AVERAGES SUCCESSFUL")

        # DSM (skip if input folder missing)
        if os.path.isdir("ANALYSIS_DSM"):
            folder = "ANALYSIS_DSM_AVE"
            gen_path(path, folder)

            ave = average(path, "ANALYSIS_DSM", start, end, dt)

            print("DSM BLOCK AVERAGES")

            gen_path(path, "{}/DIFFUSIVITY".format(folder))

            for i in dsm_names:
                ave.rdp_ave("Protein", i)
                ave.rdp_ave("RNA", i)
                ave.rdp_ave("SG", i)
                ave.rdp_ave("SM", i)

                ave.contact_ave("Acid", i)
                ave.contact_ave("Residue", i)
                ave.contact_sm_ave("Residue", i)
                ave.contact_sm_ave("Acid", i)

                ave.cluster_ave("RNA", i)
                ave.cluster_ave("Protein", i)
                ave.cluster_ave("SG", i)

                ave.pca_ave("SG", i)
                ave.pca_ave("Protein", i)
                ave.pca_ave("RNA", i)

                ave.collect_diffusion("G3BP1", i)
                ave.diffusivity_ave_time_aligned(i)

            print("DSM AGGREGATE")
            ave = aggregate(path, "DSM", tmin=start)

            ave.rdp_ave("SG", dsm_names)
            ave.rdp_ave("RNA", dsm_names)
            ave.rdp_ave("Protein", dsm_names)
            ave.rdp_ave("SM", dsm_names)

            ave.contact_ave("Acid", dsm_names)
            ave.contact_ave("Residue", dsm_names)
            ave.contact_sm_ave("Residue", dsm_names)
            ave.contact_sm_ave("Acid", dsm_names)

            ave.cluster_ave("RNA", dsm_names)
            ave.cluster_ave("Protein", dsm_names)
            ave.cluster_ave("SG", dsm_names)

            ave.pca_ave("SG", dsm_names)
            ave.pca_ave("Protein", dsm_names)
            ave.pca_ave("RNA", dsm_names)

            ave.stress_ave(dsm_names)

            ave.collect_diffusion(dsm_names)

            print("DSM BLOCK AVERAGES SUCCESSFUL")
        else:
            print("Skipping DSM block: ANALYSIS_DSM not found")

        # NDSM (skip if input folder missing)
        if os.path.isdir("ANALYSIS_NDSM"):
            folder = "ANALYSIS_NDSM_AVE"
            gen_path(path, folder)

            ave = average(path, "ANALYSIS_NDSM", start, end, dt)

            print("NDSM BLOCK AVERAGES")

            gen_path(path, "{}/DIFFUSIVITY".format(folder))

            for i in ndsm_names:
                ave.rdp_ave("Protein", i)
                ave.rdp_ave("RNA", i)
                ave.rdp_ave("SG", i)
                ave.rdp_ave("SM", i)

                ave.contact_ave("Acid", i)
                ave.contact_ave("Residue", i)
                ave.contact_sm_ave("Residue", i)
                ave.contact_sm_ave("Acid", i)

                ave.cluster_ave("RNA", i)
                ave.cluster_ave("Protein", i)
                ave.cluster_ave("SG", i)

                ave.pca_ave("SG", i)
                ave.pca_ave("Protein", i)
                ave.pca_ave("RNA", i)

                ave.collect_diffusion("G3BP1", i)
                ave.diffusivity_ave_time_aligned(i)

            print("NDSM AGGREGATE")
            ave = aggregate(path, "NDSM", tmin=start)

            ave.rdp_ave("SG", ndsm_names)
            ave.rdp_ave("RNA", ndsm_names)
            ave.rdp_ave("Protein", ndsm_names)
            ave.rdp_ave("SM", ndsm_names)

            ave.contact_ave("Acid", ndsm_names)
            ave.contact_ave("Residue", ndsm_names)
            ave.contact_sm_ave("Residue", ndsm_names)
            ave.contact_sm_ave("Acid", ndsm_names)

            ave.cluster_ave("RNA", ndsm_names)
            ave.cluster_ave("Protein", ndsm_names)
            ave.cluster_ave("SG", ndsm_names)

            ave.pca_ave("SG", ndsm_names)
            ave.pca_ave("Protein", ndsm_names)
            ave.pca_ave("RNA", ndsm_names)

            ave.stress_ave(ndsm_names)

            ave.collect_diffusion(ndsm_names)
            
            print("NDSM BLOCK AVERAGES SUCCESSFUL")
        else:
            print("Skipping NDSM block: ANALYSIS_NDSM not found")

    
        # BIOPOLYMER
        print("SG BIOPOLYMER AVERAGES")
        
        # Create main folders and all subfolders first
        gen_path(path, "FIGURES")
        gen_path(path, "FIGURES/RDP")
        gen_path(path, "FIGURES/RESIDUE_CONTACT_MAPS")
        gen_path(path, "FIGURES/ACID_CONTACT_MAPS")
        gen_path(path, "FIGURES/SM_CONTACT_MAPS")
        gen_path(path, "FIGURES/PROPERTIES")
        gen_path(path, "FIGURES/TIME")
        
        gen_path(path, "IMAGES")
        gen_path(path, "IMAGES/RDP")
        gen_path(path, "IMAGES/RESIDUE_CONTACT_MAPS")
        gen_path(path, "IMAGES/ACID_CONTACT_MAPS")
        gen_path(path, "IMAGES/SM_CONTACT_MAPS")
        gen_path(path, "IMAGES/DYNAMICS")
        
        gen_path(path, "RESULTS")
        gen_path(path, "RESULTS/RDP")
        gen_path(path, "RESULTS/RESIDUE_CONTACT_MAPS")
        gen_path(path, "RESULTS/ACID_CONTACT_MAPS")
        gen_path(path, "RESULTS/SM_CONTACT_MAPS")
        gen_path(path, "RESULTS/SUMMARY")
        
        # Create folders for MaxCluster and biopolymer analysis (only if they don't exist)
        for folder in ["ANALYSIS_SG_AVE", "ANALYSIS_DSM_AVE", "ANALYSIS_NDSM_AVE"]:
            folder_path = "{}/{}".format(path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # BIOPOLYMER files are now created directly in ANALYSIS_*_AVE folders
        try:
            print("SG BIOPOLYMER AVERAGES")
            sm = ["sg_X"]
            ave_bio_sg = average_biopolymers(path, "ANALYSIS_SG_AVE", start, end, dt)
            ave_bio_sg.gen_ave(sm,"SG")
            ave_bio_sg.gen_agg(sm, "SG")
            ave_bio_sg.additive_conc("SG")
            print("SG BIOPOLYMER AVERAGES SUCCESSFUL")
        except:
            print("SG BIOPOLYMER AVERAGES FAILED")

        if os.path.isdir(f"{path}/ANALYSIS_DSM_AVE"):
            try:
                print("DSM BIOPOLYMER AVERAGES")
                ave_bio_dsm = average_biopolymers(path, "ANALYSIS_DSM_AVE", start, end, dt)
                ave_bio_dsm.gen_ave(dsm_names, "DSM")
                ave_bio_dsm.gen_agg(dsm_names, "DSM")
                ave_bio_dsm.additive_conc("DSM")
                print("DSM BIOPOLYMER AVERAGES SUCCESSFUL")
            except:
                print("DSM BIOPOLYMER AVERAGES FAILED")
        else:
            print("Skipping DSM biopolymer averages: ANALYSIS_DSM_AVE not found")


        if os.path.isdir(f"{path}/ANALYSIS_NDSM_AVE"):
            try:
                print("NDSM BIOPOLYMER AVERAGES")
                ave_bio_ndsm = average_biopolymers(path, "ANALYSIS_NDSM_AVE", start, end, dt)
                ave_bio_ndsm.gen_ave(ndsm_names, "NDSM")
                ave_bio_ndsm.gen_agg(ndsm_names, "NDSM")
                ave_bio_ndsm.additive_conc("NDSM")
                print("NDSM BIOPOLYMER AVERAGES SUCCESSFUL")
            except:
                print("NDSM BIOPOLYMER AVERAGES FAILED")
        else:
            print("Skipping NDSM biopolymer averages: ANALYSIS_NDSM_AVE not found")


    # Output directory: TEMP_XXX/{folder}_{temp}_{dt}_{tmin}_{tmax}
    output_path = "{}_{}_{}_{}_{}" .format(args.folder, args.temp, args.dt, args.tmin, args.tmax)

    # SM lists
    if args.use_lists:
        # Load from files (for custom experiments)
        dsm_list = []
        ndsm_list = []
        if os.path.exists(f'{output_path}/dsm_list.txt'):
            with open(f'{output_path}/dsm_list.txt', 'r') as f:
                dsm_list = [line.strip() for line in f.readlines()]
        if os.path.exists(f'{output_path}/ndsm_list.txt'):
            with open(f'{output_path}/ndsm_list.txt', 'r') as f:
                ndsm_list = [line.strip() for line in f.readlines()]
    else:
        # Default SM lists (updated names; legacy variants still handled elsewhere)
        dsm_list = [
            "dsm_anisomycin",
            "dsm_daunorubicin",
            "dsm_dihydrolipoic_acid",
            "dsm_hydroxyquinoline",
            "dsm_lipoamide",
            "dsm_lipoic_acid",
            "dsm_mitoxantrone",
            "dsm_pararosaniline",
            "dsm_pyrivinium",
            "dsm_quinicrine",
        ]
        ndsm_list = [
            "ndsm_dmso",
            "ndsm_valeric_acid",
            "ndsm_ethylenediamine",
            "ndsm_propanedithiol",
            "ndsm_hexanediol",
            "ndsm_diethylaminopentane",
            "ndsm_aminoacridine",
            "ndsm_anthraquinone",
            "ndsm_acetylenapthacene",
            "ndsm_anacardic",
        ]

    print("Processing all categories: SG, DSM, NDSM")
    gen_avg(output_path, args.dt, args.tmin, args.tmax, dsm_list, ndsm_list)
