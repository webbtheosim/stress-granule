# `simulation_inputs/` — full SG / DSM / NDSM simulation inputs (LAMMPS / MPiPi)

Everything needed to reproduce the main coarse-grained MD trajectories — the SG control plus the 10
dissolving (DSM) and 10 non-dissolving (NDSM) small-molecule systems. Each system was launched once
(**STEP 0**, an initial-velocity run of 80 ns: 100k formation steps + 3.9 M production steps at 20 fs)
and continued through 24 restart segments of 80 ns each (4 M steps each) to reach 2 µs total (25 × 80 ns).
We ship the **STEP 0** and **first-resume (STEP 1)** input scripts for every system and temperature, plus
the shared starting configuration, force-field settings, and small-molecule topologies.

This covers the production stress-granule mix used for Figures 2–7. The stand-alone single-protein /
RNA-titration *model* systems behind **Figure 1** live separately under `model_systems/` (see its own
README). Both are produced by `../cg_pipeline/`; both share the same MPiPi force field.

## Layout

```
base_system/
  sg_control/      sys.data, sys.settings   # SG control starting config + MPiPi force field
  small_molecule/  sys.data, sys.settings   # base config used by all DSM and NDSM runs
small_molecule_templates/
  dsm/   dsm_<compound>.mol                  # LAMMPS molecule templates (10 dissolving)
  ndsm/  ndsm_<compound>.mol                 # LAMMPS molecule templates (10 non-dissolving)
lammps_scripts/
  TEMP_<285..315>/
    SG/   initial/  first_resume/            # SG: all 7 temperatures
    DSM/  initial/  first_resume/            # DSM/NDSM: 285, 300, 315 K only
    NDSM/ initial/  first_resume/
```

The 10 DSM and 10 NDSM compounds match the `.mol` templates: DSM = anisomycin, daunorubicin,
dihydrolipoic_acid, hydroxyquinoline, lipoamide, lipoic_acid, mitoxantrone, pararosaniline, pyrivinium,
quinicrine; NDSM = acetylenapthacene, aminoacridine, anacardic, anthraquinone, diethylaminopentane,
dmso, ethylenediamine, hexanediol, propanedithiol, valeric_acid. Inside `initial/` and `first_resume/`,
script names follow `lammps_mpipi_script_sm[_restart]_<dsm|ndsm>_<compound>_1uM[_Step_2].in` (and
`..._X_CONTROL_0uM...` for SG).

Notes:
- `sys.data` and `sys.settings` are **temperature-independent** (one copy each, deduplicated here).
  There are two base systems: the pure SG control (`base_system/sg_control/`), and the small-molecule
  base (`base_system/small_molecule/`, shared by DSM & NDSM). Both are produced by `../cg_pipeline/`.
- The dominant per-temperature change in an input script is the line `variable temperature equal <T>`
  near the top of every `.in` (e.g. `300` in the 300 K scripts); it feeds both the initial
  `velocity all create ${temperature} …` and the `fix … nvt temp ${temperature} ${temperature} …`.
  (A few scripts also carry minor `fix … balance …` load-balancing tweaks — different `every`/`shift`
  values per temperature; these affect only MPI domain decomposition, not the physics.)
- Small molecules are inserted **at run time** by the input script
  (`molecule sm <compound>.mol` defines the template, then
  `create_atoms 1 random 8324 <seed> drug mol sm <seed>` inserts 8324 copies into the box, outside the
  initial condensate sphere) — they are **not** in `sys.data`. The `1uM` token in the filenames is a
  legacy label; the actual inserted concentration is **1 mM** (8324 molecules in the cubic
  2400 Å / 0.24 µm box, ≈1.0×10⁻³ M).

## Running STEP 0 (initial segment)

**No scheduler wrappers are shipped for these production runs** (unlike `model_systems/`, which ships a
`mpipi.slurm`) — submit through your own batch system. The MPiPi pair styles need a LAMMPS build that
includes them (the runs used `lmp_intel` with 96 MPI ranks). A STEP 0 script expects exactly three files
in its working directory — `sys.data`, `sys.settings`, and (for DSM/NDSM) the matching `<compound>.mol`
— because it reads them by bare name (`read_data ${data_name}` → `sys.data`,
`include ${settings_name}` → `sys.settings`, `molecule sm <compound>.mol`).

Run from inside `simulation_inputs/` so the relative paths below resolve as written:

```bash
# Example: dissolving compound "lipoamide" at 300 K  (run from simulation_inputs/)
mkdir -p /scratch/run/SM_lipoamide_300_Step_1 && cd /scratch/run/SM_lipoamide_300_Step_1
cp "$OLDPWD"/base_system/small_molecule/sys.data .
cp "$OLDPWD"/base_system/small_molecule/sys.settings .
cp "$OLDPWD"/small_molecule_templates/dsm/dsm_lipoamide.mol .
cp "$OLDPWD"/lammps_scripts/TEMP_300/DSM/initial/lammps_mpipi_script_sm_dsm_lipoamide_1uM.in .
mpirun "${LAMMPS_BIN:-lmp_intel}" -in lammps_mpipi_script_sm_dsm_lipoamide_1uM.in > out.log 2>&1
```

For an **NDSM** compound, swap `DSM`→`NDSM`, `dsm`→`ndsm`, and the compound name (the `.mol` lives in
`small_molecule_templates/ndsm/`). For the **SG control**, copy from `base_system/sg_control/` and use
`lammps_scripts/TEMP_<T>/SG/initial/lammps_mpipi_script_sm_X_CONTROL_0uM.in` — **no `.mol`** (the SG
control has no small molecules, so it skips the `molecule`/`create_atoms` insertion). SG runs exist for
all 7 temperatures (285–315 K); DSM/NDSM only for 285/300/315 K.

What STEP 0 does, in script order: `read_data` the base box → (DSM/NDSM only) `create_atoms … mol sm`
to insert 8324 small molecules outside the condensate → a short `fix nve/limit` droplet-formation stage
(`run 100000`) → NVT production (`run 3900000`), writing the trajectory + observables below and a
`sim_<compound>_…restart*` file every 1 M steps for resuming.

## Continuing the run (STEP 1 and beyond)

The first-resume script (`first_resume/…restart…_Step_2.in`) does `read_restart ${data_name}` and
continues for another 80 ns (`run 4000000`). Its `data_name` defaults — via a `variable data_name index
sim_<compound>_1uM.restart.4000000` line at the top — to the restart STEP 0 wrote at its final step, so
if you resume in the **same directory** as STEP 0 it just works. It does **not** re-insert small
molecules (they are already in the restart) and reads the force field from `sys.settings` again. To
chain further 80 ns segments: copy the latest `sim_<…>.restart.<step>` and `sys.settings` into a new
directory and override the target on the command line, e.g.
`mpirun "${LAMMPS_BIN:-lmp_intel}" -in <resume>.in -var data_name sim_<…>.restart.<step>`. Repeat
through segment 25 to reach the full 2 µs trajectory.

## What the runs produce / downstream analysis

Each segment writes (filenames carry the `<compound>_<conc>_Step_<n>` tag, e.g. `…_dsm_lipoamide_1uM_Step_1`):

- `result_<…>_Step_<n>.lammpstrj` — unwrapped trajectory (`id mol type q xu yu zu`, every 10000 steps).
- `llps_sg_rg_<…>.out`, `llps_sg_com_<…>.out` — largest-cluster radius-of-gyration and COM time series.
- `stress_msd_<…>.out` — per-atom stress **and** MSD together (columns `c_stress[*] c_msd[*]`, every
  100000 steps); there is no separate `msd_*.out` file — the MSD lives in this dump.
- `sim_<…>.restart*` — restart checkpoints (every 1 M steps) used by the first-resume script.

These feed the analysis pipeline under the repository's `analysis/` directory
(`analysis/rcc_analysis.py` → `analysis/average_simulations.py` → …); see the top-level
`README.md`. In practice the pipeline reads **consolidated** copies of these per-frame products through
the repo's `./PYTHON_ANALYSIS` symlink (created by `setup_data_links.sh`), not the raw run directories.

## Build / LAMMPS

The MPiPi system files (`sys.data`, `sys.settings`) are produced by `../cg_pipeline/`. Simulations were
run with a LAMMPS build that includes the MPiPi pair styles (`wf/cut` + `coul/debye`; the production
runs used `lmp_intel`) with 96 MPI ranks per job. Point `${LAMMPS_BIN}` at your own such build.
