# `model_systems/` — Figure-1 model-system inputs (LAMMPS / MPiPi)

Everything needed to reproduce the single-component and composition-titration condensate runs that
underlie **Figure 1**. These are stand-alone *model* systems (not the full stress-granule mix in
`../base_system/`): each probes one protein, one protein-plus-RNA pair, or one point along the
RNA/protein mass-fraction titration. Each system ships its own complete set of LAMMPS inputs plus a
`pdb/` of the starting structure.

## Layout

```
single_protein_pure/    FUS G3BP1 PABP1 TDP43 TIA1 TTP   # single-protein pure condensates (6)
  <PROTEIN>/  lammps_mpipi_script.in  sys.data  sys.settings  mpipi.slurm
  pdb/        sys_<PROTEIN>_pure.pdb
single_protein_rna/     FUS G3BP1 PABP1 TDP43 TIA1 TTP   # same 6 proteins, each with RNA
  <PROTEIN>/  lammps_mpipi_script.in  sys.data  sys.settings  mpipi.slurm
  pdb/        sys_<PROTEIN>_rna.pdb
protein_rna_titration/  protein_0 ... protein_100        # RNA/protein titration, 0-100 % (11)
  protein_<PCT>/  lammps_mpipi_script.in  sys.data  sys.settings  mpipi.slurm
  pdb/            sys_<PCT>.pdb
```

Each per-system directory holds the four LAMMPS inputs and shares one `PDB/` folder per family with
the starting structures:

- **single_protein_pure** (6) — single-protein-pure systems: each of the six biopolymer proteins
  (FUS, G3BP1, PABP1, TDP43, TIA1, TTP) on its own, no RNA (mass fraction `w_p = 1.0`).
- **single_protein_rna** (6) — single-protein-plus-RNA systems: the same six proteins, each mixed with
  RNA at `w_p = 0.5` (equal protein/RNA mass).
- **protein_rna_titration** (11) — composition-titration series: protein+RNA systems spanning 0, 10, 20,
  30, 40, 50, 60, 70, 80, 90, 100 % protein by mass (`protein_<pct>/`, structure `sys_<pct>.pdb`), the
  RNA/protein mass-fraction sweep behind Figure 1's phase behavior.

## What each input is

- `lammps_mpipi_script.in` — the LAMMPS run script. It pulls the configuration and force field through
  two index variables at the top, `variable data_name index sys.data` and
  `variable settings_name index sys.settings`, then does `read_data ${data_name}` and
  `include ${settings_name}`. It runs two short NVE/`fix limit` + drag droplet-formation stages
  (`run 100000` each) followed by an 8 M-step (160 ns) NVT production segment (`run 8000000`,
  `timestep 20` fs). Outputs written during production:
  - **trajectory** — `result<TAG>.lammpstrj` (unwrapped `xu yu zu`, every 10000 steps).
  - **radius of gyration / COM** of the largest cluster — `llps_rg_<TAG>.out`,
    `llps_rg_tensor_<TAG>.out`, `llps_com_<TAG>.out` (these `compute cluster/atom 20.0` + Rg dumps are
    what the Figure-1 clustering analysis consumes; see "How this feeds Figure 1" below).
  - **stress / MSD** — `stress_msd_<TAG>.out`, `msd_<TAG>.out`. **Present for both single-protein
    families (`single_protein_pure` and `single_protein_rna`); omitted from the `protein_rna_titration`
    titration**, which writes only the trajectory + Rg/COM observables. `<TAG>` is the protein name
    (e.g. `FUS`) or the percentage (e.g. `50`).
- `sys.data` — the MPiPi LAMMPS data file (one bead per residue) for that system's starting box.
- `sys.settings` — the MPiPi force-field include (pair/bond coefficients, masses).
- `mpipi.slurm` — the SLURM submit wrapper (`srun ... lmp_... -in lammps_mpipi_script.in`).

The force-field block is the shared MPiPi baseline and is identical across every system here:
`pair_style hybrid/overlay wf/cut 25.0 coul/debye 0.126 0.0`, `bond_style harmonic`,
`dielectric 80.0`, `special_bonds fene`, `neighbor 3.5 multi`, `neigh_modify every 10 delay 0`,
`comm_style tiled`, `timestep 20` (fs).

## Running a system

Each system directory is **self-contained**: it already holds the three inputs the run needs
(`lammps_mpipi_script.in`, `sys.data`, `sys.settings`) plus its `mpipi.slurm` wrapper. Either submit
the directory as-is with the wrapper, or copy the three inputs into a scratch directory and run LAMMPS
directly. Run the snippet below from inside `model_systems/` so the relative paths resolve:

```bash
# Example: pure FUS condensate  (run from model_systems/)
# Option A — use the shipped SLURM wrapper (point LAMMPS_BIN at your MPiPi build):
cd single_protein_pure/FUS && LAMMPS_BIN=/path/to/lmp_intel sbatch mpipi.slurm

# Option B — run LAMMPS directly in a scratch directory:
mkdir -p /scratch/run/Fig1_FUS_Pure && cd /scratch/run/Fig1_FUS_Pure
cp "$OLDPWD"/single_protein_pure/FUS/{lammps_mpipi_script.in,sys.data,sys.settings} .
srun "${LAMMPS_BIN:-lmp_intel}" -in lammps_mpipi_script.in > output 2>&1
```

The `mpipi.slurm` wrapper requests 1 node × 96 MPI ranks and calls `srun … ${LAMMPS_BIN:-lmp} -in
lammps_mpipi_script.in`, so set `LAMMPS_BIN` to your MPiPi-enabled LAMMPS (it also edits the
`--mail-user` line — set your address or remove it). The script reads the configuration via
`read_data ${data_name}` (→ `sys.data`) and the force field via `include ${settings_name}`
(→ `sys.settings`); no `.mol` small-molecule templates are used for these Figure-1 model systems.

## How this feeds Figure 1

These runs produce the inputs for the Figure-1 clustering panels. After running them (or using their
shipped LAMMPS `CLUSTER/llps_rg_*.out` Rg dumps), the figure is built by the repo's compute→render
modules at the repository root:

| Panel | Systems | Compute (`analysis/`) → Render (`plotting/`) |
|-------|---------|----------------------------------------------|
| C — φ vs protein mass fraction `w_p` (0–100 % titration) | `protein_rna_titration` | `cluster_analysis.py` → `clustering_panels.py` |
| D — per-species φ at `w_p = 0.5` vs `w_p = 1.0` | `single_protein_rna` (0.5) + `single_protein_pure` (1.0) | `cluster_analysis.py` → `clustering_panels.py` |
| A/B — amino-acid / nucleotide composition | sequence data (`analysis/sequences/`) | `sequence_composition.py` → `composition_panels.py` |

`cluster_analysis.py` reads the `compute cluster/atom` Rg dumps (`llps_rg_*.out`) written by the scripts
here and reduces each frame to the number of clusters (`N_D`), the largest-cluster Rg, and the largest-
cluster fraction (`φ`). The composition panels (A/B) do not need these simulations — they count residues
directly from the shipped sequences in `analysis/sequences/`. See the top-level `README.md` figure table.

## Build / LAMMPS

The MPiPi system files (`sys.data`, `sys.settings`) are produced by `../../cg_pipeline/`. Simulations
were run with a LAMMPS build supporting the MPiPi pair styles. The supplied
SLURM wrappers use `${LAMMPS_BIN:-lmp}` so users can point to their own MPiPi
LAMMPS binary, with 96 MPI ranks per job by default. All 23 systems ship the complete four-file set
(`lammps_mpipi_script.in`, `sys.data`, `sys.settings`, `mpipi.slurm`).
