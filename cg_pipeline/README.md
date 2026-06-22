# CG Parameterisation Toolkit

A reproducible toolkit for deriving coarse-grained (MPiPi-style) force-field
parameters for small molecules, extending the protein/RNA parameterisation of
Joseph & Joseph. It predicts homotypic interaction parameters from molecular
descriptors, benchmarks mixing rules for the heterotypic interactions, compares
the result against reference parameters, and assembles a LAMMPS system.

## Contents

```
cg_pipeline/
├── run_full_pipeline.py     # CLI entry point that orchestrates every step
├── qspr.py                  # Regressor: Mordred descriptors -> random-forest parameter model
├── mix_params.py             # Mixer: Lorentz-Berthelot / Waldman-Hagler / Fender-Halsey / Kong
├── gen_files.py              # Settings: write sys.data, sys.settings, mix_params_JK.txt
├── gen_bonds.py              # bond-length helper from coarse-grained structures
├── convert_cg.py             # PDB/map -> LAMMPS molecule (.mol) conversion
├── requirements.txt, LICENSE, README.md, environment.yml
└── data/                    # committed parameter & reference data (CSV / txt / json):
                             #   MPiPi_Parameters.csv, MPiPi_Molecules.csv, molecules.csv,
                             #   WF_Parameters.txt, mix_params_JJ.txt, SM_SM_Parameters_JJ,
                             #   bonds.txt, masses.txt, features.txt, mixing_rules.txt,
                             #   drug_parameters.csv, R2_Dataset.csv, random_forest_columns.json
```

`cg_pipeline/` is pure parameterisation/build compute and contains **no figure
rendering**. The parameterisation figures live under `plotting/cg/`:

```
plotting/cg/
├── parameter_analysis.py     # Wang-Frenkel homotypic interaction curves
├── compare_parameters.py     # parity_comparer: JJ (reference) vs JK (this work) parity plots
├── plot_parity_from_files.py # standalone CLI wrapper around parity_comparer
└── r2_analysis.py            # QSPR R^2 / MAE vs number of descriptors
```

Generated outputs (`sys.data`, `sys.settings`, `mix_params_JK.txt`,
`single_parameters.csv`, `RUNS/`, `*.pkl`, the `*_PDB_Files/` assets …) are
recreated by the pipeline and are excluded from version control via
`.gitignore`. Large legacy data (`GENDATA/`) and the optional `GBCG/` tool are
also excluded — see **Legacy & optional components** below.

## Dependencies

Python 3.10+. Install the core requirements (`requirements.txt` and `environment.yml`
both live in this directory):

```bash
conda install -c conda-forge rdkit mordred numpy pandas scikit-learn matplotlib seaborn scipy
# or, from cg_pipeline/:  pip install -r requirements.txt   (RDKit installs most reliably via conda)
```

Equivalent Conda environment file:

```bash
conda env create -f environment.yml   # from cg_pipeline/
conda activate sg-cg
```

The optional coarse-graining step additionally needs the external GBCG tool and
OpenBabel (see below); the rest of the pipeline does not.

## Quick start (runs out of the box — no GBCG, no SG template)

The committed parameter files under `data/` (`data/R2_Dataset.csv`,
`data/drug_parameters.csv`, `data/mix_params_JJ.txt`, `data/SM_SM_Parameters_JJ`)
are enough to reproduce the diagnostic plots without any external data. From `cg_pipeline/`:

```bash
# QSPR R^2 / MAE-vs-descriptor-count plot
python ../plotting/cg/r2_analysis.py --metrics data/R2_Dataset.csv --out .

# JJ-vs-JK parity plots (all four --*-ref/--*-new args are required;
# mix_params_JK.txt is produced by the full workflow below)
python ../plotting/cg/plot_parity_from_files.py \
    --mix-ref data/mix_params_JJ.txt --mix-new mix_params_JK.txt \
    --sm-ref  data/SM_SM_Parameters_JJ --sm-new  data/drug_parameters.csv \
    --outdir .
```

## Full workflow (requires the SG template + optional GBCG)

```bash
cd cg_pipeline
python run_full_pipeline.py \
    --mpipi-parameters data/MPiPi_Parameters.csv \
    --mpipi-molecules  data/MPiPi_Molecules.csv \
    --new-molecules    data/molecules.csv \
    --wf-parameters    data/WF_Parameters.txt \
    --template-sys-data <SG_template>.data \
    --bonds            data/bonds.txt \
    --reference-mix    data/mix_params_JJ.txt \
    --reference-sm     data/SM_SM_Parameters_JJ \
    --features         data/features.txt \
    --mix-rules        data/mixing_rules.txt \
    --reuse-coarse-assets \
    --name             JK_RUN
```

The command above performs full system assembly and therefore requires a
stress-granule template `sys.data` (see the note at the end of this section).
`--reuse-coarse-assets` skips the GBCG bead-mapping regeneration; drop it only if you
have cloned GBCG into `./GBCG` and want to rebuild the mappings from scratch.

Steps performed:

1. **QSPR training / prediction** (`qspr.regressor`). A random forest is trained
   on the first 24 entries of `data/MPiPi_Parameters.csv` (amino/nucleic-acid beads),
   using greedy forward selection over Mordred descriptors (or a fixed set via
   `--features`), and predicts parameters for the molecules in `data/molecules.csv`.
   Metrics are written to `R2_Dataset.csv`; the model to
   `random_forest_model.pkl` / `random_forest_columns.json`.
2. **Mixing + system assembly** (`mix_params.mixer` + `gen_files.settings`). The
   best per-parameter mixing rule is selected (or fixed via `--mix-rules`), and
   `sys.data`, `sys.settings`, and `mix_params_JK.txt` are written.
3. **Plots** — Wang-Frenkel homotypic curves (`plotting/cg/parameter_analysis`),
   QSPR R^2 (`plotting/cg/r2_analysis`), and, if `--reference-mix` /
   `--reference-sm` are given, JJ-vs-JK parity plots
   (`plotting/cg/compare_parameters.parity_comparer`). The rendering modules live
   under `plotting/cg/`; `run_full_pipeline.py` invokes them as a convenience
   driver, but `cg_pipeline/` itself stays render-free.
4. **Archival** — inputs, outputs, and plots are copied to
   `RUNS/RUN_<name>_<YYYYMMDD_HHMMSS>/` with a `RUN_INFO.txt`.

### Running individual steps

The QSPR step runs standalone from the included data without an SG template. The
parity and R^2 plotting steps now live under `plotting/cg/` (same commands as
**Quick start** above — `plot_parity_from_files.py` requires all four
`--mix-ref/--mix-new/--sm-ref/--sm-new` arguments):

```bash
python ../plotting/cg/r2_analysis.py --metrics data/R2_Dataset.csv --out .
python ../plotting/cg/plot_parity_from_files.py \
    --mix-ref data/mix_params_JJ.txt --mix-new mix_params_JK.txt \
    --sm-ref  data/SM_SM_Parameters_JJ --sm-new  data/drug_parameters.csv --outdir .
```

`--features` takes blocks of descriptor names headed by the parameter name:

```text
epsilon
ATS0m
ATS1m

sigma
nAtom
Density
```

> Note: `--template-sys-data` (the SG starting structure) is required by
> `run_full_pipeline.py` for the assembly step and is **not** shipped here (it is
> large and system-specific); supply your own or request the SG template. The
> QSPR / parity / R^2 steps above do not need it.

## Module overview

| Module | Class / entry point | Purpose |
|--------|---------------------|---------|
| `qspr.py` | `regressor` | Mordred descriptor calculation, greedy feature selection, random-forest training/prediction, R^2/MAE reporting |
| `mix_params.py` | `mixer` | Evaluate Lorentz-Berthelot, Waldman-Hagler, Fender-Halsey, Kong rules; pick the best per parameter; build the interaction matrix |
| `gen_files.py` | `settings` | Serialise particles + mixed interactions to `sys.data`, `sys.settings`, `mix_params_JK.txt` |
| `gen_bonds.py` / `convert_cg.py` | functions | Bond lengths and PDB/map → LAMMPS `.mol` conversion |
| `run_full_pipeline.py` | CLI | Orchestrates the above and archives the run |
| `../plotting/cg/parameter_analysis.py` | `run_parameter_analysis` | Wang-Frenkel homotypic interaction curves (DSM `dsm_*` vs NDSM `ndsm_*`) |
| `../plotting/cg/compare_parameters.py` | `parity_comparer` | JJ-vs-JK parity (R^2, RMSE%, PCC) for single-molecule and mixed parameters |
| `../plotting/cg/r2_analysis.py` | `run_r2_analysis` | R^2 / MAE vs descriptor count |

## Legacy & optional components

The following are present on disk but excluded from the published repository
(`.gitignore`) because they depend on large external data or third-party tools:

- **GBCG/** — graph-based coarse-graining (Webb group), an *optional* dependency
  used when regenerating bead mappings rather than `--reuse-coarse-assets`. It is **not redistributed
  here**; clone it into `./GBCG` to enable bead-mapping regeneration. The
  pipeline imports it lazily and errors with a clear message if absent.
- **GENDATA/**, **RUNS/** — historical datasets and timestamped run archives
  (regenerated by the pipeline; not required).
- **CG_Script.py, GenSystem.py, GenSG.py** — legacy helpers that depend on
  GENDATA/GBCG and are not part of the documented workflow.

## Citation

If you use this toolkit, please cite the original MPiPi force-field publication
by Joseph & Joseph, the GBCG coarse-graining method if bead mappings are
regenerated, and this repository.
