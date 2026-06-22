# Stress-Granule Condensate Simulation & Analysis

**Coarse-grained molecular-dynamics model and analysis pipeline for stress-granule biomolecular condensates and their therapeutic dissolution by small molecules.**

This repository contains the complete simulation inputs, coarse-graining pipeline, and analysis code accompanying the paper:

> **Interactions Underlying Stress Granule Structure and Therapeutic Dissolution**
> Jay L. Kaplan and Michael A. Webb
> Department of Chemical and Biological Engineering, Princeton University
> *Cell Reports Physical Science* (in revision).

---

## Summary

Stress granules are biomolecular condensates composed of RNA and proteins that form in response to
stress; their dysregulation is implicated in neurodegenerative diseases. We develop a minimal
stress-granule model — RNA plus six key proteins associated with neurodegenerative conditions — and
study it with coarse-grained molecular-dynamics simulations using the **MPiPi** force field (one bead
per amino acid / nucleotide). RNA is essential for forming stable condensates, while the underlying
protein–protein interactions produce heterogeneous, multiphasic architectures. We then challenge these
condensates with **twenty distinct small molecules**. Simulation-derived properties classify compounds
as *dissolving* or *non-dissolving* with **85 % agreement** with experiment (leave-one-out
cross-validation). Dissolving compounds disrupt stress-granule structure by preferentially associating
with RNA and stripping the scaffold that maintains its multiphasic architecture.

---

## The model system

| Quantity | Value |
|---|---|
| Force field | MPiPi (residue-resolution; one bead per amino acid / nucleotide) |
| Biopolymer species (7) | TDP43, FUS, TIA1, G3BP1, RNA, PABP1, TTP (**3,594 residues** total) |
| System categories | **SG** (control), **DSM** (10 dissolving small molecules), **NDSM** (10 non-dissolving) |
| Temperatures | 285, 290, 295, 300, 305, 310, 315 K (300 K = reference). SG run at all 7; small molecules at 285 / 300 / 315 K |
| Box | cubic, 2400 Å (0.24 µm) edge |
| Small-molecule concentration | 1 mM |
| Trajectory length | 2 µs per system (25 × 80 ns restart segments) |
| Frame interval / timestep | 0.2 ns / 20 fs (LAMMPS `real` units) |

**Small molecules.**
*DSM (dissolving):* anisomycin, daunorubicin, dihydrolipoic acid, hydroxyquinoline, lipoamide,
lipoic acid, mitoxantrone, pararosaniline, pyrvinium, quinacrine.
*NDSM (non-dissolving):* acetylenaphthacene, aminoacridine, anacardic acid, anthraquinone,
diethylaminopentane, DMSO, ethylenediamine, 1,6-hexanediol, propanedithiol, valeric acid.

---

## Repository layout

```
.
├── analysis/                 Processing pipeline: extracts/averages trajectories and writes ALL
│                             numeric outputs (CSV/NPZ), plus the shared compute modules
│                             (rdp, diffusion, viscosity, acf, ...). Produces the data the figures read.
│   ├── sm/                       Small-molecule isolation/partitioning sub-analysis — COMPUTE side
│   │                             (trajectory->CSV, correlation-corrected stats; SI Fig. 1)
│   └── sequences/                protein sequences + PR-DOS disorder (Fig-1 composition inputs)
├── plotting/                 Figure generators: read the analysis/ outputs and render every paper
│                             panel (contact maps, RDPs, violins, binodal, SI filmstrips, ...).
│   ├── sm/                       Small-molecule isolation/partitioning sub-analysis — FIGURES side
│   │                             (renders SI Fig. 1; imports the compute side from analysis/sm/)
│   └── cg/                       Coarse-graining QSPR / parity diagnostic plots (moved out of cg_pipeline/)
├── cg_pipeline/              Coarse-graining: builds MPiPi LAMMPS systems + parameterizes small molecules
├── simulation_inputs/        LAMMPS inputs to (re)run the simulations — see its own README
│   ├── base_system/              shared starting config + force-field settings (sys.data / sys.settings),
│   │                             one set under sg_control/ and one under small_molecule/
│   ├── small_molecule_templates/ per-compound LAMMPS molecule (.mol) topologies, split into
│   │                             dsm/ (dsm_*.mol) and ndsm/ (ndsm_*.mol)
│   ├── lammps_scripts/TEMP_*/     initial (STEP 0) + first-resume (STEP 1) input scripts, per system
│   └── model_systems/          Fig-1 runs: pure-protein, protein+RNA, and RNA-titration inputs + PDBs
├── data/                     Final per-temperature Quant_Data_*.csv (correlation-corrected observables)
├── docs/                     METHODS_AND_REPRODUCIBILITY.md (units, constants, statistical framework)
├── setup_data_links.sh       Creates the ./PYTHON_ANALYSIS (+ ./SIMULATION) data symlinks
├── requirements.txt
├── LICENSE                   (GPLv3)
└── README.md
```

> **This is a code-only repository.** Raw trajectories (~0.6 TB), per-frame analysis products, and
> rendered figures are large and regenerable, so they are not committed (see `.gitignore`). The final
> manuscript figures live with the manuscript. The committed `data/*.csv` summary tables and
> `simulation_inputs/` are the deliberate exceptions.

---

## Installation

```bash
# HPC module example (adapt to your environment):
module load anaconda3/2025.12
conda create -n sg python=3.9
conda activate sg
pip install -r requirements.txt
```

Equivalent Conda environment file:

```bash
conda env create -f environment.yml
conda activate sg
```

All analysis scripts run **single-threaded**; set the BLAS/OpenMP thread caps before running:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

---

## Data setup

The pipeline reads the consolidated per-frame trajectory-analysis products through an in-repo symlink
`./PYTHON_ANALYSIS`. Create it (and the optional `./SIMULATION` link) once:

```bash
bash setup_data_links.sh                                # Princeton default data location
DATA_ROOT=/path/to/your/data bash setup_data_links.sh   # anywhere else
```

`DATA_ROOT` must contain `PYTHON_ANALYSIS/TEMP_<T>/{SG,DSM,NDSM}/...`. The links are git-ignored
(machine-specific); every script resolves `<repo-root>/PYTHON_ANALYSIS` from its own location (via
`__file__`), so once the link exists all data paths the code uses are repo-relative.

---

## How the pipeline fits together

```
 cg_pipeline/            simulation_inputs/            analysis/
 ───────────             ─────────────────            ─────────
 build MPiPi system  →   LAMMPS MD (2 µs/system)  →   per-frame extraction (RCC)
 parameterize SMs        25 restart segments          → windowed observables
                                                       → thermodynamics / dynamics / contacts
                                                       → classification & figures
```

### 1. Coarse-graining (`cg_pipeline/`)
Builds the residue-resolution MPiPi LAMMPS system (`sys.data`, `sys.settings`) and parameterizes each
small molecule (bead mapping + QSPR-derived nonbonded parameters). See `cg_pipeline/README.md`.

### 2. Simulation (`simulation_inputs/`)
LAMMPS (`real` units) production runs. Each system is launched once (**STEP 0**) and continued through
24 restart segments (**STEP 1 …**) by reading the previous segment's restart file. The repository ships
the STEP 0 and first-resume (STEP 1) input scripts for every system and temperature, the shared
`sys.data`/`sys.settings`, and the per-compound `.mol` topologies — everything needed to reproduce the
trajectories with a LAMMPS MPiPi build. See `simulation_inputs/README.md` for the run recipe (the `.in`
scripts differ across temperature only in `variable temperature equal <T>`).

### 3. Analysis & figures (`analysis/` → `plotting/`)
Processing and plotting are fully separated: the `analysis/` scripts do all computation and write every
numeric result to disk (CSV/NPZ); the `plotting/` scripts read those files and render the figures. Run
both **from the repository root** (`python analysis/<step>.py …`, then `python plotting/<step>.py …`).
Per-frame trajectory products are read through the `./PYTHON_ANALYSIS` symlink and all outputs are
written under `<repo-root>/TEMP_<T>/<FOLDER>_<T>_<tmin>_<dt>_<tmax>/`. (Each `plotting/` script adds
`analysis/` to its import path automatically, so the shared compute modules resolve.)

Ordered pipeline (each step consumes the previous step's outputs):

| # | Script | Produces |
|---|---|---|
| 0 | `rcc_analysis.py` | per-frame extraction from LAMMPS dumps (radial density, MSD/COM, stress, contacts) |
| 1 | `average_simulations.py` | 50 ns time-window averaging |
| 2 | `max_cluster.py` | biopolymer / amino-acid composition counting |
| 3 | `system_analysis.py` | thermodynamics & dynamics → `Quant_Data.csv` |
| 4 | `biopolymer_analysis.py` | per-species radial-density overlays |
| 5 | `time_analysis.py` | temporal-evolution diagnostics |
| 6 | `contact_maps.py` | domain contact-map heatmaps (written under `FIGURES/DOMAIN_CONTACT_MAPS/`) |
| 7 | `kmeans.py` | PCA + KMeans clustering, violin plots, dissolving/non-dissolving classifier |

**CLI convention.** The numbered pipeline scripts share `--path --folder --tmin --dt --tmax`; the
temperature flag is `--temp` for `rcc_analysis.py`, `average_simulations.py`, `max_cluster.py`,
`contact_maps.py`, and `--T` for `system_analysis.py`, `biopolymer_analysis.py`, `time_analysis.py`,
`kmeans.py`. (The Fig-1 composition/clustering modules use a different, file-based CLI — see step 5.)
No scheduler wrappers are shipped — invoke the scripts directly (parallelize the heavy Step 0 with your
own batch system).

---

## QUICKSTART (copy-paste, run from the repo root)

After **Installation** and **Data setup** below, the following reproduces one temperature (300 K, the
reference) end to end. Every command is run **from the repository root**; the `analysis/` steps write
CSV/NPZ, then the `plotting/` steps read them and render the panels. Per-frame products are read through
the `./PYTHON_ANALYSIS` symlink; pipeline outputs land under
`<repo-root>/TEMP_<T>/<FOLDER>_<T>_<tmin>_<dt>_<tmax>/`. (Each `plotting/` script adds `analysis/` to its
import path automatically, so the shared compute modules resolve.)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
T=300
```

**1. Per-frame extraction (Step 0 — the heavy step).** Run once per system; reads the consolidated
trajectory products and writes `ANALYSIS_<category>/`. Parallelize across systems and 50 ns windows:

```bash
python analysis/rcc_analysis.py --path ./PYTHON_ANALYSIS/TEMP_$T --folder SG --temp $T \
       --system sg_X --tmin 0 --tmax 50 --dt 1        # repeat per 50 ns window and per system
```

**2. Uncorrected windowed summary (`--folder CLASSIFY`).** Steps 1–3 of the ordered table; produces the
intermediate `Quant_Data.csv` that the correlation-corrected pass consumes:

```bash
python analysis/average_simulations.py   --path TEMP_$T --folder CLASSIFY --temp $T --tmin 50 --dt 50 --tmax 2000
python analysis/max_cluster.py            --path TEMP_$T --folder CLASSIFY --temp $T --tmin 50 --dt 50 --tmax 2000
python analysis/system_analysis.py --path TEMP_$T --folder CLASSIFY --T    $T --tmin 50 --dt 50 --tmax 2000
```

**3. Correlation-corrected pass (`--folder CLASSIFY_CORRELATED`).** Reads the `CLASSIFY` tree and applies
the Flyvbjerg–Petersen / PyMBAR error model that yields the **published numbers**:

```bash
python analysis/block_correlation_diagnostics.py    --path TEMP_$T --folder CLASSIFY            --temp $T --tmin 50 --dt 50 --tmax 2000
python analysis/average_simulations_correlated.py   --path TEMP_$T --folder CLASSIFY_CORRELATED --temp $T --tmin 50 --dt 50 --tmax 2000 \
       --layout-csv TEMP_$T/CORRELATION_CLASSIFY_${T}_50_50_2000/RESULTS/CORRELATION_DIAGNOSTICS/RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv
python analysis/max_cluster_correlated.py           --path TEMP_$T --folder CLASSIFY_CORRELATED --temp $T --tmin 50 --dt 50 --tmax 2000
python analysis/system_analysis_correlated.py --path TEMP_$T --folder CLASSIFY_CORRELATED --T    $T --tmin 50 --dt 50 --tmax 2000
```

**4. Render the main pipeline figures** (read the `CLASSIFY_CORRELATED` outputs):

```bash
python plotting/biopolymer_analysis.py   --path TEMP_$T --folder CLASSIFY_CORRELATED --T    $T --tmin 50 --dt 50 --tmax 2000
python plotting/time_analysis.py         --path TEMP_$T --folder CLASSIFY_CORRELATED --T    $T --tmin 50 --dt 50 --tmax 2000
python plotting/contact_maps.py                 --path TEMP_$T --folder CLASSIFY_CORRELATED --temp $T --tmin 50 --dt 50 --tmax 2000
python plotting/kmeans.py                --path TEMP_$T --folder CLASSIFY_CORRELATED --T    $T --tmin 50 --dt 50 --tmax 2000
```

**5. Fig-1 composition & clustering panels** (temperature-independent; a separate, file-based CLI —
no `--T/--path/--folder`). Each compute module writes a tidy CSV and the matching plotting module
renders the panels; if the CSV is absent the plotting step computes it on the fly:

```bash
# Fig 1A/1B — sequence composition (compute → render).
# Fully self-contained: reads the committed analysis/sequences/ (<PROT>_PRDOS.csv + RNA.txt).
python analysis/sequence_composition.py --seq-dir analysis/sequences --out RESULTS/COMPOSITION
python plotting/composition_panels.py   --csv RESULTS/COMPOSITION/sequence_composition.csv --out FIGURES/COMPOSITION

# Fig 1C/1D — RNA-titration clustering (compute → render).
# --data-root defaults to simulation_inputs/model_systems/; it must hold each model
# system's CLUSTER/llps_rg_*.out dump (NOT committed — produced by running the
# model_systems LAMMPS inputs first; see simulation_inputs/model_systems/README.md).
python analysis/cluster_analysis.py  --data-root simulation_inputs/model_systems --out RESULTS/CLUSTERING
python plotting/clustering_panels.py --data-root simulation_inputs/model_systems --out FIGURES/CLUSTERING
```

**Smoke tests.** After installing the requirements, reviewers can run checks that
use only committed files:

```bash
python -m unittest discover -s tests
# or, if using pytest:
pytest
```

These tests exercise the shipped sequence inputs, final `data/Quant_Data_*K.csv`
schemas, stdlib-only CLI help, and the data-link setup failure mode. Full
trajectory analysis still requires the external `PYTHON_ANALYSIS` tree described
above.

### Output convention (one canonical results tree)
The authoritative results for each temperature live in `CLASSIFY_CORRELATED_<T>_50_50_2000/`, whose
`RESULTS/SUMMARY/Quant_Data.csv` carries the **correlation-corrected** observables reported in the
paper. The legacy `CLASSIFY_<T>_…` tree is its required input (the uncorrected windowed summary), not a
separate result. See `docs/METHODS_AND_REPRODUCIBILITY.md` for the error model.

---

## Final data (`data/`)

`data/Quant_Data_<T>K.csv` — the correlation-corrected master observable table at each temperature.
The 285 / 300 / 315 K tables contain rows for SG + 10 DSM + 10 NDSM systems plus class aggregates; the
intermediate temperatures (290 / 295 / 305 / 310 K) contain the SG control only. Columns include dense
and dilute concentrations, surface tension, transfer free energy, diffusion coefficients (three
transport families), viscosities (Green–Kubo and Stokes–Einstein), radius of gyration, contact
fractions, and classifier features, each with correlation-corrected mean ± SEM.

Physical sanity checks: ΔG < 0, γ > 0, D ~ 10⁻⁸–10⁻⁷ cm²/s, η ~ 10⁻³–10⁻¹ Pa·s.

---

## Reproducing the figures

The figure-panel generators live in `plotting/` (they read the `analysis/` outputs and import the
shared compute modules, so run them from the repository root after the pipeline has produced its
outputs). `analysis/` is pure data-production — it writes every CSV/NPZ the figures consume but renders
nothing. Final manuscript figures are multi-panel composites assembled from these panels. The repo
renders only the plot groups used in the paper: the exploratory contact-map variants (P_contact,
z-score, intra-/all-chain filters) and the unused RDP overlays/difference plots were removed, though
their underlying CSV/NPZ data — including the z-score tables — are still written for downstream use.

Modules are listed as `compute (analysis/) → render (plotting/)` where the figure uses the
processing/plotting split, and the key classes each one defines are named in the last column
(function-only modules show `—`).

Modules in the first column live under `plotting/` unless they carry an `analysis/` prefix; classes
(third column) are lowercase snake_case.

| Paper figure | Modules (compute → render) | Key classes |
|---|---|---|
| Fig. 1A/1B (sequence composition) | `analysis/sequence_composition.py → composition_panels.py` | — |
| Fig. 1C/1D (RNA-titration clustering) | `analysis/cluster_analysis.py → clustering_panels.py` | `cluster_analysis` |
| Fig. 2–3 (structure, RDP, contacts) | `contact_maps.py`, `biopolymer_analysis.py` | `biopolymer_analysis`, `rdp` |
| Fig. 4 (PCA / KMeans / classifier) | `kmeans.py`, `audit_fixes.py` | `summary_plots` |
| Fig. 5 (per-species core composition) | `biopolymer_analysis.py`, `build_core_composition_shift.py` | `biopolymer_analysis`, `rdp` |
| Fig. 6 (dynamics / time) | `time_analysis.py` | `average`, `aggregate`, `average_biopolymers` |
| Fig. 7 (observable summaries, binodal) | `phase_diagram.py` | `phase_diagram`, `phase_diagram_config`, `fit_result` |
| SI per-species observables | `si_observables.py` | — |
| SI temperature trends (S7, S9, S10) | `si_temperature_trends.py`, `si_temp_figures.py`, `si_tc_spectrum.py` | — |
| SI classifier loadings (S8) | `classifier_cross_temperature.py` | — |
| SI contact filmstrips / domain maps (S11–S13) | `si_contact_filmstrips.py`, `si_selfself_domain_maps.py` | — |
| SI small-molecule isolation (S1) | `analysis/sm/ → plotting/sm/` | `summary_stats`, `dsu` |

The shared compute classes used across figures: `rdp` (radial-density-profile fit), `chain_result`
(diffusion), `acf`/`visc` (Green–Kubo viscosity), `phase_diagram`/`fit_result` (binodal fitting).

> **Auxiliary support scripts.** `plotting/audit_fixes.py` (statistical validation
> backing the Fig 4F classifier and binodal discussion) and
> `analysis/augment_binodal.py` (binodal merge-in) default to this repository and
> expose `--root` / `--out-dir` style flags. They reproduce supporting artifacts,
> not the canonical pipeline figures above.

---

## Notes on portability

These scripts were run on the Princeton *Stellar* cluster. The consolidated trajectory-input data is
reached through the `./PYTHON_ANALYSIS` symlink created by `setup_data_links.sh` (point it at your own
data with `DATA_ROOT=...`); the numbered pipeline and figure modules in the
table above contain no development-machine absolute paths. The remaining
cluster-specific values are the `anaconda3` module version and, if you batch the
heavy Step 0, your scheduler's `--qos`/`--partition` — adapt these to your
environment.

---

## Citation

If you use this code or the model, please cite:

```bibtex
@article{kaplan_webb_stress_granule,
  title   = {Interactions Underlying Stress Granule Structure and Therapeutic Dissolution},
  author  = {Kaplan, Jay L. and Webb, Michael A.},
  journal = {Cell Reports Physical Science},
  year    = {2026},
  note    = {In revision}
}
```

## License

Copyright (C) 2026 Jay L. Kaplan and Michael A. Webb (Webb Group, Department of
Chemical and Biological Engineering, Princeton University).

Released under the GNU General Public License v3.0 (GPLv3) — see [`LICENSE`](LICENSE).
