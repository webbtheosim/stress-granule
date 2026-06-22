# Unified ANALYSIS_SM pipeline

One driver (`run_analysis_sm.py`) that produces the small-molecule SI figures.
Both stages reuse the project's own code and the ground-truth data so the styles
and values match the manuscript exactly.

The sub-analysis is split across two trees, matching the repository convention
(compute writes data, plotting renders figures):

```
analysis/sm/                      COMPUTE side (no figures)
  run_analysis_sm.py   driver  (--isolation | --parameters | both); adds
                       plotting/sm/ to sys.path and calls the rendering stages
  sm_analysis.py       upstream trajectory -> CSV (cluster / RDP / RDF)
  sm_common.py         correlation-corrected statistics (shared policy)

plotting/sm/                      FIGURES side (all rendering)
  sm_parameters.py     SI Fig S1 (R^2, parity, interaction curves)
  sm_isolation.py      self-aggregation figures (house RDP style)
  sm_plotting.py       original isolation plotter (cluster loader reused above)
  run_sm_plots.py      thin CLI wrapper around sm_plotting.generate_plots
  sm_system_analysis.py  per-system SM figures from sm_analysis CSVs
  sm_common.py         house style (re-exports the compute-side statistics)
  README_PIPELINE.md   this file
```

**Prerequisites.**
- Run all commands **from the repository root** (the driver puts `plotting/sm/` on `sys.path` and
  writes to `FIGURES/` relative to the cwd; `--outdir` overrides the base directory).
- Needs **Python 3.10+** with pandas / scikit-learn / scipy / seaborn. The coarse-grained plotting
  modules use PEP-604 (`X | Y`) annotations, so the project's 3.9 `sg` env will **not** work — use a
  separate 3.10+ env.
- The **parameters** stage is self-contained: it reads only the committed CSVs in `cg_pipeline/`.
- The **isolation** stage reads the per-compound SM-isolation trajectory products through the data
  symlink `PYTHON_ANALYSIS/SM_300/{DSM,NDSM}/ANALYSIS_SM/`. Create that symlink first with the
  repo-root helper `bash setup_data_links.sh` (or `DATA_ROOT=/path/to/data bash setup_data_links.sh`).
  If the consolidated SM trajectory data is unavailable, run `--parameters` only.

```bash
module load anaconda3/2024.6 && conda activate base   # any Python 3.10+ env
python analysis/sm/run_analysis_sm.py             # both stages -> FIGURES/{isolation,parameters}
python analysis/sm/run_analysis_sm.py --parameters    # self-contained (cg_pipeline/ CSVs only)
python analysis/sm/run_analysis_sm.py --isolation     # needs the PYTHON_ANALYSIS data symlink
```

---

## Stage 1 — parameterisation (`sm_parameters.py` -> `FIGURES/parameters/`)

R^2 and parity **delegate to the original coarse-grained plotting code** (now in
`plotting/cg/`: `r2_analysis`, `compare_parameters.parity_comparer`); the interaction
curves are computed in `sm_parameters.py` itself (the `_wf_phi` Wang–Frenkel helper
+ `_interaction_panel`), styled to match the manuscript. The data lives in
`cg_pipeline/`. (The `RUNS/RUN_JJ_PARITY_..._184243` parity archive is not present in
this repository snapshot; `sm_parameters.py` falls back to the `mix_params_*` files
in `cg_pipeline/`, and skips parity gracefully if inputs are missing.)

| Figure | Maps to | Code | Data (ground truth) |
|--------|---------|------|---------------------|
| `R2_Plot.png` | S1A | `r2_analysis.run_r2_analysis` | `cg_pipeline/data/R2_Dataset.csv` = the published Table S1 (`features.txt` run): ε **0.916**, σ **0.878**, µ **0.969**, r_c **0.856** |
| `mix_{E,S,V,U,R}_JJ_vs_JK.png` | S1C | `parity_comparer('mix')` | `RUNS/RUN_JJ_PARITY_..._184243` (archived; falls back to `cg_pipeline/mix_params_*`) — reproduces PCC ε **0.59**, σ **0.86**, r **0.81** |
| `sm_{E,S,V,U,R}_JJ_vs_JK.png` | S1C | `parity_comparer('sm')` | same run (control molecules, coloured by species, `tab20`) |
| `MIX_PARAMS_DSM.png` / `MIX_PARAMS_NDSM.png` | S1B | `sm_parameters._interaction_panel` (manuscript-style Wang–Frenkel) | `cg_pipeline/data/drug_parameters.csv` — per-compound **averaged** Wang–Frenkel curves, **separate** DSM / NDSM panels, rocket partition (DSM = purple end, NDSM = orange end) |

Panel axes sizes: `R2_Plot` is square **2.2 × 2.2 in** (RDP-plot size); the `MIX_PARAMS`
panels are **2.2 (x) × 0.8 (y) in** with thick smooth curves and no legend; the parity
panels are **1.60 × 1.60 in** (property-violin size), centred so y-tick labels are not
clipped. R^2 / parity geometry is set via the optional `figsize`/`ax_rect` args added to
`r2_analysis.run_r2_analysis` and `parity_comparer.plot_all`.

Notes:
- The data lives in `cg_pipeline/`: `R2_Dataset.csv` and `drug_parameters.csv` were
  the **original uploaded data** (see `DATASETS/`); the parity uses run `184243`.
- The parity (a greedy-selection run) and the R^2 (a fixed-`features.txt` run) come
  from **different model runs** — that is what reproduces both published value sets.
  The exact production parameterisation predates the archived snapshots (sims are
  dated 2023), so these are the closest preserved/uploaded reproductions.

## Stage 2 — isolation (`sm_isolation.py` -> `FIGURES/isolation/{aggregated,dsm,ndsm}/`)

Each small molecule simulated alone (no protein/RNA); the per-compound cluster /
RDF / RDP CSVs are read through the repository data symlink:
`PYTHON_ANALYSIS/SM_300/{DSM,NDSM}/ANALYSIS_SM/`.
Cluster series use `sm_plotting`'s loader (correct 1 ns sampling, 20 ns-window
`_t<NN>` tokens → 0–200 ns); RDP/RDF are loaded per compound (mean over windows).

**Style, matching SI Fig S1 D/E:**
- axes sizes: cluster series (number / fraction) at the **RDP-plot size 2.2 × 2.2 in**;
  RDP/RDF profiles at the **violin size 1.6 × 1.6 in**. Inward ticks, no axis titles,
  **5 ticks** per axis (time plots x = 0–200), y limits floor→ceiling (2-sig-fig)
- class means = green DSM `#40641b` / NDSM `#bfe49b`, with filled **scatter markers
  (black edge)** and a **SEM band** (std-over-compounds / √N) — not SD, not error bars
- to de-clutter / smooth: cluster series take **every 5th frame**; RDP/RDF are
  **averaged over every 5 distance bins** (also tames the small-r shell noise)
- a **grey dashed** line marks the all-monomeric baseline where it fits
  (`cluster_number` → 8324); omitted where no SG reference is defined
- `dsm/` and `ndsm/` additionally hold `*_individual.png` with every molecule,
  coloured by a **rocket partition** (DSM = purple end, NDSM = orange end; matches SI_1 B)

Observables (× `aggregated` / `dsm` / `ndsm`): `cluster_fraction_vs_time`,
`cluster_number_vs_time`, `rdp_number`, `rdp_mass`, `rdf`.

---

## Provenance (where the parameters come from — `cg_pipeline/`)
A small-molecule SMILES → **GBCG** coarse-graining → **Mordred** descriptors →
**random-forest QSPR** (predicts ε, σ, ν, µ, r_c; trained on the 24 MPiPi residues
of Joseph & Joseph) → **Lorentz–Berthelot-family mixing** for heterotypic
interactions → **LAMMPS MPiPi** simulation. See `cg_pipeline/README.md`.
