# Methods & Reproducibility Notes

Condensed reference for the quantities computed by the analysis pipeline, the physical constants and
units behind them, and the statistical framework used for the published uncertainties. This is the
companion to the top-level [`README.md`](../README.md), which has the full repository layout, the
installation steps, and the end-to-end command recipe; this file is the conventions/constants
reference. See the module docstrings in `analysis/` for per-script detail.

## Orientation (where things live)

The repository is split into a **compute side** (`analysis/`, writes every CSV/NPZ) and a **plotting
side** (`plotting/`, reads those files and renders the panels). Module names below are given with
their directory so they can be located directly. A few anchors a reviewer usually needs:

| You want… | Look in |
|---|---|
| Heavy per-frame extraction (RDP, MSD/COM, stress, contacts) | `analysis/rcc_analysis.py` |
| Shared compute classes (`rdp`, `chain_result`, `acf`/`visc`) | `analysis/rdp.py`, `analysis/diffusion.py`, `analysis/acf.py`, `analysis/viscosity.py` |
| Correlation-corrected error model | `analysis/block_correlation_diagnostics.py` → `analysis/system_analysis_correlated.py` |
| Fig-1 composition + clustering | `analysis/sequence_composition.py` + `analysis/cluster_analysis.py`, rendered by `plotting/composition_panels.py` + `plotting/clustering_panels.py` |
| Classifier (Fig. 4F) and cross-temperature loadings | `plotting/audit_fixes.py`, `plotting/classifier_cross_temperature.py` |
| Binodal / phase-diagram fitting | `plotting/phase_diagram.py` (`phase_diagram`, `fit_result`) |
| LAMMPS inputs to (re)run the MD | `simulation_inputs/` (Fig-1 model systems in `simulation_inputs/model_systems/`; per-compound topologies in `simulation_inputs/small_molecule_templates/{dsm,ndsm}/`, named `dsm_*.mol` / `ndsm_*.mol`) |
| Final correlation-corrected observable tables | `data/Quant_Data_<T>K.csv` |

All compute/plotting classes are **lowercase snake_case** (e.g. `rdp`, `chain_result`, `visc`, `acf`,
`cluster_analysis`, `phase_diagram`); `analysis/sequence_composition.py` is function-only (no class).

## LAMMPS units (`real`)

| Quantity | Unit |
|---|---|
| Distance | Å |
| Time | fs (timestep 20 fs; frame interval 0.2 ns) |
| Energy | kcal/mol |
| Pressure / stress | atm |

Trajectory coordinates are written **unwrapped** (`xu yu zu`). The MSD pipeline converts Å² → m²
(×1e-20) and reports D in cm²/s (×1e4) or µm²/s (×1e12) as labeled.

## Key physical constants & cutoffs

- k_B = 1.3806 × 10⁻²³ J/K , N_A = 6.022 × 10²³ mol⁻¹
- **Cluster-membership cutoff: 20 Å** (defines which chains belong to the condensate).
- **Contact cutoff: 16 Å** (domain/residue contact maps; ≈ 2.0–2.6 σ across MPiPi bead types,
  accommodating the large nucleotide beads, σ ≈ 8.3 Å). These two cutoffs serve different purposes and
  should not be conflated.
- Density conversion: Da/Å³ → mg/mL × 1660.5390666 (exact per N_A).

## Radial density & interfacial properties

- Density profiles are fit to an error-function form:
  ρ(r) = B − A·erf((r − R)/(√2·W)), giving dense/dilute concentrations, the interface radius R and
  width W. Histograms are **mass-weighted** per radial bin (exact per-atom mass accounting).
- **Surface tension** from condensate shape fluctuations: K₁ = 15 k_BT/16π,
  γ₁ = K₁ / ⟨Σ(δa+δb)²⟩ (and the K₂ = 45 k_BT/16π mode).
- **Transfer free energy:** ΔG = RT · ln(c_dilute / c_dense).

## Transport properties

**Diffusion** is measured in the condensate co-moving frame (per-chain COM drift subtracted, chains
required to stay inside the condensate over the analysis window), using an unbiased FFT MSD kernel
(Kneller/nMoldyn, (T−m) normalization) and D = MSD/(6t) with log–log regime detection (slope
1.0 ± 0.2). Three transport *families* are computed and kept distinct (selected via
`source_mode`/`estimator_mode` in `analysis/diffusion.py`):

| Family | Source MSD |
|---|---|
| RCC full | condensate-frame RCC MSD, full estimator |
| Raw LAMMPS | raw per-chunk COM series |
| RCC segmented | segmented condensate-frame MSD (the family carried into the correlation-corrected pass) |

Per-family results are written to DIFFUSION's `*_diffusion_summary.csv` sidecars and then summarised
into the master table. In the master `Quant_Data.csv` the diffusion/viscosity quantities appear under
**LaTeX display-name headers** — e.g. `$D_{cage}$`, `$D_{loglog}$`, `$D_{SE,Rh}$`, `$D_{SE,Rg}$`,
`$\eta_{GK}$`, `$\eta_{GK Theo}$`, `$\tau_{cage}$`, `$\tau_{conf...}$` (each paired with a `SIG…` SEM
column) — rather than the raw `source_mode` family names above. Confinement length from the MSD
plateau: ℓ = √(MSD_plateau), with cage time τ_cage from the same fit.

**Viscosity.**
- *Green–Kubo* from the **condensate-only** stress autocorrelation (per-atom stress summed over
  cluster atoms only), with the dimensionally-verified prefactor dt·V/(k_BT) → Pa·s.
- *Stokes–Einstein* reported two ways: using the radius of gyration R_g (`eta_Rg`, legacy) and the
  Kirkwood hydrodynamic radius R_h (`eta_Rh`, physically preferred). Kirkwood:
  1/R_h = (2/N²) Σ_{i<j} 1/|r_i − r_j|.

## Windowing

Default `tmin=50, dt=50, tmax=2000` → **39 production blocks of 50 ns** (first 50 ns discarded as
equilibration). Temporal bins used in time-resolved plots: Early (0–0.3 µs), Middle (0.85–1.15 µs),
Late (1.7–2.0 µs).

## Correlation-corrected uncertainties (the "CORRELATED" pipeline)

Because each system is a single long trajectory, within-run time correlation is corrected explicitly
rather than assuming independent frames:

- Per-window values are winsorized at the 5th/95th percentiles before block analysis (guards the SEM
  against rare outlier windows).
- Block size selected as `max(Flyvbjerg–Petersen super-block plateau, PyMBAR statistical
  inefficiency)`, taken conservatively (max over the core recommendation observables per system).
- An all-offset batch estimator (median SEM across block offsets) yields the corrected mean and SEM,
  with `SEM = std(ddof=1)/√n_blocks`.
- 95 % CIs use the Student-t distribution with df = n_blocks − 1.
- A single long trajectory per system (`n_replicates = 1`); the scope is therefore explicitly
  *within-run time-correlation-corrected*, not an across-replicate error. Equilibration uses the fixed
  `tmin` cutoff with a Chodera `detect_equilibration` audit run as a diagnostic only.
- Every observable — diffusion, viscosity, RDP-derived quantities, cluster composition — flows through
  the same correction; class averages (DSM_AVG / NDSM_AVG) are computed from the corrected
  per-system means. This is implemented in `analysis/block_correlation_diagnostics.py` (which also
  writes `STATISTICAL_POLICY.csv`) and applied by `analysis/system_analysis_correlated.py`,
  producing `CLASSIFY_CORRELATED_*/RESULTS/SUMMARY/Quant_Data.csv`.

## Classification (dissolving vs non-dissolving)

The final classifier (paper Fig. 4F) is a fixed **two-feature** model on the small-molecule
partitioning P_SM and the cluster count N_D, achieving **85 % leave-one-out cross-validated accuracy**
(p ≈ 0.005). Feature definitions and the scorer are in `plotting/audit_fixes.py`; cross-temperature
loadings are in `plotting/classifier_cross_temperature.py`.
