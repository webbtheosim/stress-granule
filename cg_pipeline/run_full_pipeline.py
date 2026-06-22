"""Entry point for the MPiPi coarse-graining workflow.

Orchestrates every legacy step -- coarse-grain mapping (optional GBCG),
descriptor generation and greedy feature selection (qspr.py), mixing-rule
application, force-field file generation (GenFiles), and parity/R^2 plots --
in-process, without shelling out to external Python scripts. Behaviour matches
the original standalone scripts but is exposed through a single configurable CLI.

Purpose: take MPiPi reference parameters plus a catalogue of new small-molecule
beads and produce a ready-to-simulate LAMMPS system (sys.data / sys.settings)
together with the predicted per-bead parameters and diagnostic plots.

Inputs (CLI paths): MPiPi parameter and molecule CSVs, a new-molecule CSV, WF
parameters, a template sys.data, and a master bonds file; optional feature /
mixing-rule / pretrained-model / reference files (see ``build_arg_parser``).

Outputs (written under ``<run-root>/RUN_<name>_<timestamp>/``): parameters.csv,
sm_parameters.csv, mix_params_JK.txt, SM_SM_Parameters_JK.csv, sys.data,
sys.settings, the RandomForest model/columns, PLOTS/, and RUN_INFO.txt.

CLI:
    python run_full_pipeline.py --mpipi-parameters data/MPiPi_Parameters.csv \
        --mpipi-molecules data/MPiPi_Molecules.csv --new-molecules drug_molecules.csv \
        --wf-parameters data/WF_Parameters.txt --template-sys-data sys.data \
        --bonds data/bonds.txt [--reuse-coarse-assets] [--features ...] [...]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import os
import sys

import pandas as pd

# Make the sibling cg_pipeline compute modules importable when run as a script,
# and make the relocated figure-rendering modules (now under plotting/cg/)
# importable as well. The CG pipeline itself stays render-free; rendering lives
# in plotting/cg/ and is invoked from here purely as a convenience driver.
_CG_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTTING_CG_DIR = os.path.join(_CG_DIR, os.pardir, "plotting", "cg")
for _p in (_CG_DIR, _PLOTTING_CG_DIR):
    _p = os.path.normpath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from convert_cg import convert_cg
from gen_bonds import gen_bonds
from gen_files import settings


def _load_gbcg():
    """Lazily import the optional external GBCG package.

    GBCG (graph-based coarse-graining, Webb group) is an optional dependency used
    when regenerating bead mappings rather than reusing committed coarse assets.
    It is not bundled; clone it into ``./GBCG``
    (see the README "Coarse-graining (optional)" section). The rest of the
    pipeline (QSPR, mixing, parity, plots) runs without it.
    """
    try:
        from GBCG import GB_mapping_spectral_pdb as gbcg
        return gbcg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Regenerating coarse assets requires the external GBCG package, which is "
            "not bundled. Clone it into ./GBCG (see README). Original error: "
            f"{exc}"
        ) from exc

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # committed reference / parameter data
R2_DATASET_CSV = DATA_DIR / "R2_Dataset.csv"
RANDOM_FOREST_MODEL = DATA_DIR / "random_forest_model.pkl"
RANDOM_FOREST_COLUMNS = DATA_DIR / "random_forest_columns.json"

PARAMETER_KEYS = {
    'epsilon': 'E',
    'eps': 'E',
    'e': 'E',
    'sigma': 'S',
    's': 'S',
    'nu': 'V',
    'v': 'V',
    'mu': 'U',
    'u': 'U',
    'rc': 'R',
    'cutoff': 'R',
}

MIXING_RULES = {'LB', 'WH', 'FH', 'K', 'G', 'A'}


def _parse_block_file(path: Path) -> Dict[str, List[str]]:
    """Parse a feature override file into {parameter: [descriptor, ...]}.

    The file is a sequence of blank-line-separated blocks, each headed by a
    parameter name (mapped through ``PARAMETER_KEYS``) followed by one descriptor
    name per line. Unknown headings are warned about and skipped. Empty lists are
    dropped from the result.
    """
    mapping: Dict[str, List[str]] = {key: [] for key in ['E', 'S', 'U', 'R', 'V']}
    current: Optional[str] = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            current = None
            continue
        if current is None:
            key = PARAMETER_KEYS.get(line.lower())
            if key is None:
                LOGGER.warning("Ignoring unknown parameter heading '%s' in %s", line, path)
                continue
            current = key
            continue
        mapping.setdefault(current, []).append(line)
    return {k: v for k, v in mapping.items() if v}


def _parse_mixing_file(path: Path) -> Dict[str, str]:
    """Parse a mixing-rule override file into {parameter: rule}.

    Same blank-line-separated block layout as ``_parse_block_file``, but each
    block body is a single mixing-rule code (one of ``MIXING_RULES``). Unknown
    parameter headings and unknown rule codes are warned about and skipped.
    """
    result: Dict[str, str] = {}
    current: Optional[str] = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            current = None
            continue
        if current is None:
            key = PARAMETER_KEYS.get(line.lower())
            if key is None:
                LOGGER.warning("Ignoring unknown parameter heading '%s' in %s", line, path)
                continue
            current = key
            continue
        if line not in MIXING_RULES:
            LOGGER.warning("Unknown mixing rule '%s' for %s in %s", line, current, path)
            continue
        result[current] = line
    return result


def _combine_molecule_files(mpipi_file: Path, new_file: Path, output_file: Path) -> None:
    """Concatenate the MPiPi reference and new-molecule catalogues into one CSV."""
    base_df = pd.read_csv(mpipi_file)
    new_df = pd.read_csv(new_file)
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined.to_csv(output_file, index=False)


def _write_sm_parameters(mpipi_dataset: pd.DataFrame, output_file: Path) -> None:
    """Export the small-molecule rows (beyond the first 24) to a parameters CSV.

    Writes the fixed column order (Biomolecule, Molecule Number, E, S, V, U, R,
    Q, M). If any required column is absent the export is skipped with a warning.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sm_df = mpipi_dataset.iloc[24:, :].copy()
    columns = ["Biomolecule", "Molecule Number", "E", "S", "V", "U", "R", "Q", "M"]
    missing = [col for col in columns if col not in sm_df.columns]
    if missing:
        LOGGER.warning("Skipping small-molecule parameter export; missing columns: %s", ", ".join(missing))
        return
    sm_df[columns].to_csv(output_file, index=False)


def _convert_rules_for_mixer(rules: Dict[str, str]) -> Dict[str, str]:
    """Rename parameter keys (E/S/U/R/V) to the mixer's names (eps/sig/mu/rc/v)."""
    mapping = {'E': 'eps', 'S': 'sig', 'U': 'mu', 'R': 'rc', 'V': 'v'}
    return {mapping[key]: value for key, value in rules.items() if key in mapping}


@dataclass
class coarse_grain_config:
    """GBCG coarse-graining settings and input/output directories.

    When ``regenerate`` is False the GBCG step is skipped and pre-existing
    coarse assets are reused; the remaining paths then locate those assets.
    """

    pdb_input_dir: Path = BASE_DIR / "GBCG" / "SM_PDB_FILES"
    molecule_dir: Path = BASE_DIR / "GBCG" / "Molecule_Files"
    bead_pdb_dir: Path = BASE_DIR / "Bead_PDB_Files"
    bead_smi_dir: Path = BASE_DIR / "Bead_SMI_Files"
    cg_pdb_dir: Path = BASE_DIR / "CG_PDB_Files"
    cg_mol_dir: Path = BASE_DIR / "CG_MOL_Files"
    map_dir: Path = BASE_DIR / "map_files"
    temp_pdb_dir: Path = BASE_DIR / "pdb_files"
    masses_file: Path = DATA_DIR / "masses.txt"
    bonds_file: Path = DATA_DIR / "bonds.txt"
    iterations: int = 4
    weights: str = "mass"
    max_size: float = 400.0
    regenerate: bool = True


@dataclass
class cg_pipeline_config:
    """Full pipeline configuration: required inputs, optional overrides, outputs.

    Bundles the QSPR/mixing/force-field inputs and the default output paths used
    by ``cg_pipeline``; the nested ``coarse`` field holds the GBCG settings. The
    pipeline reroutes most output paths into a per-run directory at runtime.
    """

    mpipi_parameters: Path
    mpipi_molecules: Path
    new_molecules: Path
    wf_parameters: Path
    template_sys_data: Path
    bonds_file: Path
    run_name: Optional[str] = None
    interaction_plot_num: int = 0
    random_seed: int = 20240509
    heterotypic_parameters: Optional[Path] = None
    template_sys_settings: Optional[Path] = None
    feature_file: Optional[Path] = None
    mix_rules_file: Optional[Path] = None
    reference_mix_file: Optional[Path] = None
    reference_sm_file: Optional[Path] = None
    pretrained_model: Optional[Path] = None
    pretrained_columns: Optional[Path] = None
    output_sys_data: Path = BASE_DIR / "sys.data"
    output_sys_settings: Path = BASE_DIR / "sys.settings"
    mix_params_output: Path = BASE_DIR / "mix_params_JK.txt"
    sm_parameters_output: Path = BASE_DIR / "SM_SM_Parameters_JK.csv"
    run_root: Path = BASE_DIR / "RUNS"
    coarse: coarse_grain_config = field(default_factory=coarse_grain_config)


class coarse_grain_manager:
    """Drive GBCG coarse-graining of each small-molecule PDB into bead assets.

    For every input PDB, runs GBCG mapping and emits the per-molecule map/PDB,
    appends bead masses to the master masses file, converts structures, and
    generates bonds. A no-op when ``cfg.regenerate`` is False.
    """

    def __init__(self, config: coarse_grain_config) -> None:
        """Store the coarse-grain configuration."""
        self.cfg = config

    def run(self) -> None:
        """Coarse-grain every PDB in the input directory (or reuse existing assets)."""
        if not self.cfg.regenerate:
            LOGGER.info("Reusing pre-existing coarse-grain assets in %s", self.cfg.molecule_dir)
            return

        LOGGER.info("Generating coarse-grain assets from %s", self.cfg.pdb_input_dir)
        self._prepare_directories()
        pdb_files = sorted(self.cfg.pdb_input_dir.glob("*.pdb"))
        if not pdb_files:
            LOGGER.warning("No PDB files found in %s; skipping coarse-grain generation", self.cfg.pdb_input_dir)
            return
        for pdb_file in pdb_files:
            self._process_single(pdb_file)

    def _prepare_directories(self) -> None:
        """Create the coarse-grain output directories and truncate master files."""
        for path in [
            self.cfg.molecule_dir,
            self.cfg.bead_pdb_dir,
            self.cfg.bead_smi_dir,
            self.cfg.cg_pdb_dir,
            self.cfg.cg_mol_dir,
            self.cfg.map_dir,
            self.cfg.temp_pdb_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.cfg.masses_file.write_text("")
        self.cfg.bonds_file.write_text("")

    def _process_single(self, pdb_file: Path) -> None:
        """Coarse-grain one PDB: run GBCG, then emit masses, structures, and bonds.

        Raises:
            FileNotFoundError: if GBCG does not produce the expected map / CG PDB.
        """
        mol_name = pdb_file.stem
        LOGGER.debug("Coarse-graining %s", mol_name)
        files, options = self._run_gbcg(pdb_file)
        try:
            _load_gbcg().CGmapping(files, options)
        finally:
            summary = files.get('summary')
            if summary is not None and not summary.closed:
                summary.close()

        map_file = self.cfg.map_dir / f"iter.{self.cfg.iterations}.map"
        cg_pdb = self.cfg.temp_pdb_dir / f"mol_0.{self.cfg.iterations}.pdb"
        if not map_file.exists():
            raise FileNotFoundError(f"Expected map file '{map_file}' not produced by GBCG")
        if not cg_pdb.exists():
            raise FileNotFoundError(f"Expected coarse PDB '{cg_pdb}' not produced by GBCG")

        mol_folder = self.cfg.molecule_dir / mol_name
        mol_folder.mkdir(parents=True, exist_ok=True)

        self._append_masses(map_file, mol_name)
        self._convert_structures(map_file, pdb_file, mol_folder, mol_name, cg_pdb)
        self._generate_bonds(mol_folder, mol_name)

    def _run_gbcg(self, pdb_file: Path):
        """Build GBCG's argument/option objects for one PDB from the config."""
        parser = _load_gbcg().create_parser()
        args = parser.parse_args([
            '-pdb', str(pdb_file),
            '-niter', str(self.cfg.iterations),
            '-weights', self.cfg.weights,
            '-max_size', str(self.cfg.max_size),
        ])
        return _load_gbcg().convert_args(args)

    def _append_masses(self, map_file: Path, mol_name: str) -> None:
        """Append ``<mol>_b<bead_id> <mass>`` lines from a GBCG map to the masses file."""
        lines = map_file.read_text().splitlines()
        with self.cfg.masses_file.open('a') as handle:
            for line in lines:
                parts = line.split()
                if len(parts) < 3:
                    continue
                bead_id, _, mass = parts[:3]
                handle.write(f"{mol_name}_b{bead_id} {mass}\n")

    def _convert_structures(
        self,
        map_file: Path,
        pdb_file: Path,
        mol_folder: Path,
        mol_name: str,
        cg_pdb: Path,
    ) -> None:
        """Write per-bead CG structures and copy the map/CG/atomistic PDBs into place."""
        converter = convert_cg(str(map_file), str(pdb_file))
        converter.convert_cg(f"{mol_folder}/")
        converter.convert_cg(f"{self.cfg.bead_pdb_dir}/")

        shutil.copy2(map_file, mol_folder / f"cg_{mol_name}.map")
        shutil.copy2(cg_pdb, mol_folder / f"cg_{mol_name}.pdb")
        shutil.copy2(cg_pdb, self.cfg.cg_pdb_dir / f"{mol_name}.pdb")
        shutil.copy2(pdb_file, mol_folder / f"{mol_name}.pdb")

    def _generate_bonds(self, mol_folder: Path, mol_name: str) -> None:
        """Generate the molecule's bond file and append its bonds to the master file."""
        cg_pdb_relative = mol_folder / f"cg_{mol_name}.pdb"
        bonding = gen_bonds(cg_pdb_relative, mol_folder)
        bonding.write_file()
        with self.cfg.bonds_file.open('a') as master:
            for line in bonding.write_master_file():
                master.write(f"{line}\n")


class cg_pipeline:
    """End-to-end driver: coarse-grain -> QSPR -> force field -> plots -> archive.

    Creates a timestamped run directory, reroutes transient and output artefacts
    into it, and runs each stage in sequence via :meth:`run`.
    """

    def __init__(self, config: cg_pipeline_config) -> None:
        """Create the run directory and point coarse-grain transient files at it."""
        self.config = config
        self.run_dir = self._create_run_dir()
        self.coarse_manager = coarse_grain_manager(config.coarse)
        # Use run-local paths for transient artefacts
        self.molecules_path = self.run_dir / "molecules.csv"
        # Ensure coarse-grain transient files also go into the RUN dir
        self.config.coarse.masses_file = self.run_dir / 'masses.txt'
        if self.config.coarse.regenerate:
            self.config.coarse.bonds_file = self.run_dir / 'bonds.txt'
            self.config.bonds_file = self.config.coarse.bonds_file
        else:
            self.config.coarse.bonds_file = self.config.bonds_file

    def _create_run_dir(self) -> Path:
        """Create and return ``run_root/RUN_<name>_<timestamp>/`` for this run."""
        self.config.run_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"RUN_{self.config.run_name}_" if self.config.run_name else "RUN_"
        run_dir = self.config.run_root / f"{prefix}{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Run artefacts will be stored in %s", run_dir)
        return run_dir

    def run(self) -> None:
        """Execute the full pipeline end to end into the run directory.

        Combines molecule catalogues, coarse-grains, runs QSPR to predict
        parameters, writes the combined and small-molecule parameter CSVs,
        generates the force-field files (sys.data / sys.settings / mix params)
        via GenFiles, then runs analyses, parity plots, and archives the run.
        """
        LOGGER.info("Combining molecule catalogues")
        _combine_molecule_files(self.config.mpipi_molecules, self.config.new_molecules, self.molecules_path)

        self.coarse_manager.run()

        regressor, column_map = self._run_qspr()

        # Write the combined parameters only inside this run folder
        run_parameters_csv = self.run_dir / 'parameters.csv'
        regressor.mpipi_dataset.to_csv(run_parameters_csv, index=False)
        # Also write small-molecule-only parameters for plotting MIX curves
        sm_params_csv = self.run_dir / 'sm_parameters.csv'
        sm_df = regressor.mpipi_dataset.iloc[24:, :].copy()
        if not sm_df.empty:
            cols_full = ["Biomolecule", "Molecule Number", "E", "S", "V", "U", "R"]
            if all(c in sm_df.columns for c in cols_full):
                sm_df[cols_full].to_csv(sm_params_csv, index=False)
            else:
                keep = [c for c in ["Biomolecule", "E", "S", "V", "U", "R"] if c in sm_df.columns]
                sm_df[keep].to_csv(sm_params_csv, index=False)

        mix_override = {}
        if self.config.mix_rules_file:
            LOGGER.info("Using mixing rules from %s", self.config.mix_rules_file)
            mix_override = _convert_rules_for_mixer(_parse_mixing_file(self.config.mix_rules_file))

        LOGGER.info("Generating force-field outputs via GenFiles")
        if not self.config.bonds_file.exists() or self.config.bonds_file.stat().st_size == 0:
            raise FileNotFoundError(
                f"Bond file is missing or empty: {self.config.bonds_file}. "
                "Provide --bonds when using --reuse-coarse-assets, or regenerate "
                "coarse assets from PDB inputs."
            )
        # Route all generated artefacts into the run directory
        self.config.mix_params_output = self.run_dir / 'mix_params_JK.txt'
        self.config.sm_parameters_output = self.run_dir / 'SM_SM_Parameters_JK.csv'
        self.config.output_sys_data = self.run_dir / 'sys.data'
        self.config.output_sys_settings = self.run_dir / 'sys.settings'
        settings_obj = settings(
            parameters_csv=run_parameters_csv,
            sys_data=self.config.template_sys_data,
            bonds=self.config.bonds_file,
            wf_parameters=self.config.wf_parameters,
            rules_override=mix_override or None,
            output_sys_data=self.config.output_sys_data,
            output_sys_settings=self.config.output_sys_settings,
            mix_params_output=self.config.mix_params_output,
            mol_output_dir=self.run_dir / 'CG_MOL_Files',
        )

        for pdb_file in sorted(self.config.coarse.cg_pdb_dir.glob('*.pdb')):
            settings_obj.write_mol_file(str(pdb_file))

        _write_sm_parameters(regressor.mpipi_dataset, self.config.sm_parameters_output)

        self._run_analyses()
        self._run_parity_plots()
        self._archive_run(column_map)

    def _run_qspr(self):
        """Run QSPR descriptor training/prediction and return (regressor, columns).

        Resolves optional feature overrides (an R2 CSV implies fixed-cumulative
        order; a block file gives explicit per-parameter features) and optional
        pretrained models, then trains or reuses models to predict the
        small-molecule parameters, copying any reused artefacts into the run dir.
        """
        from qspr import regressor

        feature_map: Dict[str, List[str]] = {}
        features_is_r2 = False
        fixed_cumulative = False
        ordered_from_r2: Dict[str, List[str]] = {}
        if self.config.feature_file:
            path_str = str(self.config.feature_file)
            if path_str.lower().endswith('.csv'):
                # Parse R2 dataset to produce fixed descriptor order per parameter
                try:
                    df_r2 = pd.read_csv(self.config.feature_file)
                    # Normalize parameter names to E/S/U/R
                    name_map = {
                        '$\\epsilon$': 'E', '$\\sigma$': 'S', '$\\mu$': 'U', '$r_{c}$': 'R',
                        'E': 'E', 'S': 'S', 'U': 'U', 'R': 'R', 'epsilon': 'E', 'sigma': 'S', 'mu': 'U', 'rc': 'R'
                    }
                    for key, grp in df_r2.groupby(df_r2['Parameter'].map(lambda v: name_map.get(str(v), str(v)))):
                        if key in {'E','S','U','R'}:
                            g = grp.sort_values('x')
                            desc = g['Mordred Descriptor'].astype(str).tolist()
                            # Keep first occurrence order
                            seen = set(); order = []
                            for d in desc:
                                if d not in seen:
                                    seen.add(d); order.append(d)
                            ordered_from_r2[key] = order
                    feature_map = ordered_from_r2
                    fixed_cumulative = True
                except Exception as e:
                    LOGGER.warning("Failed to parse features R2 CSV %s: %s", self.config.feature_file, e)
                    feature_map = {}
            else:
                LOGGER.info("Loading feature overrides from %s", self.config.feature_file)
                feature_map = _parse_block_file(self.config.feature_file)

        pretrained_models = None
        pretrained_columns = None
        if self.config.pretrained_model:
            LOGGER.info("Using pre-trained RandomForest models from %s", self.config.pretrained_model)
            with self.config.pretrained_model.open('rb') as handle:
                models_bundle = pickle.load(handle)
            if isinstance(models_bundle, dict) and 'models' in models_bundle:
                pretrained_models = models_bundle['models']
                pretrained_columns = models_bundle.get('columns')
            else:
                pretrained_models = models_bundle
            if pretrained_columns is None and self.config.pretrained_columns:
                pretrained_columns = json.loads(self.config.pretrained_columns.read_text())
            if pretrained_columns is None:
                raise ValueError("Pre-trained model provided without column mapping; supply --pretrained-columns or embed in pickle")

        reg = regressor(
            molecules_csv=str(self.molecules_path),
            parameters_csv=str(self.config.mpipi_parameters),
            feature_sets=feature_map,
            out_dir=str(self.run_dir),
            allow_write_r2=True,
            fixed_cumulative=fixed_cumulative,
            random_state=self.config.random_seed,
        )

        if pretrained_models is not None:
            column_map = reg.model_predict(pretrained=pretrained_models, columns_override=pretrained_columns)
            shutil.copy2(self.config.pretrained_model, self.run_dir / RANDOM_FOREST_MODEL.name)
            if self.config.pretrained_columns:
                shutil.copy2(self.config.pretrained_columns, self.run_dir / RANDOM_FOREST_COLUMNS.name)
            else:
                (self.run_dir / RANDOM_FOREST_COLUMNS.name).write_text(json.dumps(column_map))
            # If an external R2 dataset was provided along with the model, copy it into RUN
            if self.config.feature_file and str(self.config.feature_file).lower().endswith('.csv'):
                try:
                    shutil.copy2(self.config.feature_file, self.run_dir / 'R2_Dataset.csv')
                except Exception:
                    LOGGER.warning("Could not copy external R2 dataset into RUN")
        else:
            column_map = reg.model_predict()

        # Do not write shared parameters.csv here; caller writes into run dir only
        return reg, column_map

    def _run_analyses(self) -> None:
        """Render the small-molecule parameter plots and the R^2 selection plot."""
        from parameter_analysis import run_parameter_analysis
        from r2_analysis import run_r2_analysis

        plots_dir = self.run_dir / 'PLOTS'
        plots_dir.mkdir(parents=True, exist_ok=True)
        sm_params_csv = self.run_dir / 'sm_parameters.csv'
        if sm_params_csv.exists():
            run_parameter_analysis(
                sm_params_csv,
                plots_dir,
                self.config.interaction_plot_num,
                self.molecules_path
            )
        # Handle R2 dataset/plot according to features flag
        r2_metrics = self.run_dir / 'R2_Dataset.csv'
        # Only copy external R2 dataset when a pretrained model is used (no retraining)
        if self.config.pretrained_model and self.config.feature_file and str(self.config.feature_file).lower().endswith('.csv'):
            try:
                shutil.copy2(self.config.feature_file, r2_metrics)
            except Exception:
                LOGGER.warning("Failed to copy external R2 dataset from %s", self.config.feature_file)
        if r2_metrics.exists():
            run_r2_analysis(r2_metrics, plots_dir)

    def _run_parity_plots(self) -> None:
        """Render predicted-vs-reference parity plots for mixing and homotypic SM params.

        Homotypic small-molecule parity is restricted to the dsm_/ndsm_ beads
        present in this run's molecule catalogue when that filter is non-empty.
        """
        from compare_parameters import parity_comparer

        plots_dir = self.run_dir / 'PLOTS'
        plots_dir.mkdir(parents=True, exist_ok=True)
        if self.config.reference_mix_file:
            comparer = parity_comparer('mix', str(self.config.reference_mix_file), str(self.config.mix_params_output))
            comparer.plot_all(save=True, out_dir=str(plots_dir))
        # Always attempt homotypic SM parity plots
        jj_sm = self.config.reference_sm_file or (DATA_DIR / 'SM_SM_Parameters_JJ')
        if Path(jj_sm).exists():
            allowed_sm: set[str] | None = None
            if self.molecules_path.exists():
                df_mols = pd.read_csv(self.molecules_path)
                if 'Molecules' in df_mols.columns:
                    prefixed = df_mols['Molecules'].astype(str)
                    allowed_sm = set(prefixed[prefixed.str.startswith(('ndsm_', 'dsm_'))])
                    if not allowed_sm:
                        allowed_sm = None
            sm_comparer = parity_comparer('sm', str(jj_sm), str(self.config.sm_parameters_output))
            sm_comparer.plot_all(save=True, out_dir=str(plots_dir), allowed=allowed_sm)

    def _archive_run(self, column_map: Dict[str, List[str]]) -> None:
        """Copy key outputs and inputs into the run dir and write RUN_INFO.txt.

        RUN_INFO.txt records the resolved input paths, options, random seed, and
        the selected feature columns so a run is reproducible from its folder.
        """
        artefacts: Iterable[Path] = [
            self.run_dir / 'parameters.csv',
            self.run_dir / 'sm_parameters.csv',
            self.config.mix_params_output,
            self.config.sm_parameters_output,
            self.config.output_sys_data,
            self.config.output_sys_settings,
            self.config.coarse.masses_file,
            self.config.bonds_file,
            self.molecules_path,
            self.run_dir / 'random_forest_model.pkl',
            self.run_dir / 'random_forest_columns.json',
        ]

        for artefact in artefacts:
            if artefact.exists():
                dest = self.run_dir / artefact.name
                try:
                    if artefact.resolve() != dest.resolve():
                        shutil.copy2(artefact, dest)
                except FileNotFoundError:
                    continue

        info_file = self.run_dir / 'RUN_INFO.txt'
        with info_file.open('w') as handle:
            handle.write(f"MPiPi parameters: {self.config.mpipi_parameters}\n")
            handle.write(f"MPiPi molecules: {self.config.mpipi_molecules}\n")
            handle.write(f"New molecules: {self.config.new_molecules}\n")
            handle.write(f"WF parameters: {self.config.wf_parameters}\n")
            handle.write(f"Template sys.data: {self.config.template_sys_data}\n")
            if self.config.template_sys_settings:
                handle.write(f"Template sys.settings: {self.config.template_sys_settings}\n")
            handle.write(f"Bonds file: {self.config.bonds_file}\n")
            handle.write(f"Features: {self.config.feature_file or 'None'}\n")
            handle.write(f"Mix rules: {self.config.mix_rules_file or 'None'}\n")
            handle.write(f"Reference mix: {self.config.reference_mix_file or 'None'}\n")
            handle.write(f"Reference SM: {self.config.reference_sm_file or 'None'}\n")
            handle.write(f"Pretrained model: {self.config.pretrained_model or 'None'}\n")
            handle.write(f"Pretrained columns: {self.config.pretrained_columns or 'None'}\n")
            handle.write(f"Random seed: {self.config.random_seed}\n")
            handle.write(f"Feature columns: {json.dumps(column_map)}\n")

        inputs_to_copy = [
            self.config.mpipi_parameters,
            self.config.mpipi_molecules,
            self.config.new_molecules,
            self.config.wf_parameters,
        ]
        if self.config.mix_rules_file:
            inputs_to_copy.append(self.config.mix_rules_file)
        if self.config.feature_file:
            # Avoid overwriting regenerated R2 dataset when retraining with feature CSV
            if not (self.config.pretrained_model and str(self.config.feature_file).lower().endswith('.csv')):
                if str(self.config.feature_file).lower().endswith('.csv') and not self.config.pretrained_model:
                    pass
                else:
                    inputs_to_copy.append(self.config.feature_file)
        if self.config.pretrained_columns:
            inputs_to_copy.append(self.config.pretrained_columns)
        if self.config.pretrained_model:
            inputs_to_copy.append(self.config.pretrained_model)
        if self.config.reference_mix_file:
            inputs_to_copy.append(self.config.reference_mix_file)
        if self.config.reference_sm_file:
            inputs_to_copy.append(self.config.reference_sm_file)

        for src in inputs_to_copy:
            try:
                shutil.copy2(src, self.run_dir / src.name)
            except FileNotFoundError:
                LOGGER.warning("Input file %s not found for archival", src)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the coarse-graining pipeline."""
    parser = argparse.ArgumentParser(description='Run the MPiPi coarse-graining pipeline')
    parser.add_argument('--mpipi-parameters', required=True, type=Path)
    parser.add_argument('--mpipi-molecules', required=True, type=Path)
    parser.add_argument('--new-molecules', required=True, type=Path)
    parser.add_argument('--wf-parameters', required=True, type=Path)
    parser.add_argument('--template-sys-data', required=True, type=Path)
    parser.add_argument('--bonds', required=True, type=Path, help='Master bonds file to use or regenerate')
    parser.add_argument('--heterotypic-parameters', type=Path, default=None)
    parser.add_argument('--template-sys-settings', type=Path, default=None)
    parser.add_argument('--features', type=Path, default=None)
    parser.add_argument('--mix-rules', type=Path, default=None)
    parser.add_argument('--reference-mix', type=Path, default=None)
    parser.add_argument('--reference-sm', type=Path, default=None)
    parser.add_argument('--pretrained-model', type=Path, default=None)
    parser.add_argument('--pretrained-columns', type=Path, default=None)
    parser.add_argument('--output-sys-data', type=Path, default=None)
    parser.add_argument('--output-sys-settings', type=Path, default=None)
    parser.add_argument('--output-mix-params', type=Path, default=None)
    parser.add_argument('--output-sm-parameters', type=Path, default=None)
    parser.add_argument('--name', default=None, help='Optional run name to include in RUN_<name>_<timestamp>')
    parser.add_argument('--interaction-plot-num', type=int, default=0, help='Number of dsm_/ndsm_ molecules to include in interaction plots (0 = all)')
    parser.add_argument('--seed', type=int, default=20240509, help='Random seed for QSPR train/test splits and random forests')
    parser.add_argument('--run-root', type=Path, default=None)

    parser.add_argument('--pdb-input-dir', type=Path, default=None)
    parser.add_argument('--molecule-dir', type=Path, default=None)
    parser.add_argument('--bead-pdb-dir', type=Path, default=None)
    parser.add_argument('--bead-smi-dir', type=Path, default=None)
    parser.add_argument('--cg-pdb-dir', type=Path, default=None)
    parser.add_argument('--cg-mol-dir', type=Path, default=None)
    parser.add_argument('--map-dir', type=Path, default=None)
    parser.add_argument('--temp-cg-pdb-dir', type=Path, default=None)
    parser.add_argument('--masses-file', type=Path, default=None)
    parser.add_argument('--gbcg-iterations', type=int, default=4)
    parser.add_argument('--gbcg-max-size', type=float, default=400.0)
    parser.add_argument('--gbcg-weights', default='mass')
    parser.add_argument('--reuse-coarse-assets', action='store_true', help='Skip GBCG regeneration and reuse existing assets')

    parser.add_argument('--log-level', default='INFO')
    return parser


def _make_coarse_config(args: argparse.Namespace) -> coarse_grain_config:
    """Build a ``coarse_grain_config`` from parsed CLI args (overriding defaults)."""
    cfg = coarse_grain_config()
    if args.pdb_input_dir:
        cfg.pdb_input_dir = args.pdb_input_dir
    if args.molecule_dir:
        cfg.molecule_dir = args.molecule_dir
    if args.bead_pdb_dir:
        cfg.bead_pdb_dir = args.bead_pdb_dir
    if args.bead_smi_dir:
        cfg.bead_smi_dir = args.bead_smi_dir
    if args.cg_pdb_dir:
        cfg.cg_pdb_dir = args.cg_pdb_dir
    if args.cg_mol_dir:
        cfg.cg_mol_dir = args.cg_mol_dir
    if args.map_dir:
        cfg.map_dir = args.map_dir
    if args.temp_cg_pdb_dir:
        cfg.temp_pdb_dir = args.temp_cg_pdb_dir
    if args.masses_file:
        cfg.masses_file = args.masses_file
    cfg.bonds_file = args.bonds
    cfg.iterations = args.gbcg_iterations
    cfg.max_size = args.gbcg_max_size
    cfg.weights = args.gbcg_weights
    if args.reuse_coarse_assets:
        cfg.regenerate = False
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI args, configure logging, and run the pipeline once."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(levelname)s] %(message)s',
    )

    coarse_cfg = _make_coarse_config(args)

    config = cg_pipeline_config(
        mpipi_parameters=args.mpipi_parameters,
        mpipi_molecules=args.mpipi_molecules,
        new_molecules=args.new_molecules,
        wf_parameters=args.wf_parameters,
        template_sys_data=args.template_sys_data,
        bonds_file=args.bonds,
        run_name=args.name,
        interaction_plot_num=args.interaction_plot_num,
        random_seed=args.seed,
        heterotypic_parameters=args.heterotypic_parameters,
        template_sys_settings=args.template_sys_settings,
        feature_file=args.features,
        mix_rules_file=args.mix_rules,
        reference_mix_file=args.reference_mix,
        reference_sm_file=args.reference_sm,
        pretrained_model=args.pretrained_model,
        pretrained_columns=args.pretrained_columns,
        output_sys_data=args.output_sys_data or BASE_DIR / "sys.data",
        output_sys_settings=args.output_sys_settings or BASE_DIR / "sys.settings",
        mix_params_output=args.output_mix_params or BASE_DIR / "mix_params_JK.txt",
        sm_parameters_output=args.output_sm_parameters or BASE_DIR / "SM_SM_Parameters_JK.csv",
        run_root=args.run_root or BASE_DIR / "RUNS",
        coarse=coarse_cfg,
    )

    pipeline = cg_pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
