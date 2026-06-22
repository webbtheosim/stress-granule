"""QSPR mapping from molecular descriptors to MPiPi force-field parameters.

Quantitative structure-property relationship (QSPR) step of the coarse-graining
workflow. Given SMILES for the bonded biomolecule beads (the 24 amino-acid /
nucleotide reference beads) plus the new small-molecule beads, it computes Mordred
2D/3D descriptors, selects descriptors per MPiPi parameter, trains one
RandomForest regressor per parameter on the 24 reference beads, and predicts the
four MPiPi parameters (E=epsilon, S=sigma, U=mu, R=cutoff) for every new bead.

Purpose: pure compute (descriptor generation, greedy/cumulative feature
selection, RandomForest training + prediction). All figure rendering has been
relocated to ``plotting/cg/``; do not reintroduce matplotlib/seaborn here.

Inputs:
    molecules_csv    -- bead catalogue with columns Molecules, SMILES, Mass
                        (rows 0-23 = reference beads, rows 24+ = new beads).
    parameters_csv   -- MPiPi reference parameters (columns Biomolecule, E, S,
                        U, R, ...) for the first 24 beads.
    feature_sets     -- optional {parameter: [descriptor, ...]} overrides.

Outputs (written to ``out_dir`` when training):
    R2_Dataset.csv             -- per-parameter R^2 / MAE selection trace
                                  (consumed by plotting/cg/r2_analysis.py).
    random_forest_model.pkl    -- pickled {parameter: fitted RandomForest}.
    random_forest_columns.json -- {parameter: [selected descriptor, ...]}.
    self.mpipi_dataset         -- reference table with predicted rows appended.

CLI (standalone smoke test only; normally driven by run_full_pipeline.py):
    python qspr.py   # reads data/molecules.csv + data/MPiPi_Parameters.csv, writes
                     # single_parameters.csv
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parent / "data"


class regressor:
    """Train per-parameter RandomForests on bead descriptors and predict MPiPi.

    One instance owns the molecule/parameter tables, the per-parameter fitted
    models (``self.models``), and the selected descriptor columns
    (``self.columns``). Typical use: ``model_predict()`` trains (or reuses
    pretrained models) and appends predicted parameter rows to
    ``self.mpipi_dataset`` for the new small-molecule beads.
    """

    def __init__(
        self,
        molecules_csv,
        parameters_csv,
        feature_sets=None,
        out_dir: str | None = None,
        allow_write_r2: bool = True,
        fixed_cumulative: bool = False,
        random_state: int = 20240509,
    ):
        """Load the molecule and parameter tables and configure training.

        Args:
            molecules_csv: Bead catalogue (Molecules, SMILES, Mass); rows 0-23
                are the MPiPi reference beads, rows 24+ the new beads to predict.
            parameters_csv: MPiPi reference parameters for the first 24 beads.
            feature_sets: Optional {parameter: [descriptor, ...]} overrides; when
                given, greedy selection is replaced by these explicit features.
            out_dir: Directory for R2_Dataset.csv / pickled model / columns JSON.
            allow_write_r2: If False, skip writing R2_Dataset.csv.
            fixed_cumulative: If True (and feature_sets given), train cumulatively
                over the provided descriptor order; else one-shot on the full set.
            random_state: Seed for train/test splits and the RandomForests.
        """
        self.molecule_dataset = pd.read_csv(molecules_csv)
        self.mpipi_dataset = pd.read_csv(parameters_csv)
        self.models = {}
        self.descriptors = pd.DataFrame
        self.parameters = pd.DataFrame
        self.feature_sets = feature_sets or {}
        # Suppress verbose greedy-selection prints whenever explicit features are given.
        self.verbose = not bool(self.feature_sets)
        self.columns = {}
        self.out_dir = Path(out_dir) if out_dir is not None else Path('.')
        self.allow_write_r2 = allow_write_r2
        self.fixed_cumulative = fixed_cumulative
        self.random_state = int(random_state)

    @staticmethod
    def _canonical_name(value):
        """Canonicalize molecule/bead names for descriptor-parameter alignment."""
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    def _aligned_training_descriptors(self, descriptors):
        """Return the first-24 descriptor rows aligned to parameter-table order.

        The bundled molecule and parameter CSVs are intentionally not sorted in
        the same order. Training by row position would pair descriptors with the
        wrong MPiPi labels, so this routine joins by canonical bead name and
        fails if the tables are not one-to-one.
        """
        parameter_names = self.mpipi_dataset.iloc[:24]["Biomolecule"].astype(str)
        molecule_names = self.molecule_dataset.iloc[:24]["Molecules"].astype(str)

        parameter_keys = parameter_names.map(self._canonical_name)
        molecule_keys = molecule_names.map(self._canonical_name)

        if parameter_keys.duplicated().any():
            duplicates = sorted(parameter_names[parameter_keys.duplicated()].unique())
            raise ValueError(f"Duplicate MPiPi parameter names: {duplicates}")
        if molecule_keys.duplicated().any():
            duplicates = sorted(molecule_names[molecule_keys.duplicated()].unique())
            raise ValueError(f"Duplicate MPiPi molecule names: {duplicates}")

        missing = sorted(set(parameter_keys) - set(molecule_keys))
        extra = sorted(set(molecule_keys) - set(parameter_keys))
        if missing or extra:
            raise ValueError(
                "MPiPi descriptor/parameter names do not align. "
                f"Missing molecule descriptors for parameters: {missing}; "
                f"extra descriptor rows: {extra}"
            )

        training = descriptors.iloc[:24].copy()
        training.index = molecule_keys
        aligned = training.loc[parameter_keys].copy()
        aligned.index = range(len(aligned))
        return aligned

    def mordred_descriptors(self, data):
        """Compute the full Mordred 2D/3D descriptor table for a SMILES iterable.

        Args:
            data: Iterable of SMILES strings.

        Returns:
            DataFrame of Mordred descriptors, one row per input molecule.
        """
        calc = Calculator(descriptors, ignore_3D=False)
        mols = [Chem.MolFromSmiles(smi) for smi in data]
        if self.verbose:
            print(mols)
        df = calc.pandas(mols)
        return df

    def parse_molecule_files(self):
        """Compute and cache numeric, non-degenerate descriptors for all beads.

        Populates ``self.descriptors`` with the Mordred descriptors of every
        molecule in the catalogue, keeping only numeric columns whose variance
        exceeds 0.01 (drops constant / near-constant descriptors).

        Returns:
            Tuple ``(mol_name, mol_mass)`` of the Molecules and Mass columns.
        """
        mol_name = self.molecule_dataset['Molecules']
        mol_mass = self.molecule_dataset['Mass']
        if self.verbose:
            print(self.mordred_descriptors(self.molecule_dataset['SMILES']))
            print(self.molecule_dataset['SMILES'])
        descript = self.mordred_descriptors(self.molecule_dataset['SMILES']).select_dtypes(['number'])

        # Remove descriptors with low variance
        description = descript.loc[:, descript.var() > 0.01]
        self.descriptors = description
        return mol_name, mol_mass

    def train_model(self):
        """Select descriptors and fit one RandomForest per MPiPi parameter.

        For each parameter (E, S, U, R) chooses descriptors either by greedy
        forward selection (default), one-shot on a provided feature set, or
        cumulatively over a provided ordering (``fixed_cumulative``), fits the
        model on the 24 reference beads, and records the R^2/MAE selection trace.
        Writes R2_Dataset.csv (unless suppressed), random_forest_model.pkl, and
        random_forest_columns.json into ``self.out_dir``.

        Returns:
            ``self.columns`` -- {parameter: [selected descriptor, ...]}.
        """
        descriptors = self.descriptors
        parameters = self.mpipi_dataset.iloc[:24].loc[:, ['E', 'S', 'U', 'R']].reset_index(drop=True)
        training_descriptors = self._aligned_training_descriptors(descriptors)

        # split data into input and output variables
        X_all = training_descriptors.copy()
        X_all_copy = X_all.copy()
        col_dict = {}

        df = pd.DataFrame(columns=["Parameter", "x", "R^2", "MAE", "Mordred Descriptor"])

        for column in parameters.columns:
            X_all = training_descriptors.copy()
            feature_override = None
            if column in self.feature_sets and self.feature_sets[column]:
                # Maintain order provided
                allowed = [feat for feat in self.feature_sets[column] if feat in X_all.columns]
                if allowed:
                    X_all = X_all.loc[:, allowed]
                    feature_override = allowed

            if self.verbose:
                print(column)
            X = pd.DataFrame(index=X_all.index)
            y = parameters[column]
            if column == "E":
                param = "$\\epsilon$"
            elif column == "S":
                param = "$\\sigma$"
            elif column == "U":
                param = "$\\mu$"
            else:
                param = "$r_{c}$"
            mae = 0.0
            mae_best = 0.0
            mae_arr = []
            mae_diff_arr = []
            mae_dict = {}
            col_list = []
            score = 0.0
            score_best = 0.0
            r2_score_arr = []
            r2_diff_arr = []
            r2_dict = {}
            model = RandomForestRegressor(random_state=self.random_state)
            test = True
            i = 0
            r2_diff = 1
            x_arr = []
            par_arr = []

            # If explicit features are provided for this parameter
            if feature_override and self.fixed_cumulative:
                # Cumulative training using provided order
                order = feature_override
                scores = []
                maes = []
                for i in range(1, len(order) + 1):
                    X_use = training_descriptors.loc[:, order[:i]].copy()
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_use, y, test_size=0.2, random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae_step = mean_absolute_error(y_test, y_pred)
                    score_step = r2_score(y_test, y_pred)
                    df = df._append({
                        "Parameter": param,
                        "x": i,
                        "R^2": score_step,
                        "MAE": mae_step,
                        "Mordred Descriptor": order[i-1]
                    }, ignore_index=True)
                    scores.append(score_step)
                    maes.append(mae_step)
                # Train final model on full provided set
                X_final = training_descriptors.loc[:, order].copy()
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y, test_size=0.2, random_state=self.random_state
                )
                model.fit(X_train, y_train)
                self.models[column] = model
                r2_dict[column] = scores[-1] if scores else 0.0
                mae_dict[column] = maes[-1] if maes else 0.0
                col_dict[column] = order
                X_all = X_all_copy
                continue

            # If explicit features are provided for this parameter, train once on
            # the provided set (no greedy forward selection) — legacy .txt mode
            if feature_override and not self.fixed_cumulative:
                if len(X_all.columns) == 0:
                    test = False
                else:
                    # Train model one-shot on the provided features
                    X = X_all.copy()
                    col_list = list(X.columns)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=self.random_state
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae_best = mean_absolute_error(y_test, y_pred)
                    score_best = r2_score(y_test, y_pred)
                    # Log a single point into the R2 dataset for plotting
                    df = df._append({
                        "Parameter": param,
                        "x": len(col_list),
                        "R^2": score_best,
                        "MAE": mae_best,
                        "Mordred Descriptor": ",".join(col_list)
                    }, ignore_index=True)
                    # Save model and selected columns
                    self.models[column] = model
                    r2_dict[column] = score_best
                    mae_dict[column] = mae_best
                    col_dict[column] = col_list
                    # Skip greedy loop entirely for this parameter
                    X_all = X_all_copy
                    continue

            # initial column (greedy start)
            if X_all.empty:
                test = False
            else:
                if not feature_override:
                    col_id = X_all.var().idxmax()
                    max_col = X_all.pop(col_id)
                    col_list.append(col_id)
                    X[col_id] = max_col
                else:
                    # Should not reach here because one-shot path handles feature_override
                    test = False

            # param already set above

            while len(col_list) <= len(X_all.columns) + len(feature_override or []) and test:
                if ((score_best >= 1) and (r2_diff <= 0.0)):
                    test = False

                else:
                    if feature_override:
                        if not feature_override:
                            break
                        col_id = feature_override[0]
                        feature_override = feature_override[1:]
                        if col_id not in X_all.columns:
                            continue
                        max_col = X_all.pop(col_id)
                    else:
                        if X_all.empty:
                            break
                        col_id = X_all.var().idxmax()
                        max_col = X_all.pop(col_id)

                    if score >= score_best and score > 0 and np.abs(score_best-score) <= 0.25:
                        if self.verbose:
                            print(col_id)
                        i += 1
                        x_arr.append(i)
                        col_list.append(col_id)
                        X[col_id] = max_col
                        par_arr.append(column)

                        mae_diff = np.abs(mae_best - mae)
                        r2_diff = np.abs(score_best - score)

                        r2_score_arr.append(score_best)
                        r2_diff_arr.append(r2_diff)

                        mae_arr.append(mae_best)
                        mae_diff_arr.append(mae_diff)

                        score_best = score
                        mae_best = mae

                        if self.verbose:
                            print(i)
                            print(score_best)
                            print(r2_diff)
                            print(mae_best)

                        df = df._append({"Parameter": param,
                                        "x": i,
                                        "R^2": score_best,
                                        "MAE": mae_best,
                                        "Mordred Descriptor": col_id
                                        }, ignore_index=True)

                    # split data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=self.random_state
                    )

                    # fit the model to the training data
                    model.fit(X_train, y_train)

                    # make predictions on the test data
                    y_pred = model.predict(X_test)

                    # evaluate the model using mean absolute error
                    mae = mean_absolute_error(y_test, y_pred)

                    score = r2_score(y_test, y_pred)

            self.models[column] = model

            r2_dict[column] = score_best
            mae_dict[column] = mae_best
            col_dict[column] = col_list
            X_all = X_all_copy

            # Always append a final row for this parameter
            df = df._append({
                "Parameter": param,
                "x": len(col_list),
                "R^2": score_best,
                "MAE": mae_best,
                "Mordred Descriptor": ",".join(col_list) if col_list else ""
            }, ignore_index=True)

            if self.verbose:
                print(len(X_all))
                print("Parameter "+column+": "+str(len(col_list)) + " Mordred Descriptors with r2 = " + str(score_best) + " and MAE = "+ str(mae_best))

        if self.verbose:
            print(df)
        # R2 metrics table (consumed by plotting/cg/r2_analysis.py for the R2 plot).
        if self.allow_write_r2:
            out_metrics = self.out_dir / "R2_Dataset.csv"
            out_metrics.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_metrics, sep=",")

        with open(self.out_dir / 'random_forest_model.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        with open(self.out_dir / 'random_forest_columns.json', 'w') as f:
            json.dump(col_dict, f)

        self.columns = col_dict
        return col_dict

    def model_predict(self, pretrained=None, columns_override=None):
        """Predict MPiPi parameters for the new beads and append them to the table.

        Computes descriptors (``parse_molecule_files``), then either trains fresh
        models (``train_model``) or reuses the supplied pretrained models, and
        predicts E/S/U/R for every bead beyond the first 24, appending one row per
        new bead to ``self.mpipi_dataset`` (U/V predictions rounded to integers).

        Args:
            pretrained: Optional {parameter: fitted model} to reuse instead of
                retraining; must be paired with ``columns_override``.
            columns_override: Optional {parameter: [descriptor, ...]} matching the
                pretrained models.

        Returns:
            The {parameter: [descriptor, ...]} column mapping used for prediction.
        """
        mol_names, mol_mass = self.parse_molecule_files()
        if pretrained is not None and columns_override is not None:
            self.models = pretrained
            columns = columns_override
        else:
            columns = self.train_model()
        value = 0.0

        index = 0

        for row in range(24, len(self.molecule_dataset)):
            name = mol_names[row]
            number = row + 1
            m = mol_mass[row]
            sm_row = [name, number, 0.0, 0.0, 1, 0, 0.0, 0.0, m]
            self.mpipi_dataset.loc[len(self.mpipi_dataset.index)] = sm_row
            index += 1

        for col in columns.keys():
            sm_descriptors = self.descriptors.copy().iloc[24:, :]
            sm_descriptors = sm_descriptors.loc[:, columns[col]]
            predictions = self.models[col].predict(sm_descriptors)

            for row in range(24, len(self.molecule_dataset)):
                value = predictions[row - 24]
                if col == "V" or col == "U":
                    value = int(np.round(value))
                self.mpipi_dataset.loc[row, col] = value

        self.columns = columns
        return columns


if __name__ == '__main__':
    molecule_csv = str(_DATA_DIR / "molecules.csv")
    parameters_csv = str(_DATA_DIR / "MPiPi_Parameters.csv")
    MPiPi_Model = regressor(molecule_csv, parameters_csv)
    MPiPi_Model.model_predict()
    pd.set_option('display.max_columns', None)
    print(MPiPi_Model.molecule_dataset)
    print(MPiPi_Model.mpipi_dataset)
    MPiPi_Model.mpipi_dataset.to_csv("single_parameters.csv", index=False)
