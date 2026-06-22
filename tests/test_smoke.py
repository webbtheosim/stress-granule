"""Clone-runnable smoke tests for the publication repository.

These tests intentionally use only small committed inputs. They do not require
the private/raw trajectory tree behind ``PYTHON_ANALYSIS``.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))


class publication_smoke_tests(unittest.TestCase):
    def test_sequence_composition_from_committed_inputs(self):
        import sequence_composition

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(sequence_composition.write_composition_csv(tmp))
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 124)
        protein_rows = [row for row in rows if row["kind"] == "protein"]
        rna_rows = [row for row in rows if row["kind"] == "rna"]
        self.assertEqual(len(protein_rows), 120)
        self.assertEqual({row["residue"] for row in rna_rows}, {"A", "U", "C", "G"})

    def test_rna_acid_counts_are_not_contaminated_by_protein_counts(self):
        import max_cluster_correlated as max_cluster

        vector = [0, 0, 0, 0, 0, 0, 1]
        acid_counts, _acid_pairs, _bond_counts = max_cluster.gen_acid_contacts(vector)
        self.assertTrue((acid_counts[:20] == 0).all())

        seq = (ANALYSIS_DIR / "sequences" / "RNA.txt").read_text()
        expected = {"A": 0, "C": 0, "U": 0, "G": 0}
        for char in seq:
            base = char.strip().upper()
            if base == "T":
                base = "U"
            if base in expected:
                expected[base] += 1
        self.assertEqual(int(acid_counts[20]), expected["A"])
        self.assertEqual(int(acid_counts[21]), expected["C"])
        self.assertEqual(int(acid_counts[22]), expected["U"])
        self.assertEqual(int(acid_counts[23]), expected["G"])

    def test_committed_quant_data_schema(self):
        required_columns = {
            "Small Molecule ID",
            "Small Molecule Name",
            "Compound Class",
            "$P_{SM}$",
            "SIG$P_{SM}$",
            "$N_{D}$",
            "SIG$N_{D}$",
            "$\\eta_{GK}$ Pa s",
            "SIG$\\eta_{GK}$ Pa s",
        }
        expected_rows = {285: 25, 290: 1, 295: 1, 300: 25, 305: 1, 310: 1, 315: 25}
        for temp, n_rows in expected_rows.items():
            path = REPO_ROOT / "data" / f"Quant_Data_{temp}K.csv"
            self.assertTrue(path.exists(), path)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(len(rows), n_rows, path.name)
            self.assertTrue(required_columns.issubset(reader.fieldnames), path.name)

    def test_stdlib_cli_help_and_data_setup_failure(self):
        for script in [
            ANALYSIS_DIR / "thin_lammpsdump_stride.py",
            ANALYSIS_DIR / "validate_rcc_outputs.py",
        ]:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

        env = dict(os.environ)
        env["DATA_ROOT"] = "/tmp/sg_condensate_missing_data_root_for_tests"
        result = subprocess.run(
            ["bash", "setup_data_links.sh"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required target not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
