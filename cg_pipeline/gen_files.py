"""
Serialise mixed MPiPi parameters into LAMMPS sys.data and sys.settings files.

Final assembly step of the small-molecule parameterisation pipeline, imported by
``run_full_pipeline.py`` (and runnable standalone via ``__main__``). It reads the
per-bead force-field parameters, drives ``mix_params.mixer`` to obtain the full
heterotypic interaction matrix, and writes the LAMMPS input files needed to
simulate the parameterised system: atom/bond topology, masses, box, charges,
bonded coefficients, and pairwise (Wang-Frenkel) + Coulombic (Debye) coeffs.
Per-molecule ``.mol`` template files are written on demand via
:meth:`write_mol_file`.

Inputs: a per-bead parameter CSV (``name, num, eps, sig, v, mu, rc, q, m``), a
template LAMMPS ``sys.data`` (for atom/bond coordinates), a ``bonds.txt`` of
coarse-grained bond lengths, and the reference WF cross-parameter file.
Outputs: ``sys.data``, ``sys.settings``, ``mix_params_JK.txt`` (SM-protein pair
coeffs), an optional ``mixing_rules_used.txt`` audit, and ``.mol`` files in
``mol_output_dir``.

CLI: ``python gen_files.py <parameters.csv> <sys.data> <bonds.txt>``
"""

import logging
from pathlib import Path

import mix_params
import numpy as np
import sys

_DATA_DIR = Path(__file__).resolve().parent / "data"
LOGGER = logging.getLogger(__name__)


class settings:
    """Build LAMMPS sys.data / sys.settings from mixed MPiPi parameters."""

    def __init__(self, parameters_csv, sys_data, bonds, wf_parameters=str(_DATA_DIR / "WF_Parameters.txt"),
                 rules_override=None, output_sys_data='sys.data', output_sys_settings='sys.settings',
                 mix_params_output='mix_params_JK.txt', mol_output_dir='CG_MOL_Files'):
        """Load parameters, run mixing, and write sys.data / sys.settings.

        The full generation chain runs in the constructor: parse parameters,
        mix heterotypic interactions, build charge/pairwise/bond lines, and emit
        the LAMMPS data and settings files.

        Args:
            parameters_csv: Per-bead parameter CSV
                (``name, num, eps, sig, v, mu, rc, q, m``).
            sys_data: Template LAMMPS ``sys.data`` providing atom/bond geometry.
            bonds: ``bonds.txt`` of coarse-grained bond lengths.
            wf_parameters: Reference WF cross-parameter file for mixing-rule
                selection.
            rules_override: Optional ``{param: rule}`` forcing combining rules.
            output_sys_data: Output path for the generated ``sys.data``.
            output_sys_settings: Output path for the generated ``sys.settings``.
            mix_params_output: Output path for the SM-protein pair-coeff dump.
            mol_output_dir: Directory for per-molecule ``.mol`` files.
        """
        self.parameters_csv = Path(parameters_csv)
        self.sys_data_template = Path(sys_data)
        self.bonds_source = Path(bonds)
        self.wf_parameters = Path(wf_parameters)
        self.output_sys_data = Path(output_sys_data)
        self.output_sys_settings = Path(output_sys_settings)
        self.mix_params_output = Path(mix_params_output)
        self.mol_output_dir = Path(mol_output_dir)
        self.mol_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_sys_data.parent.mkdir(parents=True, exist_ok=True)
        self.output_sys_settings.parent.mkdir(parents=True, exist_ok=True)
        self.mix_params_output.parent.mkdir(parents=True, exist_ok=True)

        self.particles_num = {}
        self.particles_name = {}
        self.parse_file(self.parameters_csv)

        self.mixer = mix_params.mixer(self.parameters_csv, parameters_file=self.wf_parameters,
                                      rules_override=rules_override)
        self.mixer.calc_sm_mixing()
        self.parameters = self.mixer.sm_parameters
        # If rules were auto-selected, record them near the outputs
        try:
            if rules_override is None and getattr(self.mixer, 'used_rules', None):
                out_rules = self.mix_params_output.parent / 'mixing_rules_used.txt'
                with open(out_rules, 'w') as fh:
                    def header(k):
                        return {'eps': 'epsilon', 'sig': 'sigma', 'v': 'v', 'mu': 'mu', 'rc': 'rc'}.get(k, k)
                    # Write block format
                    order = ['eps', 'sig', 'v', 'mu', 'rc']
                    for key in order:
                        if key in self.mixer.used_rules:
                            fh.write(f"{header(key)}\n")
                            fh.write(f"{self.mixer.used_rules[key]}\n\n")
        except Exception:
            pass

        self.pairwise_lines = []
        self.coulombic_lines = []
        self.get_pairwise()

        self.charge_lines = []
        self.get_charges()

        self.bonds = {}
        self.bond_num = 0
        self.parse_bonds(self.bonds_source)

        self.bond_lines = []
        self.bond_map = {}
        self.bond_lines = self.gen_bonds()

        self.gen_sys_data(self.sys_data_template)
        self.gen_sys_settings()

    def parse_file(self, parameters_csv):
        """Load per-bead parameters into ``particles_num`` / ``particles_name``."""
        with open(parameters_csv, 'r') as params:
            lines = params.readlines()[1:]
            for line in lines:
                line_split = line.split(",")
                name = line_split[0]
                particle_number = int(line_split[1])
                eps = float(line_split[2])
                sig = float(line_split[3])
                v = int(line_split[4])
                mu = int(line_split[5])
                rc = float(line_split[6])
                q = float(line_split[7])
                m = float(line_split[8])
                self.particles_num[particle_number] = {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc, "q": q, "m": m}
                self.particles_name[name] = {"num": particle_number, "eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc, "q": q, "m": m}

    def get_mass_lines(self):
        """Return LAMMPS ``Masses`` section lines, one per bead type."""
        mass_lines = []
        for key in sorted(self.particles_num.keys()):
            mass_lines.append("{:<3d}{:<8f}{:<4s}\n".format(key, self.particles_num[key]["m"], " # "+str(key)))
        return mass_lines

    def gen_sys_data(self, sys_data):
        """Write the LAMMPS ``sys.data`` (header, box, masses, atoms, bonds).

        Copies atom and bond records from the template ``sys_data`` while
        substituting the regenerated atom-type count, bond-type count, masses,
        and fixed simulation box, and assigning each bond a type (1 for
        protein/RNA beads <= 20, 2 for nucleotide-backbone beads 21-24).
        """
        fid = open(self.output_sys_data, "w")
        atom_dict = {}
        with open(sys_data, 'r') as params:
            lines = params.readlines()
            misc = lines[0:3]
            fid.writelines(misc)
            fid.write("{} atom types\n".format(len(self.particles_num.keys())))
            misc = lines[4:5]
            fid.writelines(misc)
            fid.writelines("{} bond types\n".format(self.bond_num))
            misc = lines[6:lines.index("Masses\n")+1]
            fid.write("\n-0.9000000000000000e+03   1.5000000000000000e+03    xlo xhi\n")
            fid.write("-0.9000000000000000e+03   1.5000000000000000e+03    ylo yhi\n")
            fid.write("-0.9000000000000000e+03   1.5000000000000000e+03    zlo zhi\n")
            fid.write("\nMasses\n\n")
            masses = self.get_mass_lines()
            fid.writelines(masses)
            atoms = lines[lines.index("Atoms\n")+2: lines.index("Bonds\n")-1]
            bonds = lines[lines.index("Bonds\n")+2:]

        fid.write("\nAtoms\n\n")

        for i in atoms:
            fid.write(i)
            atom_num = int(i.split()[0])
            atom_type = int(i.split()[2])
            atom_dict[atom_num] = atom_type

        fid.write("\nBonds\n\n")
        for i in bonds:
            atm1 = int(i.split()[2])
            atm2 = int(i.split()[3])
            atm1_type = int(atom_dict[atm1])
            b_type = 0
            if atm1_type <= 20:
                b_type = 1
            elif atm1_type in range(21, 25):
                b_type = 2
            fid.write("{:<6d}{:<2d}{:<6d}{:<6d}\n".format(int(i.split()[0]), b_type, atm1, atm2))
        fid.close()

    def gen_sys_settings(self):
        """Write the LAMMPS ``sys.settings`` (charges, bonds, pair styles, coeffs)."""
        fid = open(self.output_sys_settings, "w")
        fid.write("# MPiPi Settings File\n\n# Charges\n")
        fid.writelines(self.charge_lines)
        fid.write("\n# Bonds\n")
        fid.writelines(self.bond_lines)
        fid.write("\n# Pair Style\n")
        fid.write("pair_style  hybrid/overlay wf/cut 25.0 coul/debye 0.126 0.0")
        fid.write("\n\n# Pairwise Interactions\n")
        fid.writelines(self.pairwise_lines)
        fid.write("\n# Coulombic Interactions\n")
        fid.writelines(self.coulombic_lines)

    def get_charges(self):
        """Build ``set charge`` and Debye ``pair_coeff`` lines for charged beads."""
        for key1 in self.particles_num.keys():
            charge1 = np.round(self.particles_num[key1]["q"], 3)
            if charge1 != 0:
                self.charge_lines.append("set type {:<3d} charge {:>3f}\n".format(key1, charge1))
                for key2 in self.parameters[int(key1)].keys():
                    charge2 = np.round(self.particles_num[key2]["q"], 3)
                    if charge2 != 0:
                        self.coulombic_lines.append("pair_coeff {:<3d}{:<3d}coul/debye 35.0\n".format(int(key1), int(key2)))

    def get_pairwise(self):
        """Build Wang-Frenkel ``pair_coeff`` lines for every interaction pair.

        Populates ``self.pairwise_lines`` for all pairs (written to sys.settings)
        and, separately, dumps only the small-molecule-protein pairs (bead types
        key1 <= 20 and key2 >= 25) to ``mix_params_output``.
        """
        sm_protein_lines = []
        for key1 in self.parameters.keys():
            for key2 in self.parameters[key1].keys():
                eps = self.parameters[key1][key2]["eps"]
                sig = self.parameters[key1][key2]["sig"]
                v = self.parameters[key1][key2]["v"]
                mu = self.parameters[key1][key2]["mu"]
                rc = self.parameters[key1][key2]["rc"]

                line = "pair_coeff {:<3d}{:<3d}wf/cut {:>8f} {:>8f} {:<2d}{:<3d}{:<8f}\n".format(
                    int(key1), int(key2), eps, sig, v, mu, rc)
                self.pairwise_lines.append(line)
                if key1 <= 20 and key2 >= 25:
                    sm_protein_lines.append(line)

        with open(self.mix_params_output, "w") as f:
            for line in sm_protein_lines:
                f.write(line)

    def parse_bonds(self, bonds):
        """Build the bead-pair bond table from defaults plus the bonds file.

        Seeds standard protein (3.810 A) and nucleotide (5.000 A) backbone bonds,
        then adds small-molecule bond lengths read from ``bonds`` for bead pairs
        present in the parameter table. Sets ``self.bond_num`` to the total
        number of distinct bond types.
        """
        for key1 in range(1, 21):
            for key2 in range(1, 21):
                if key2 >= key1:
                    if key1 in self.bonds:
                        self.bonds[key1][key2] = {"l": 3.810, "k": 9.600}
                    else:
                        self.bonds[key1] = {key2: {"l": 3.810, "k": 9.600}}

        for key1 in range(21, 25):
            for key2 in range(21, 25):
                if key2 >= key1:
                    if key1 in self.bonds:
                        self.bonds[key1][key2] = {"l": 5.000, "k": 9.600}
                    else:
                        self.bonds[key1] = {key2: {"l": 5.000, "k": 9.600}}

        self.bond_num = 2

        with open(bonds, 'r') as bond:
            lines = bond.readlines()
            for line in lines:
                line_split = line.split()
                if len(line_split) < 3:
                    continue
                name1, name2, length_s = line_split[0], line_split[1], line_split[2]
                # Skip bonds for molecules not present in parameters table
                if name1 not in self.particles_name or name2 not in self.particles_name:
                    continue
                key1 = self.particles_name[name1]["num"]
                key2 = self.particles_name[name2]["num"]
                length = np.round(float(length_s), 3)
                self.bond_num += 1
                if key1 in self.bonds:
                    self.bonds[key1][key2] = {"l": length, "k": 9.600}
                else:
                    self.bonds[key1] = {key2: {"l": length, "k": 9.600}}

    def gen_bonds(self):
        """Return LAMMPS ``bond_coeff`` lines and fill ``self.bond_map``.

        Emits the two standard backbone bond types (protein, nucleotide) followed
        by one type per distinct small-molecule (>= 25) bead-pair bond, recording
        each pair's bond-type index in ``self.bond_map``.
        """
        bond_array = []
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(1, 9.600, 3.810))
        bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(2, 9.600, 5.000))
        count = 3
        for key1 in self.bonds.keys():
            for key2 in self.bonds[key1].keys():
                if key1 >= 25 and key2 >= 25:
                    length = self.bonds[key1][key2]["l"]
                    bond_array.append("bond_coeff {:<3d} {:<5f} {:<5f}\n".format(count, 9.600, length))
                    if key1 in self.bond_map:
                        self.bond_map[key1][key2] = count
                    else:
                        self.bond_map[key1] = {key2: count}
                    count += 1
        return bond_array

    def parse_sm_pdb(self, sm):
        """Parse a coarse-grained SM PDB into bead and bond dicts for a .mol file.

        Returns ``(mol_name, beads, bonds)`` where ``beads`` maps the bead id to
        its type and coordinates and ``bonds`` maps a bond index to its bond type
        and the two beads it connects. Raises ``KeyError`` if a bead has no entry
        in the parameter table.
        """
        sm_path = Path(sm)
        mol_name = sm_path.stem
        beads = {}
        bonds = {}
        bond_num = 1
        with open(sm, 'r') as mol:
            lines = mol.readlines()
            for line in lines[2:]:
                element = line.split()
                if element[0] == 'HETATM':
                    bead_name = f"{mol_name}_b{element[1]}"
                    if bead_name not in self.particles_name:
                        raise KeyError(bead_name)
                    beads[int(element[1])] = {"type": int(self.particles_name[bead_name]["num"]), "x": float(element[5]), "y": float(element[6]), "z": float(element[7])}
                elif element[0] == 'CONECT':
                    atm1 = element[1]
                    for atm2 in element[2:]:
                        if atm2 > atm1:
                            key1 = beads[int(atm1)]["type"]
                            key2 = beads[int(atm2)]["type"]
                            btype = self.bond_map[key1][key2]
                            bonds[bond_num] = {"btype": btype, "atm1": int(atm1), "atm2": int(atm2)}
                            bond_num += 1
        return mol_name, beads, bonds

    def write_mol_file(self, sm):
        """Write a LAMMPS ``.mol`` template for one small molecule.

        Parses the coarse-grained SM PDB and emits ``Coords``, ``Types`` and (if
        present) ``Bonds`` sections to ``<mol_output_dir>/<mol>.mol``. Molecules
        with beads missing from the parameter table are skipped with a warning.
        """
        try:
            mol_name, beads, bonds = self.parse_sm_pdb(sm)
        except KeyError as exc:
            LOGGER.warning("Skipping molecule %s due to missing parameters for bead %s", sm, exc)
            return
        coord_lines = []
        type_lines = []

        for key in beads.keys():
            coord_lines.append("{:<3d} {:>9f} {:>9f} {:<9f}\n".format(key, beads[key]["x"], beads[key]["y"], beads[key]["z"]))
            type_lines.append("{:<3d} {:>3d}\n".format(key, (int(beads[key]["type"])-1)))

        mol_path = self.mol_output_dir / f"{mol_name.split('.')[0]}.mol"
        fid = open(mol_path, "w")
        fid.write("# LAMMPS Molecule Input File\n\n")
        fid.write("{:<3d} atoms\n".format(len(beads)))
        fid.write("{:<3d} bonds\n".format(len(bonds)))

        fid.write("\nCoords\n\n")
        fid.writelines(coord_lines)

        fid.write("\nTypes\n\n")
        fid.writelines(type_lines)

        if len(bonds.keys()) > 0:
            fid.write("\nBonds\n\n")
            for key in bonds.keys():
                fid.write("{:<3d} {:<3d} {:<3d} {:<3d}\n".format(key, bonds[key]["btype"], bonds[key]["atm1"], bonds[key]["atm2"]))


if __name__ == '__main__':
    parameter_csv = sys.argv[1]
    sys_data_file = sys.argv[2]
    bond_file = sys.argv[3]

    genset = settings(parameter_csv, sys_data_file, bond_file)
