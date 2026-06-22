"""
All-atom PDB to coarse-grained per-bead PDB conversion (CG pipeline helper).

Coarse-graining helper used by the small-molecule parameterisation pipeline
(``CG_Script.py`` / ``run_full_pipeline.py``). Given a GBCG bead map and the
parent all-atom PDB, it splits each coarse-grained bead into its own PDB file,
re-numbering the constituent atoms and capping severed bonds with explicit
hydrogens so each fragment is chemically complete. These per-bead PDBs feed the
downstream descriptor (QSPR) and bond-length (``gen_bonds.py``) steps.

Inputs: a GBCG map file (``bead_id type mass charge atom_ids...`` per line) and
the corresponding all-atom PDB (HETATM + CONECT records).
Outputs: one ``<molecule>_b<n>.pdb`` per bead written under the requested path.

Not runnable as a script; instantiate ``convert_cg`` from the pipeline.
"""


class convert_cg:
    """Split an all-atom molecule into per-bead PDB fragments from a GBCG map."""

    def __init__(self, map_file, pdb_file):
        """Store the GBCG map and all-atom PDB paths for this molecule.

        Args:
            map_file: GBCG bead-map path (one bead per line:
                ``bead_id type mass charge atom_ids...``).
            pdb_file: All-atom PDB path with HETATM and CONECT records.
        """
        self.map_file = map_file
        self.pdb_file = pdb_file

    def read_map_file(self):
        """Parse the GBCG map into ``{bead_id: {type, mass, charge, atoms}}``."""
        beads = {}
        with open(self.map_file, 'r') as mapping:
            lines = mapping.readlines()
            for line in lines:
                line_split = line.split()
                bead_id = int(line_split[0])
                beads[bead_id] = {"type": (line_split[1])}
                beads[bead_id]["mass"] = (line_split[2])
                beads[bead_id]["charge"] = (line_split[3])
                beads[bead_id]["atoms"] = (line_split[4:])
        return beads

    def read_pdb_file(self):
        """Parse HETATM/CONECT records into ``{atom_id: {type, crds, connect}}``."""
        het_atoms = {}
        with open(self.pdb_file, 'r') as pdb:
            lines = pdb.readlines()
            for line in lines:
                line_split = line.split()
                if line_split[0] == 'HETATM':
                    atom_id = int(line_split[1])
                    het_atoms[atom_id] = {"type": line_split[2]}
                    het_atoms[atom_id]["crds_x"] = float(line_split[5])
                    het_atoms[atom_id]["crds_y"] = float(line_split[6])
                    het_atoms[atom_id]["crds_z"] = float(line_split[7])
                    het_atoms[atom_id]["type_end"] = line_split[10]

                elif line_split[0] == 'CONECT':
                    atom_id = int(line_split[1])
                    het_atoms[atom_id]["connect"] = line_split[2:]

        return het_atoms

    def find_atoms(self, bead, atoms):
        """Build re-numbered HETATM lines for the atoms belonging to one bead.

        Returns a tuple ``(atom_lines, mapping)`` where ``atom_lines`` maps the
        new 1-based atom index to a formatted PDB HETATM line and ``mapping``
        maps each original atom id to its new index.
        """
        atom_lines = {}
        mapping = {}
        n = 1
        atom_list = bead["atoms"]
        for atm in atom_list:
            atom_id = int(atm)
            type = (atoms[atom_id]["type"])
            x = (atoms[atom_id]["crds_x"])
            y = (atoms[atom_id]["crds_y"])
            z = (atoms[atom_id]["crds_z"])
            type_end = (atoms[atom_id]["type_end"])

            mapping[atom_id] = n

            atom_lines[n] = (
                "{:<6s}{:>5d}  {:<3s}{:1s}{:>3s}  {:>4d}{:<1s}  {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}       {:>4s}{:<s}".
                    format("HETATM", n, type, "", "UNL", 1, " ",
                           x, y, z, 1.00, 0.00, "",
                           type_end))
            n += 1
        return atom_lines, mapping

    def find_connections(self, bead, atom_lines, mapping, atoms):
        """Resolve intra-bead bonds and cap severed bonds with hydrogens.

        Bonds to atoms inside the bead are re-indexed via ``mapping``; bonds that
        cross the bead boundary get a capping H atom appended to ``atom_lines``.
        Returns ``(atom_lines, mapping, connect_lines)`` with sorted CONECT
        adjacency in ``connect_lines``.
        """
        connect_lines = {}
        atom_list = bead["atoms"]

        for atm in atom_list:
            atom_int = int(atm)
            atom_id = mapping[atom_int]
            connect = atoms[atom_int]["connect"]

            for con in connect:
                con_int = int(con)
                if con in atom_list:
                    con_id = mapping[con_int]
                    if atom_id in connect_lines.keys():
                        connect_lines[atom_id].append(con_id)
                    else:
                        connect_lines[atom_id] = [con_id]
                else:
                    con_id = len(atom_lines) + 1

                    type = ('H')
                    x = (atoms[con_int]["crds_x"])
                    y = (atoms[con_int]["crds_y"])
                    z = (atoms[con_int]["crds_z"])
                    type_end = 'H'

                    atom_lines[len(atom_lines) + 1] = (
                        "{:<6s}{:>5d}  {:<3s}{:1s}{:>3s}  {:>4d}{:<1s}  {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}{:<s}".
                            format("HETATM", con_id, type, "", "UNL", 1, " ",
                                   x, y, z, 1.00, 0.00, "",
                                   type_end))

                    if atom_id in connect_lines.keys():
                        connect_lines[atom_id].append(con_id)

                    else:
                        connect_lines[atom_id] = [con_id]

                    connect_lines[con_id] = [atom_id]

            connect_lines[atom_id].sort()

        myKeys = list(atom_lines.keys())
        myKeys.sort()
        atom_lines = {i: atom_lines[i] for i in myKeys}

        myKeys = list(connect_lines.keys())
        myKeys.sort()
        connect_lines = {i: connect_lines[i] for i in myKeys}

        return atom_lines, mapping, connect_lines

    def write_bead_pdb(self, atom_lines, connect_lines, n, name, path):
        """Write one bead's HETATM/CONECT records to ``<name>_b<n>.pdb``."""
        fid = open("{}{}_b{}.pdb".format(path, name.split("/")[-1], str(n)), "w")
        fid.write("COMPND    UNNAMED\n")
        fid.write("AUTHOR    GENERATED BY convert_cg.py\n")
        for i in atom_lines.keys():
            fid.write(atom_lines.get(i) + "\n")

        for i in connect_lines.keys():
            fid.write("CONECT" + ("{:>5d}".format(i)))
            con_atms = connect_lines.get(i)
            for j in con_atms:
                fid.write(("{:>5d}".format(j)))
            fid.write("\n")
        fid.write("MASTER        0    0    0    0    0    0    0    0    {}    0    {}    0".format(len(atom_lines),
                                                                                                    len(atom_lines)))
        fid.write("\nEND")

    def convert_cg(self, path):
        """Convert every bead in the molecule, writing one PDB per bead to ``path``."""
        beads = self.read_map_file()
        atoms = self.read_pdb_file()
        mol_name = self.pdb_file.split('.')[0]
        i = 1
        for bead in beads.values():
            atom_output = self.find_atoms(bead, atoms)
            connect_output = self.find_connections(bead, atom_output[0], atom_output[1], atoms)
            self.write_bead_pdb(connect_output[0], connect_output[2], i, mol_name, path)
            i += 1
