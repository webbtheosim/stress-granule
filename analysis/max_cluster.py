"""Biopolymer / amino-acid composition counting from maximal condensate clusters.

Pipeline role
-------------
Step 2 of the stress-granule (SG) MD analysis pipeline. Runs after
``average_simulations.py`` (Step 1) and before ``system_analysis.py``
(Step 3). For each system it reads the per-window maximal-continuous-cluster
residue masks written by Step 0, maps cluster residue IDs to the seven
biopolymer species (G3BP1, PABP1, TIA1, TTP, FUS, TDP43, RNA), and converts
those species counts into the per-species, per-amino-acid, and per-bond
``n_i * n_j`` pair-count matrices used downstream for contact-map normalization.

Key inputs
----------
- ``ANALYSIS_{SG,DSM,NDSM}/Max_Continuous_Cluster_<sm>_<t>.txt`` cluster masks
  (one per time window; integer or float timestamp labels both accepted).
- Per-species sequence files under ``analysis/sequences/`` (``<SPECIES>_seq.txt``
  and ``RNA.txt`` by default) giving the amino-acid / nucleotide sequence of
  each chain.
- CLI flags: ``--path`` (TEMP_XXX dir), ``--folder`` (output prefix, default
  ``CLASSIFY``), ``--temp``, ``--tmin``, ``--dt``, ``--tmax``, ``--use-lists``,
  ``--plot-only``.

Key outputs (written under ``{folder}_{temp}_{dt}_{tmin}_{tmax}/ANALYSIS_*_AVE/``)
-------------------------------------------------------------------------------
- ``BioNumDF_<sm>.csv``, ``BioNum_<sm>.csv`` : per-species cluster counts.
- ``BioPolNum_<sm>.csv``, ``AcidNum_<sm>.csv``, ``AcidPolNum_<sm>.csv``,
  ``BondNum_<sm>.csv`` : ``n_i * n_j`` pair-count matrices and amino-acid vectors,
  plus class-aggregated (DSM / NDSM) variants.
- ``CM_NORM/MAPS/*_SYSTEM.csv`` : generic whole-system normalization matrices.

CLI invocation
--------------
    python max_cluster.py --path TEMP_300 --folder CLASSIFY --temp 300 \
        --tmin 50 --dt 50 --tmax 2000 --seq-dir analysis/sequences
"""
import numpy as np
import pandas as pd
import sys
import os
import argparse
import re

DEFAULT_SEQ_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequences")


def _time_labels(t):
    """Return candidate timestamp strings (integer- and float-style) for window ``t``."""
    labels = []
    try:
        tf = float(t)
        if tf.is_integer():
            labels.append(str(int(tf)))
        labels.append(str(tf))
    except Exception:
        labels.append(str(t))

    seen = set()
    ordered = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _resolve_max_cluster_file(category, sm, t):
    """Locate the maximal-cluster mask for ``sm`` at window ``t``, trying current and legacy name forms.

    Returns the existing file path, or ``None`` if no integer/float timestamp
    variant (including the legacy ``_acid`` -> ``acid`` spelling) is found.
    """
    for label in _time_labels(t):
        primary = "{}/Max_Continuous_Cluster_{}_{}.txt".format(category, sm, label)
        legacy = "{}/Max_Continuous_Cluster_{}_{}.txt".format(
            category, sm.replace("_acid", "acid"), label
        )
        if os.path.isfile(primary):
            return primary
        if os.path.isfile(legacy):
            return legacy
    return None


def _parse_max_cluster_resids(file_path):
    """Parse a 'resid X or resid Y ...' cluster mask file into a list of integer residue IDs."""
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    nums = re.findall(r"resid\s+(\d+)", text)
    if not nums:
        nums = re.findall(r"\b(\d+)\b", text)
    return [int(x) for x in nums]


def _read_sequence(seq_dir, species):
    """Read one protein/RNA sequence from the shipped sequence directory."""
    filename = "RNA.txt" if species == "RNA" else "{}_seq.txt".format(species)
    path = os.path.join(seq_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing sequence file for {}: {}".format(species, path))
    with open(path, "r", encoding="utf-8") as handle:
        return [char.upper() for char in handle.read() if char.isalpha()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate biopolymer and amino acid counts from maximal clusters')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', default='CLASSIFY', help='Output folder prefix (default: CLASSIFY)')
    parser.add_argument('--temp', type=int, required=True, help='Temperature in Kelvin (e.g., 300)')
    parser.add_argument('--tmin', type=int, required=True, help='Start frame')
    parser.add_argument('--dt', type=int, required=True, help='Frame stride')
    parser.add_argument('--tmax', type=int, required=True, help='End frame')
    parser.add_argument('--use-lists', action='store_true',
                        help='Use dsm_list.txt and ndsm_list.txt instead of default SM lists')
    parser.add_argument('--seq-dir', default=DEFAULT_SEQ_BASE_DIR,
                        help='Directory with <PROT>_seq.txt and RNA.txt sequence inputs '
                             '(default: shipped analysis/sequences).')
    parser.add_argument('--plot-only', action='store_true', help='No-op for this count script; exits after verifying the existing output root')

    args = parser.parse_args()

    # Validate path exists
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    args.path = os.path.abspath(args.path)
    args.seq_dir = os.path.abspath(args.seq_dir)
    if not os.path.isdir(args.seq_dir):
        print(f"Error: sequence directory {args.seq_dir} does not exist")
        sys.exit(1)
    if args.plot_only:
        output_root = os.path.join(args.path, f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}")
        if not os.path.isdir(output_root):
            print(f"Error: Analysis root {output_root} does not exist; run full MaxCluster first.")
            sys.exit(1)
        print(f"[plot-only] MaxCluster has no plot products; existing root verified: {output_root}")
        sys.exit(0)

    # Check for input directories and warn if missing
    for cat in ['SG', 'DSM', 'NDSM']:
        input_dir = os.path.join(args.path, f"ANALYSIS_{cat}")
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found - {cat} analysis will be skipped")

    # Change to TEMP_* directory
    os.chdir(args.path)

    # Output directory: {path}/{folder}_{temp}_{dt}_{tmin}_{tmax}
    output_path = f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}"

    def max_cluster_ave(folder, cat_save, sm, tmin, tmax, dt):
        """Count per-species cluster membership over all time windows for system ``sm``.

        Reads each window's maximal-cluster mask, maps residue IDs to the seven
        biopolymer species via fixed resid ranges, writes the per-window mean/SEM
        species table (``BioNumDF_<sm>.csv``), and returns a dict of average
        per-species counts keyed by species index (0-6) for downstream pair-count
        generation.
        """
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        biopolymer_cols = ["SG", "TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]
        residues = []
        for i in range(tmin, tmax, dt):
            file_path = _resolve_max_cluster_file(category, sm, i)
            if file_path is None:
                raise FileNotFoundError(
                    f"Max_Continuous_Cluster file missing for {sm} at t={i}: "
                    f"tried integer and float timestamp variants"
                )

            residues.append(_parse_max_cluster_resids(file_path))

        count = 0
        res_list = []
        rows = []
        for res in residues:
            if len(res) > 2:
                res_list.extend(res)
                count += 1

        step = 1
        for res in residues:
            biopolymer_name = {
                "Step": 0,
                "SG": 0,
                "TDP43": 0,
                "FUS": 0,
                "TIA1": 0,
                "G3BP1": 0,
                "PABP1": 0,
                "TTP": 0,
                "RNA": 0
            }
            biopolymer_name["Step"] = step
            if len(res) > 2:
                for i in res:
                    biopolymer_name["SG"] += 1
                    if i <= 33:
                        biopolymer_name["G3BP1"] += 1
                    elif 33 < i <= 49:
                        biopolymer_name["PABP1"] += 1
                    elif 49 < i <= 65:
                        biopolymer_name["TIA1"] += 1
                    elif 65 < i <= 81:
                        biopolymer_name["TTP"] += 1
                    elif 81 < i <= 97:
                        biopolymer_name["FUS"] += 1
                    elif 97 < i <= 113:
                        biopolymer_name["TDP43"] += 1
                    elif 113 < i <= 135:
                        biopolymer_name["RNA"] += 1
                step += 1
                rows.append(biopolymer_name)

        if rows:
            df = pd.DataFrame.from_records(rows)
            for col in ["Step"] + biopolymer_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric = df[biopolymer_cols]
            row_mean = numeric.mean(axis=0).reindex(biopolymer_cols).to_numpy(dtype=float)
            row_sem = numeric.sem(axis=0).reindex(biopolymer_cols).to_numpy(dtype=float)
        else:
            df = pd.DataFrame(columns=["Step"] + biopolymer_cols)
            row_mean = np.zeros(len(biopolymer_cols), dtype=float)
            row_sem = np.full(len(biopolymer_cols), np.nan, dtype=float)
        df_mean = pd.DataFrame(columns = ["Biopolymer"])
        df_mean["Biopolymer"] = biopolymer_cols
        df_mean["Mean"] = row_mean
        df_mean["SEM"] = row_sem

        df_mean.to_csv("{}/{}/BioNumDF_{}.csv".format(folder,cat_save,sm), index=False)

        biopolymer_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0
        }

        biopolymer_name = {
            "G3BP1": 0,
            "PABP1": 0,
            "TTP": 0,
            "TIA1": 0,
            "TDP43": 0,
            "FUS": 0,
            "RNA": 0
        }

        for i in res_list:
            if i <= 33:
                #G3BP1 SIM (0)
                #G3BP1 CON (0)
                biopolymer_count[0] += 1
                biopolymer_name["G3BP1"] += 1
            elif 33 < i <= 49:
                # PABP1 SIM (1)
                # PABP1 CON (1)
                biopolymer_count[1] += 1
                biopolymer_name["PABP1"] += 1
            elif 49 < i <= 65:
                # TIA1 SIM (2)
                # TIA1 CON (3)
                biopolymer_count[3] += 1
                biopolymer_name["TIA1"] += 1
            elif 65 < i <= 81:
                # TTP SIM (3)
                # TTP CON (2)
                biopolymer_count[2] += 1
                biopolymer_name["TTP"] += 1
            elif 81 < i <= 97:
                # FUS SIM (4)
                # FUS CON (5)
                biopolymer_count[5] += 1
                biopolymer_name["FUS"] += 1
            elif 97 < i <= 113:
                # TDP43 SIM (5)
                # TDP43 CON (4)
                biopolymer_count[4] += 1
                biopolymer_name["TDP43"] += 1
            elif 113 < i <= 135:
                # RNA SIM (6)
                # RNA CON (6)
                biopolymer_count[6] += 1
                biopolymer_name["RNA"] += 1

        if count > 0:
            for i in biopolymer_count.keys():
                biopolymer_count[i] /= count
        else:
            for i in biopolymer_count.keys():
                biopolymer_count[i] = 0.0

        print(sm)
        return biopolymer_count

    def gen_ave_biopolymer_ni_nj(path, biopolymer_num, sm, category):
        """Write the 7x7 species pair-count matrix n_i*n_j (n_i*(n_i-1)/2 on the diagonal) to ``BioPolNum_<sm>.csv``."""
        n = 7
        bio_arr = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    bio_arr[i,j] = (biopolymer_num[i]*(biopolymer_num[j]-1))/2
                else:
                    bio_arr[i,j] = (biopolymer_num[i] * (biopolymer_num[j]))
        np.savetxt("{}/{}/BioPolNum_{}.csv".format(path,category,sm), bio_arr, delimiter=",")

    def gen_agg_biopolymer_ni_nj(sm_list, path):
        """Average the per-system ``BioPolNum`` species pair-count matrices over ``sm_list`` into a class aggregate."""
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BioPolNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BioPolNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/BioPolNum_{}.csv".format(path, category, category), index=False, header=False)



    def gen_acid_contacts(biopolymer_num_dict, path, category, sm):
        """Build the 24x24 bonded amino-acid/nucleotide pair-count matrix and per-acid totals for system ``sm``.

        Walks each species' sequence (and the RNA nucleotide sequence), tallies
        intra-chain neighbor pairs within a short sequence window weighted by the
        chain copy number, writes the symmetric bond matrix to ``BondNum_<sm>.csv``,
        and returns ``(acid_array, biopolymer_acids)`` where ``acid_array`` is the
        per-acid bead count vector and ``biopolymer_acids`` is the per-species
        chain length.
        """
        biopolymer_acids = {
            "G3BP1": 0,
            "PABP1": 0,
            "TTP": 0,
            "TIA1": 0,
            "TDP43": 0,
            "FUS": 0,
            "RNA": 0
        }
        n = 24
        count = 0
        count_bio = 0
        bond_array = np.zeros((n, n))
        biopolymers = ["G3BP1","TDP43","FUS","PABP1","TIA1","TTP"]

        acid_array = np.zeros((n))

        aa_dict = {'M': 0,
                   'G': 1,
                   'K': 2,
                   'T': 3,
                   'R': 4,
                   'A': 5,
                   'D': 6,
                   'E': 7,
                   'Y': 8,
                   'V': 9,
                   'I': 10,
                   'Q': 11,
                   'W': 12,
                   'F': 13,
                   'S': 14,
                   'H': 15,
                   'N': 16,
                   'P': 17,
                   'C': 18,
                   'L': 19,
                   }

        na_dict = {'A': 20,
                   'C': 21,
                   'T': 22,
                   'U': 22,
                   'G': 23,
                  }

        biopolymer_num = {
            "G3BP1": biopolymer_num_dict[0],
            "PABP1": biopolymer_num_dict[1],
            "TTP": biopolymer_num_dict[2],
            "TIA1": biopolymer_num_dict[3],
            "TDP43": biopolymer_num_dict[4],
            "FUS": biopolymer_num_dict[5],
            "RNA": biopolymer_num_dict[6]
        }


        for i in biopolymers:
            bio_array = np.zeros((n, n))
            na_array = np.zeros((n))
            aa = _read_sequence(args.seq_dir, i)

            count += len(aa) * biopolymer_num[i]
            biopolymer_acids[i] = len(aa)


            for j in range(len(aa)):
                for k in range(j+1,j+4):
                    try:
                        if aa_dict[aa[j]] <= aa_dict[aa[k]]:
                            bio_array[aa_dict[aa[j]], aa_dict[aa[k]]] += 1
                        else:
                            bio_array[aa_dict[aa[k]], aa_dict[aa[j]]] += 1
                    except:
                        pass

            for j in range(len(aa)):
                na_array[aa_dict[aa[j]]] += 1

            acid_array += na_array * biopolymer_num[i]
            bond_array += bio_array * biopolymer_num[i]

        bio_array = np.zeros((n, n))
        rna_counts = np.zeros((n))
        na = _read_sequence(args.seq_dir, "RNA")

        count += len(na) * biopolymer_num["RNA"]
        biopolymer_acids["RNA"] = len(na)

        for j in range(len(na)):
            for k in range(j + 1, j + 3):
                try:
                    if na_dict[na[j]] <= na_dict[na[k]]:
                        bio_array[na_dict[na[j]], na_dict[na[k]]] += 1
                    else:
                        bio_array[na_dict[na[k]], na_dict[na[j]]] += 1
                except:
                    pass
        bond_array += bio_array * biopolymer_num["RNA"]

        for j in range(len(na)):
            rna_counts[na_dict[na[j]]] += 1

        acid_array += rna_counts * biopolymer_num["RNA"]

        for i in range(n):
            for j in range(i + 1, n):
                bond_array[j][i] = bond_array[i][j]

        print("Biopolymers: {}".format(count))
        print("Acids: {}".format(np.sum(bond_array)))
        np.savetxt("{}/{}/BondNum_{}.csv".format(path, category, sm), bond_array, delimiter=",")

        return acid_array, biopolymer_acids


    def gen_acid_ni_nj(path, category, sm, nAAcids):
        """Write the 24x24 amino-acid pair-count matrix n_i*n_j from the per-acid count vector to ``AcidPolNum_<sm>.csv``."""
        n = 24
        bio_arr = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                bio_arr[i, j] = nAAcids[i] * nAAcids[j]

        np.savetxt("{}/{}/AcidPolNum_{}.csv".format(path, category, sm), bio_arr, delimiter=",")


    def gen_agg_acid_ni_nj(sm_list, path):
        """Average the per-system ``AcidPolNum`` amino-acid pair-count matrices over ``sm_list`` into a class aggregate."""
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidPolNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidPolNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/AcidPolNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_bond_ni_nj(sm_list, path):
        """Average the per-system ``BondNum`` bonded-pair matrices over ``sm_list`` into a class aggregate."""
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BondNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BondNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/BondNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_bio_count(sm_list, path):
        """Average the per-system ``BioNum`` species count vectors over ``sm_list`` into a class aggregate."""
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/BioNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/BioNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/BioNum_{}.csv".format(path, category, category), index=False, header=False)

    def gen_agg_acid_count(sm_list, path):
        """Average the per-system ``AcidNum`` per-acid count vectors over ``sm_list`` into a class aggregate."""
        category = "{}".format(sm_list[0].split("_")[0].upper())
        df = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidNum_{}.csv".format(path, category, sm_list[0]), header=None)
        count = 1
        for sm in sm_list[1:]:
            try:
                df_temp = pd.read_csv("{}/ANALYSIS_{}_AVE/AcidNum_{}.csv".format(path, category, sm), header=None)
                df = df.add(df_temp, fill_value=0)
                count += 1
            except:
                print("{}/ANALYSIS/Contacts_Mean_{}.csv File Missing".format(path, sm))

        df = df.div(count, fill_value = 0)
        df.to_csv("{}/ANALYSIS_{}_AVE/AcidNum_{}.csv".format(path, category, category), index=False, header=False)


    # SM lists
    if args.use_lists:
        dsm = []
        ndsm = []
        if os.path.exists(f'{output_path}/dsm_list.txt'):
            with open(f'{output_path}/dsm_list.txt', 'r') as f:
                dsm = [line.strip() for line in f.readlines()]
        if os.path.exists(f'{output_path}/ndsm_list.txt'):
            with open(f'{output_path}/ndsm_list.txt', 'r') as f:
                ndsm = [line.strip() for line in f.readlines()]
    else:
        # Updated SM names (legacy variants still supported elsewhere)
        dsm = [
            "dsm_anisomycin",
            "dsm_daunorubicin",
            "dsm_dihydrolipoic_acid",
            "dsm_hydroxyquinoline",
            "dsm_lipoamide",
            "dsm_lipoic_acid",
            "dsm_mitoxantrone",
            "dsm_pararosaniline",
            "dsm_pyrivinium",
            "dsm_quinicrine",
        ]
        ndsm = [
            "ndsm_dmso",
            "ndsm_valeric_acid",
            "ndsm_ethylenediamine",
            "ndsm_propanedithiol",
            "ndsm_hexanediol",
            "ndsm_diethylaminopentane",
            "ndsm_aminoacridine",
            "ndsm_anthraquinone",
            "ndsm_acetylenapthacene",
            "ndsm_anacardic",
        ]

    # Process SG (skip if input analysis folder missing)
    if os.path.isdir("ANALYSIS_SG"):
        print("Processing SG...")
        sm = "sg_X"
        category = "ANALYSIS_SG_AVE"
        biopolymer_count = max_cluster_ave(output_path, category, sm, args.tmin, args.tmax, args.dt)

        df = pd.DataFrame(biopolymer_count, index=[0])
        df.to_csv("{}/{}/BioNum_{}.csv".format(output_path, category, sm), index=False, header=False)
        gen_ave_biopolymer_ni_nj(output_path, biopolymer_count, sm, category)
        nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, output_path, category, sm)
        np.savetxt("{}/{}/AcidNum_{}.csv".format(output_path, category, sm), np.array(nAAcids), delimiter=",")
        gen_acid_ni_nj(output_path, category, sm, nAAcids)
    else:
        print("Skipping SG: ANALYSIS_SG not found")

    # Process DSM (skip if input analysis folder missing)
    if os.path.isdir("ANALYSIS_DSM"):
        print("Processing DSM...")
        category = "ANALYSIS_DSM_AVE"
        for i in dsm:
            biopolymer_count = max_cluster_ave(output_path, category, i, args.tmin, args.tmax, args.dt)
            df = pd.DataFrame(biopolymer_count, index=[0])
            df.to_csv("{}/{}/BioNum_{}.csv".format(output_path, category, i), index=False, header=False)
            gen_ave_biopolymer_ni_nj(output_path, biopolymer_count, i, category)
            nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, output_path, category, i)
            np.savetxt("{}/{}/AcidNum_{}.csv".format(output_path, category, i), np.array(nAAcids), delimiter=",")
            gen_acid_ni_nj(output_path, category, i, nAAcids)

        gen_agg_biopolymer_ni_nj(dsm, output_path)
        gen_agg_acid_ni_nj(dsm, output_path)
        gen_agg_bond_ni_nj(dsm, output_path)
        gen_agg_bio_count(dsm, output_path)
        gen_agg_acid_count(dsm, output_path)
    else:
        print("Skipping DSM: ANALYSIS_DSM not found")

    # Process NDSM (skip if input analysis folder missing)
    if os.path.isdir("ANALYSIS_NDSM"):
        print("Processing NDSM...")
        category = "ANALYSIS_NDSM_AVE"
        for i in ndsm:
            biopolymer_count = max_cluster_ave(output_path, category, i, args.tmin, args.tmax, args.dt)
            df = pd.DataFrame(biopolymer_count, index=[0])
            df.to_csv("{}/{}/BioNum_{}.csv".format(output_path, category, i), index=False, header=False)
            gen_ave_biopolymer_ni_nj(output_path, biopolymer_count, i, category)
            nAAcids, bio_acids = gen_acid_contacts(biopolymer_count, output_path, category, i)
            np.savetxt("{}/{}/AcidNum_{}.csv".format(output_path, category, i), np.array(nAAcids), delimiter=",")
            gen_acid_ni_nj(output_path, category, i, nAAcids)

        gen_agg_biopolymer_ni_nj(ndsm, output_path)
        gen_agg_acid_ni_nj(ndsm, output_path)
        gen_agg_bond_ni_nj(ndsm, output_path)
        gen_agg_bio_count(ndsm, output_path)
        gen_agg_acid_count(ndsm, output_path)
    else:
        print("Skipping NDSM: ANALYSIS_NDSM not found")


    def generic():
        """Emit whole-system (full-stoichiometry) normalization matrices under ``CM_NORM/MAPS/`` independent of cluster occupancy."""
        biopolymer_num = {
            0: 33,
            1: 16,
            2: 16,
            3: 16,
            4: 16,
            5: 16,
            6: 21
        }

        biopolymer_count = {
            "G3BP1": 33,
            "PABP1": 16,
            "TTP": 16,
            "TIA1": 16,
            "TDP43": 16,
            "FUS": 16,
            "RNA": 21
        }

        df = pd.DataFrame(biopolymer_count, index=[0])
        cm_norm_dir = os.path.join(output_path, "CM_NORM")
        cm_norm_maps = os.path.join(cm_norm_dir, "MAPS")
        os.makedirs(cm_norm_maps, exist_ok=True)
        df.to_csv(os.path.join(cm_norm_maps, "BioNum_SYSTEM.csv"), index=False, header=False)

        gen_ave_biopolymer_ni_nj(cm_norm_dir, biopolymer_num, "SYSTEM", "MAPS")

        nAAcids, bio_acids = gen_acid_contacts(biopolymer_num, cm_norm_dir, "MAPS", sm="SYSTEM")

        np.savetxt(os.path.join(cm_norm_maps, "AcidNum_SYSTEM.csv"), np.array(nAAcids), delimiter=",")

        gen_acid_ni_nj(cm_norm_dir, "MAPS", "SYSTEM", nAAcids)


    generic()
