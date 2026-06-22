"""
Temporal evolution diagnostics (pipeline Step 5).

Builds the time-resolved figures that show how the stress-granule condensate
evolves over the 2 us trajectory for the SG control and the DSM/NDSM small-molecule
sets. It time-averages per-window cluster and radial-density-profile (RDP) data over
three temporal bins (Early 0.0-0.3 us, Middle 0.85-1.15 us, Late 1.7-2.0 us) and
plots:
  - phi(t): fraction of chains in the largest cluster vs time (SG/DSM/NDSM overlay)
  - N_cluster(t): number of droplets vs time
  - RDP(t): protein/RNA (and SM) density profiles at the three temporal bins,
    sigmoid-fit via RDP_TIME.RDP.
DSM/NDSM blocks are skipped automatically when their ANALYSIS_* folders are absent
(SG-only mode).

Pipeline role:
    Runs after max_cluster.py / system_analysis.py; consumes their per-window
    Cluster_* and DensityProfile_* CSVs.

Key inputs (read from the temperature root used as CWD):
    - ANALYSIS_{SG,DSM,NDSM}/Cluster_{biopolymer}_{sm}_{t}.csv
    - ANALYSIS_{SG,DSM,NDSM}/DensityProfile_{biopolymer}_{sm}_{t}.csv
    - {analysis_root}/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv, Cluster_Protein_sg_X.csv
    - CLI flags: --path, --folder, --T, --dt, --tmin, --tmax,
      --yticks1, --yticks2, [--plot-only]

Key outputs (under {analysis_root}/FIGURES/TIME/):
    - TIME_PHI.png, TIME_CLUSTER.png
    - TIME_SG_RDP.png, TIME_DSM_RDP.png, TIME_NDSM_RDP.png

CLI:
    python time_analysis.py --path TEMP_300 --folder CLASSIFY --T 300 \
        --tmin 50 --dt 50 --tmax 2000
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import sem
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute/support modules that live in analysis/ (RDP_TIME).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))
from rdp_time import rdp


# Fixed-axes geometry for consistent panel sizes (mirrors kmeans.py).
#   non-RDP DSM/NDSM curves (phi, N_cluster): 1.6 x 1.6 in axes (matches violins)
#   RDP overlay curves: 2.4 x 2.4 in axes
_CURVE_FIG_SIZE = (2.40, 2.40)
_CURVE_AX_RECT = [0.40 / 2.40, 0.30 / 2.40, 1.60 / 2.40, 1.60 / 2.40]   # ax 1.6 x 1.6 in
_TIME_RDP_FIG_SIZE = (2.50, 2.50)
_TIME_RDP_AX_RECT = [0.50 / 2.50, 0.50 / 2.50, 1.50 / 2.50, 1.50 / 2.50]  # ax 1.5 x 1.5 in


def _make_fixed_axes(figsize, rect):
    """Create a figure with a single fixed-size axes placed at *rect* (axes fraction)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    return fig, ax


def _time_labels(t):
    """Return candidate filename time tokens for *t* (e.g. 50 -> ['50', '50.0']), de-duplicated."""
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


def _resolve_time_file(path_builder, t):
    """Return the first existing path produced by *path_builder* over the time tokens of *t*, else None."""
    for label in _time_labels(t):
        path = path_builder(label)
        if os.path.isfile(path):
            return path
    return None


class average:
    """Per-category (SG/DSM/NDSM) time averaging of RDP and cluster CSVs for a single small molecule.

    Holds the temporal window (tmin/tmax/dt) and provides rdp_ave/cluster_ave to
    average per-frame density profiles and cluster metrics across that window.

    Note: retained for parity with the live copy in ``analysis/average_simulations.py``;
    this rendering script does its averaging via the module-level helpers in
    ``__main__`` and never instantiates this class.
    """

    def __init__(self, category, tmin, tmax, dt):
        self.df = pd.DataFrame()
        self.category = category
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        """Window-average one species' density profiles; write the per-SM averaged CSV."""
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        distance_col = None
        for i in range(self.tmin,self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(category,biopolymer, sm, str(i)))
                continue
            try:
                cur_df = pd.read_csv(input_path)
                distance_col = cur_df["Distance from center of mass (A)"]
                if biopolymer == "SM":
                    density_col = list(cur_df["SM density (mg/mL)"])
                else:
                    density_col = list(cur_df["Protein density (mg/mL)"])
                df_temp[str(i)]=density_col
            except Exception:
                print("{}/DensityProfile_{}_{}_{}.csv unreadable".format(category,biopolymer, sm, str(i)))
                continue

        if distance_col is None or df_temp.empty:
            print("No DensityProfile data found for {} {} in {}".format(biopolymer, sm, category))
            return pd.DataFrame()

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def cluster_ave(self, biopolymer, sm):
        """Per-window mean of cluster metrics; return a DataFrame with a Phi column."""
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        df_list = []
        index_list = []
        first_path = _resolve_time_file(
            lambda label: "{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, label),
            self.tmin,
        )
        if first_path is None:
            print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(self.tmin)))
            return pd.DataFrame()
        df_cols = pd.read_csv(first_path).columns.tolist()
        for i in range(self.tmin,self.tmax,self.dt):
            input_path = _resolve_time_file(
                lambda label: "{}/Cluster_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))
                continue
            try:
                mean_values = pd.read_csv(input_path).mean()
                df_list.append(mean_values)
                index_list.append(i)
            except Exception:
                print("Cluster_{}_{}_{}.csv unreadable".format(biopolymer, sm, str(i)))
                continue

        if not df_list:
            print("No Cluster data found for {} {} in {}".format(biopolymer, sm, category))
            return pd.DataFrame()

        df = pd.DataFrame(columns=df_cols,data=df_list, index=index_list)
        df['Timestep'] = np.array(df['Timestep'])
        df['Phi'] = (np.array(df['Chains in Largest Droplet'])/np.array(df['Total Chain Number']))
        return df


class aggregate:
    """Aggregate already-averaged per-SM RDP and cluster CSVs into a category-level mean + SEM.

    Reads the per-small-molecule averaged files under ANALYSIS_{category}_AVE and
    writes the category aggregate (mean density profile, mean cluster metrics with
    SEM columns) back to the same directory.

    Note: retained for parity with ``analysis/average_simulations.py``; not
    instantiated by this rendering script.
    """

    def __init__(self, path, category):
        self.df = pd.DataFrame()
        self.path=path
        self.category = category

    def rdp_ave(self, biopolymer, sm_list):
        """Mean (and SEM) the per-SM averaged density profiles into one category file."""
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            distance_col = (pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))["SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "Protein density (mg/mL)"])
            df_temp[str(sm)]=density_col

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/ANALYSIS_{}_AVE/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False)

    def cluster_ave(self, biopolymer, sm_list):
        """Mean the per-SM cluster files and write the category aggregate with SEM columns."""
        df_list = []
        index_list = []
        df_cols = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm_list[0])).columns.tolist()
        for sm in sm_list:
            mean_values = pd.read_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, sm)).mean()
            df_list.append(mean_values)
            index_list.append(sm)

        df = pd.DataFrame(columns=df_cols, data=df_list, index=index_list)
        rg_sem = df["Largest Droplet Radius of Gyration"].sem()
        nd_sem = df["Number of Droplets"].sem()
        chains_largest_sem = df["Chains in Largest Droplet"].sem()
        mass_largest_sem = df["Mass of Largest Droplet (mg)"].sem()
        number_external_sem = df["Number of External Chains"].sem()
        mass_external_sem = df["Mass of External Chains"].sem()

        df_mean = df.mean()
        df = pd.DataFrame(df_mean).transpose()
        df["RG SEM"] = rg_sem
        df["ND SEM"] = nd_sem
        df["Chains Largest SEM"] = chains_largest_sem
        df["Mass Largest SEM"] = mass_largest_sem
        df["NE SEM"] = number_external_sem
        df["ME SEM"] = mass_external_sem
        df.to_csv("{}/ANALYSIS_{}_AVE/Cluster_{}_{}.csv".format(self.path, self.category, biopolymer, self.category), index=False, header=True)



class average_biopolymers():
    """Time-average and aggregate per-biopolymer density profiles across a small-molecule set.

    Like ``average`` but iterates over the full biopolymer/RNA-component species
    list (RNA, ADENINE, UCG, Protein, and the six proteins). rdp_ave averages one
    species over the temporal window; rdp_sm_ave/gen_ave/gen_agg build per-SM and
    category-aggregate density-profile CSVs under {path}/{category}.

    Note: retained for parity with ``analysis/average_simulations.py``; not
    instantiated by this rendering script.
    """

    def __init__(self, path, category, tmin, tmax, dt):
        self.path = path
        self.category=category
        self.df = pd.DataFrame()
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt

    def rdp_ave(self, biopolymer, sm):
        """Window-average one species' density profiles for *sm*; write the per-SM averaged CSV."""
        category = sm.split("_")[0].upper()
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        distance_col = None
        for i in range(self.tmin, self.tmax, self.dt):
            input_path = _resolve_time_file(
                lambda label: "ANALYSIS_{}/DensityProfile_{}_{}_{}.csv".format(category, biopolymer, sm, label),
                i,
            )
            if input_path is None:
                print("ANALYSIS_{}/DensityProfile_{}_{}_{}.csv File Missing".format(category, biopolymer, sm, str(i)))
                continue
            cur_df = pd.read_csv(input_path)
            distance_col = cur_df["Distance from center of mass (A)"]
            if biopolymer == "SM":
                density_col = list(cur_df["SM density (mg/mL)"])
            else:
                density_col = list(cur_df["Protein density (mg/mL)"])
            df_temp[str(i)] = density_col

        if distance_col is None or df_temp.empty:
            print("No DensityProfile data found for {} {} in ANALYSIS_{}".format(biopolymer, sm, category))
            return

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)

        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        if sm == "sg_X":
            sm = "SG"
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm), index=False)

    def rdp_sm_ave(self, biopolymer, sm_list, sm_type):
        """Mean the per-SM averaged profiles into one *sm_type* (e.g. DSM/NDSM) category CSV."""
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        for sm in sm_list:
            distance_col = (pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                "Distance from center of mass (A)"])
            if biopolymer == "SM":
                density_col = list(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "SM density (mg/mL)"])
            else:
                density_col = list(
                    pd.read_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm))[
                        "Protein density (mg/mL)"])
            df_temp[str(sm)] = density_col
        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        df.to_csv("{}/{}/Density_Profile_{}_{}.csv".format(self.path, self.category, biopolymer, sm_type), index=False)

    def gen_ave(self, sm_list, sm_type):
        """Run rdp_ave over every biopolymer/RNA-component species for each SM in *sm_list*."""
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for sm in tqdm(sm_list):
            for biopolymer in biopolymer_list:
                self.rdp_ave(biopolymer, sm)

    def gen_agg(self, sm_list, sm_type):
        """Aggregate the per-SM profiles into the *sm_type* category (skipped for SG)."""
        biopolymer_list = ["RNA","ADENINE","UCG","Protein","G3BP1","TDP43","TTP","TIA1","FUS","PABP1"]
        for biopolymer in biopolymer_list:
            if sm_type != "SG":
                self.rdp_sm_ave(biopolymer,sm_list,sm_type)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Time-resolved RDP and cluster analysis (SYSTEM_ANALYSIS-style CLI)')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', required=True, help='Output folder prefix (e.g., CLASSIFY)')
    parser.add_argument('--T', type=int, required=True, help='Simulation temperature in Kelvin')
    parser.add_argument('--dt', type=int, required=True, help='Cluster time stride (ns)')
    parser.add_argument('--tmin', type=int, required=True, help='Start of analysis window (ns)')
    parser.add_argument('--tmax', type=int, required=True, help='End of analysis window (ns)')
    parser.add_argument('--yticks1', default='T', help='Show left y-ticks on RDP (T/F)')
    parser.add_argument('--yticks2', default='T', help='Show right y-ticks on SM axis (T/F)')
    parser.add_argument('--plot-only', action='store_true', help='Accepted for pipeline plot-only mode; this script only regenerates time plots')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    # Normalize to absolute path for consistency
    args.path = os.path.abspath(args.path)

    analysis_root = os.path.join(args.path, f"{args.folder}_{args.T}_{args.dt}_{args.tmin}_{args.tmax}")
    if not os.path.exists(analysis_root):
        print(f"Error: Analysis root {analysis_root} does not exist")
        sys.exit(1)
    data_root = args.path
    os.chdir(data_root)

    def cluster_ave(tmin, tmax, dt, biopolymer, sm):
        """Per-window mean of the Cluster_* CSVs for *sm*; return a DataFrame with a Phi column.

        Reads one Cluster_{biopolymer}_{sm}_{t}.csv per time window in [tmin, tmax),
        means each, and appends Phi (chains in largest droplet / total chains).
        NaN Phi / Number-of-Droplets entries are back-filled with their column mean.
        Returns an empty DataFrame when no window files are found.
        """
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        category_path = os.path.join(data_root, category)
        df_list = []
        index_list = []
        seed_file = _resolve_time_file(
            lambda label: os.path.join(category_path, f"Cluster_{biopolymer}_{sm}_{label}.csv"),
            tmin,
        )
        if seed_file is None:
            print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(tmin)))
            return pd.DataFrame()
        df_cols = pd.read_csv(seed_file).columns.tolist()
        for i in range(tmin,tmax,dt):
            try:
                file_i = _resolve_time_file(
                    lambda label: os.path.join(category_path, f"Cluster_{biopolymer}_{sm}_{label}.csv"),
                    i,
                )
                if file_i is None:
                    print("Cluster_{}_{}_{}.csv File Missing".format(biopolymer, sm, str(i)))
                    continue
                mean_values = pd.read_csv(file_i).mean()
                df_list.append(mean_values)
                index_list.append(i)
            except Exception:
                print("Cluster_{}_{}_{}.csv unreadable".format(biopolymer, sm, str(i)))
                continue

        if not df_list:
            print("No Cluster data found for {} {} in {}".format(biopolymer, sm, category))
            return pd.DataFrame()

        df = pd.DataFrame(columns=df_cols,data=df_list, index=index_list)
        df['Timestep'] = np.array(df['Timestep'])
        df['Phi'] = (np.array(df['Chains in Largest Droplet'])/np.array(df['Total Chain Number']))
        df['Phi'].fillna(df['Phi'].mean(), inplace=True)
        df['Number of Droplets'].fillna(df['Number of Droplets'].mean(), inplace=True)
        return df

    def plot_cluster(timestep, mean_sg, mean_dsm, sem_dsm, mean_ndsm, sem_ndsm):
        """Overlay the SG/DSM/NDSM time traces (phi or N_cluster) with SEM bands.

        Returns the (fig, ax1) so the caller can set axis limits/ticks/legend per
        observable before saving. SG is drawn without a band; DSM/NDSM get shaded
        +/- SEM fills.
        """
        col_pal_sg = sns.color_palette(["#808080"], 1)[0]
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = _make_fixed_axes(_CURVE_FIG_SIZE, _CURVE_AX_RECT)

        ax1.plot(timestep, mean_sg, color=col_pal_sg, label='CONTROL', linewidth=4, zorder=2)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_sg, color=col_pal_sg, legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)

        ax1.plot(timestep, mean_dsm, color=col_pal_sm[0], label='DSM', linewidth=4, zorder=2)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_dsm, color=col_pal_sm[0], legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.fill_between(timestep, mean_dsm - sem_dsm, mean_dsm + sem_dsm, color=col_pal_sm[0], alpha=0.2, zorder=1, clip_on=False)

        ax1.plot(timestep, mean_ndsm, color=col_pal_sm[1], label='NDSM', linewidth=4, zorder=2)
        sns.scatterplot(ax=ax1, x=timestep, y=mean_ndsm, color=col_pal_sm[1], legend=False,
                        s=40,
                        edgecolor="k", linewidth=1, zorder=3, clip_on=False)
        ax1.fill_between(timestep, mean_ndsm - sem_ndsm, mean_ndsm + sem_ndsm, color=col_pal_sm[1], alpha=0.2, zorder=1, clip_on=False)

        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        return fig, ax1

    def rdp_ave(tmin, tmax, dt, biopolymer, sm):
        """Window-average one species' density profiles over [tmin, tmax); return a DataFrame.

        Means the DensityProfile_{biopolymer}_{sm}_{t}.csv files across the window
        into columns (distance, mean density, SEM). Returns an empty DataFrame when
        no profiles are found; used to build the three temporal-bin RDP overlays.
        """
        df_temp = pd.DataFrame()
        df = pd.DataFrame()
        category = "ANALYSIS_{}".format(sm.split("_")[0].upper())
        category_path = os.path.join(data_root, category)
        distance_col = None
        for i in range(tmin,tmax,dt):
            try:
                profile_file = _resolve_time_file(
                    lambda label: os.path.join(category_path, f"DensityProfile_{biopolymer}_{sm}_{label}.csv"),
                    i,
                )
                if profile_file is None:
                    print("{}/DensityProfile_{}_{}_{}.csv File Missing".format(category,biopolymer, sm, str(i)))
                    continue
                cur_df = pd.read_csv(profile_file)
                distance_col = cur_df["Distance from center of mass (A)"]
                if biopolymer == "SM":
                    density_col = list(cur_df["SM density (mg/mL)"])
                else:
                    density_col = list(cur_df["Protein density (mg/mL)"])
                df_temp[str(i)]=density_col
            except Exception:
                print("{}/DensityProfile_{}_{}_{}.csv unreadable".format(category,biopolymer, sm, str(i)))
                continue

        if distance_col is None or df_temp.empty:
            print("No DensityProfile data found for {} {} in {}".format(biopolymer, sm, category))
            return pd.DataFrame()

        row_avg = df_temp.mean(axis=1)
        row_sem = df_temp.sem(axis=1)
        df["Distance from center of mass (A)"] = distance_col
        if biopolymer == "SM":
            df["SM density (mg/mL)"] = row_avg
        else:
            df["Protein density (mg/mL)"] = row_avg
        df["Standard mean error"] = row_sem
        return df

    def plot_rdp(df_list, time_list, sm, base_path, T, show_ax1_yticks=True, show_ax2_yticks=True, plot_sm=True):
        """Overlay sigmoid-fit RDPs for the three temporal bins onto one axes.

        For each temporal-bin dict in *df_list*, sigmoid-fits the SG/Protein/RNA
        (and, for DSM/NDSM, the SM) profiles via RDP_TIME.rdp and draws the fit
        curve plus scatter/error bars, distinguishing bins by line style and
        marker. The SM curve goes on a twin right-hand axis. Data are clipped to
        r <= 400 A. Returns (fig, ax1, ax2); ax2 is None when no SM axis is drawn.
        """
        pca_file_A = f"{base_path}/ANALYSIS_SG_AVE/PCA_Protein_sg_X.csv"
        cluster_file_A = f"{base_path}/ANALYSIS_SG_AVE/Cluster_Protein_sg_X.csv"
        init = [80, 80, 200, 80]
        linestyle_list = ['-', '--', ':']
        marker_list = ['o', 'D', 'v']

        col_pal_sg = sns.color_palette(["#808080"], 1)[0]
        col_pal_protein = sns.color_palette(["#C7383A"], 1)[0]
        col_pal_rna = sns.color_palette(["#0066ff"], 1)[0]
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = _make_fixed_axes(_TIME_RDP_FIG_SIZE, _TIME_RDP_AX_RECT)

        for i, df_dict in enumerate(df_list):
            label_sg = f"SG {time_list[i]}"
            fitterSG = rdp(df_dict["SG"], pca_file_A, cluster_file_A, T, init, label_sg)

            label_protein = f"Protein {time_list[i]}"
            fitterProtein = rdp(df_dict["Protein"], pca_file_A, cluster_file_A, T, init, label_protein)

            label_rna = f"RNA {time_list[i]}"
            fitterRNA = rdp(df_dict["RNA"], pca_file_A, cluster_file_A, T, init, label_rna)

            if sm == "DSM" or sm == "NDSM":
                label_sm = f"{sm} {time_list[i]}"
                fitterSM = rdp(df_dict["SM"], pca_file_A, cluster_file_A, T, init, label_sm)

            else:
                fitterSM = None

            # Clip data to xlim=400
            xlim_max = 400

            # SG
            sg_mask_fit = np.array(fitterSG.fit_x) <= xlim_max
            sg_mask_data = np.array(fitterSG.distances) <= xlim_max
            sg_scatter_indices = [idx for idx in range(min(24, len(fitterSG.distances))) if fitterSG.distances[idx] <= xlim_max]
            
            ax1.plot(np.array(fitterSG.fit_x)[sg_mask_fit], np.array(fitterSG.fit_rho)[sg_mask_fit], 
                    color=col_pal_sg, linewidth=4, zorder=1, linestyle=linestyle_list[i], clip_on=True)
            if sg_scatter_indices:
                sns.scatterplot(ax=ax1, x=np.array(fitterSG.distances)[sg_scatter_indices], 
                              y=np.array(fitterSG.densities)[sg_scatter_indices], color=col_pal_sg, legend=False, s=40,
                              edgecolor="k", linewidth=1, zorder=3, clip_on=True, marker=marker_list[i])
            ax1.errorbar(x=np.array(fitterSG.distances)[sg_mask_data], y=np.array(fitterSG.densities)[sg_mask_data], 
                        yerr=np.array(fitterSG.errors)[sg_mask_data], fmt=".", color=col_pal_sg, zorder=2, clip_on=True)

            # Protein
            prot_mask_fit = np.array(fitterProtein.fit_x) <= xlim_max
            prot_mask_data = np.array(fitterProtein.distances) <= xlim_max
            prot_scatter_indices = [idx for idx in range(min(24, len(fitterProtein.distances))) if fitterProtein.distances[idx] <= xlim_max]
            
            ax1.plot(np.array(fitterProtein.fit_x)[prot_mask_fit], np.array(fitterProtein.fit_rho)[prot_mask_fit], 
                    color=col_pal_protein, linewidth=4, zorder=1, linestyle=linestyle_list[i], clip_on=True)
            if prot_scatter_indices:
                sns.scatterplot(ax=ax1, x=np.array(fitterProtein.distances)[prot_scatter_indices], 
                              y=np.array(fitterProtein.densities)[prot_scatter_indices], color=col_pal_protein,
                              legend=False, s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=True, marker=marker_list[i])
            ax1.errorbar(np.array(fitterProtein.distances)[prot_mask_data], np.array(fitterProtein.densities)[prot_mask_data], 
                        yerr=np.array(fitterProtein.errors)[prot_mask_data], fmt=".", color=col_pal_protein, zorder=2, clip_on=True)

            # RNA
            rna_mask_fit = np.array(fitterRNA.fit_x) <= xlim_max
            rna_mask_data = np.array(fitterRNA.distances) <= xlim_max
            rna_scatter_indices = [idx for idx in range(min(24, len(fitterRNA.distances))) if fitterRNA.distances[idx] <= xlim_max]
            
            ax1.plot(np.array(fitterRNA.fit_x)[rna_mask_fit], np.array(fitterRNA.fit_rho)[rna_mask_fit], 
                    color=col_pal_rna, linewidth=4, zorder=2, linestyle=linestyle_list[i], clip_on=True)
            if rna_scatter_indices:
                sns.scatterplot(ax=ax1, x=np.array(fitterRNA.distances)[rna_scatter_indices], 
                              y=np.array(fitterRNA.densities)[rna_scatter_indices], color=col_pal_rna, legend=False,
                              s=40, edgecolor="k", linewidth=1, zorder=3, clip_on=True, marker=marker_list[i])
            ax1.errorbar(np.array(fitterRNA.distances)[rna_mask_data], np.array(fitterRNA.densities)[rna_mask_data], 
                        yerr=np.array(fitterRNA.errors)[rna_mask_data], fmt=".", color=col_pal_rna, zorder=2, clip_on=True)
            if fitterSM is not None and plot_sm:
                ax2 = ax1.twinx()
                if "ND" not in sm:
                    col = col_pal_sm[0]
                else:
                    col = col_pal_sm[1]

                # SM - clip to xlim
                sm_mask_fit = np.array(fitterSM.fit_x) <= xlim_max
                sm_mask_data = np.array(fitterSM.distances) <= xlim_max
                sm_scatter_indices = [idx for idx in range(min(24, len(fitterSM.distances))) if fitterSM.distances[idx] <= xlim_max]
                
                if sm_scatter_indices:
                    sns.scatterplot(ax=ax2, x=np.array(fitterSM.distances)[sm_scatter_indices], 
                                  y=np.array(fitterSM.densities)[sm_scatter_indices], color=col, legend=False, s=40,
                                  edgecolor="k", linewidth=1, zorder=3, clip_on=True, marker=marker_list[i])
                ax2.errorbar(np.array(fitterSM.distances)[sm_mask_data], np.array(fitterSM.densities)[sm_mask_data], 
                            yerr=np.array(fitterSM.errors)[sm_mask_data], fmt=".", color=col, zorder=2, clip_on=True)
                sns.lineplot(ax=ax2, x=np.array(fitterSM.fit_x)[sm_mask_fit], y=np.array(fitterSM.fit_rho)[sm_mask_fit], 
                            color=col, linewidth=4, zorder=1, linestyle=linestyle_list[i])

                # Remove legend if it exists
                legend = ax2.get_legend()
                if legend is not None:
                    legend.remove()
                ax2.tick_params(left=False, right=True, top=False, bottom=False, labelbottom=False, 
                                labelleft=False, labelright=show_ax2_yticks, direction='in',
                                length=4, width=2)
                ax2.set_ylim(0.0, 0.4)

                ax1.tick_params(left=True, right=False, top=True, bottom=False, labelbottom=True, 
                                labelleft=show_ax1_yticks, direction='in',
                                length=4, width=2)
                ax2.spines['bottom'].set_visible(False)
                ax2.spines['left'].set_visible(False)

            else:
                ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, 
                                labelleft=show_ax1_yticks, direction='in',
                                length=4, width=2)
                ax2=None

        return fig, ax1, ax2


    # Output directory is analysis_root, inputs read from current dir (args.path)
    output_path = analysis_root
    dt = int(args.dt)
    start = int(args.tmin)
    end = int(args.tmax)
    
    # Optional arguments for RDP y-tick labels (default: True, True)
    show_ax1_yticks = (str(args.yticks1).upper() == 'T')
    show_ax2_yticks = (str(args.yticks2).upper() == 'T')

    df_sg = cluster_ave(start, end, dt, "SG", "sg_X")
    timestep_sg = np.array(df_sg['Timestep'])
    phi_sg = np.array(df_sg['Phi'])
    n_cluster = np.array(df_sg['Number of Droplets'])

    # Detect SG-only mode by checking for ANALYSIS_DSM / ANALYSIS_NDSM folders.
    # When missing, the per-compound DSM/NDSM aggregates below would crash on
    # empty DataFrames; instead, skip them and only plot SG cluster traces.
    has_dsm = os.path.isdir(os.path.join(data_root, "ANALYSIS_DSM"))
    has_ndsm = os.path.isdir(os.path.join(data_root, "ANALYSIS_NDSM"))

    dsm_list = ["dsm_anisomycin", "dsm_daunorubicin", "dsm_dihydrolipoic_acid", "dsm_hydroxyquinoline", "dsm_lipoamide",
                "dsm_lipoic_acid", "dsm_mitoxantrone", "dsm_pararosaniline", "dsm_pyrivinium", "dsm_quinicrine"]
    ndsm_list = ["ndsm_dmso", "ndsm_valeric_acid", "ndsm_ethylenediamine", "ndsm_propanedithiol",
                 "ndsm_hexanediol", "ndsm_diethylaminopentane", "ndsm_aminoacridine", "ndsm_anthraquinone",
                 "ndsm_acetylenapthacene", "ndsm_anacardic"]

    phi_mean_dsm = phi_sem_dsm = n_cluster_mean_dsm = n_cluster_sem_dsm = None
    phi_mean_ndsm = phi_sem_ndsm = n_cluster_mean_ndsm = n_cluster_sem_ndsm = None

    if has_dsm:
        phi_list = []
        n_cluster_list = []
        for dsm in dsm_list:
            df = cluster_ave(start, end, dt, "SG", dsm)
            if df.empty or 'Phi' not in df.columns:
                continue
            phi_list.append(df['Phi'])
            n_cluster_list.append(np.array(df['Number of Droplets']))
        if phi_list:
            phi_arr = np.array(phi_list)
            phi_mean_dsm = np.mean(phi_arr, axis=0)
            phi_sem_dsm = sem(phi_arr, axis=0)
            n_cluster_arr = np.array(n_cluster_list)
            n_cluster_mean_dsm = np.mean(n_cluster_arr, axis=0)
            n_cluster_sem_dsm = sem(n_cluster_arr, axis=0)
    else:
        print("[INFO] TIME_ANALYSIS: ANALYSIS_DSM missing — skipping DSM time aggregation")

    if has_ndsm:
        phi_list = []
        n_cluster_list = []
        for ndsm in ndsm_list:
            df = cluster_ave(start, end, dt, "SG", ndsm)
            if df.empty or 'Phi' not in df.columns:
                continue
            phi_list.append(df['Phi'])
            n_cluster_list.append(np.array(df['Number of Droplets']))
        if phi_list:
            phi_arr = np.array(phi_list)
            phi_mean_ndsm = np.mean(phi_arr, axis=0)
            phi_sem_ndsm = sem(phi_arr, axis=0)
            n_cluster_arr = np.array(n_cluster_list)
            n_cluster_mean_ndsm = np.mean(n_cluster_arr, axis=0)
            n_cluster_sem_ndsm = sem(n_cluster_arr, axis=0)
    else:
        print("[INFO] TIME_ANALYSIS: ANALYSIS_NDSM missing — skipping NDSM time aggregation")

    # Ensure output directory
    os.makedirs(f"{output_path}/FIGURES/TIME", exist_ok=True)

    if phi_mean_dsm is not None and phi_mean_ndsm is not None:
        fig, ax1 = plot_cluster(timestep_sg, phi_sg, phi_mean_dsm, phi_sem_dsm, phi_mean_ndsm, phi_sem_ndsm)
        ax1.set_xlim(0, 2)
        ax1.set_xticks(np.linspace(0, 2, 5))
        ax1.set_ylim(0.5, 1)
        leg = ax1.legend(loc='lower right', ncol=1, frameon=False, fontsize=8,
                         handlelength=1.0, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
        leg.get_frame().set_alpha(0)
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_PHI.png", format="png", dpi=400)

        fig, ax1 = plot_cluster(timestep_sg, n_cluster, n_cluster_mean_dsm, n_cluster_sem_dsm, n_cluster_mean_ndsm, n_cluster_sem_ndsm)
        ax1.set_xlim(0, 2)
        ax1.set_xticks(np.linspace(0, 2, 5))
        leg = ax1.legend(loc='upper right', ncol=1, frameon=False, fontsize=8,
                         handlelength=1.0, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
        leg.get_frame().set_alpha(0)
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_CLUSTER.png", format="png", dpi=400)
    else:
        # SG-only mode: simple φ(t) and N_cluster(t) for SG control.
        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.plot(timestep_sg, phi_sg, color="#808080", linewidth=2, label="SG")
        ax1.set_xlim(0, 2)
        ax1.set_xticks(np.linspace(0, 2, 5))
        ax1.set_ylim(0.5, 1)
        ax1.set_xlabel("Time (μs)")
        ax1.set_ylabel("φ_bp (fraction in largest cluster)")
        ax1.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_PHI.png", format="png", dpi=400)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.plot(timestep_sg, n_cluster, color="#808080", linewidth=2, label="SG")
        ax1.set_xlim(0, 2)
        ax1.set_xticks(np.linspace(0, 2, 5))
        ax1.set_xlabel("Time (μs)")
        ax1.set_ylabel("Number of Droplets")
        ax1.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_CLUSTER.png", format="png", dpi=400)
        plt.close(fig)


    # SG block
    try:
        df_1_dict = {}
        df_2_dict = {}
        df_3_dict = {}

        df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", "sg_X")
        df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", "sg_X")
        df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", "sg_X")

        df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", "sg_X")
        df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", "sg_X")
        df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", "sg_X")

        df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", "sg_X")
        df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", "sg_X")
        df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", "sg_X")

        df_list = [df_1_dict, df_2_dict, df_3_dict]
        time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]

        fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="sg_X", base_path=analysis_root, T=args.T,
                             show_ax1_yticks=show_ax1_yticks, show_ax2_yticks=show_ax2_yticks)
        ax1.set_xlim(0, 400)
        ax1.set_ylim(0, 800)
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_SG_RDP.png", format="png", dpi=400)
        print("SG time RDP saved")
    except Exception as exc:
        print(f"SG time RDP failed: {exc}")

    # DSM block — skip entirely when no DSM data is present
    if not has_dsm:
        print("[INFO] TIME_ANALYSIS: skipping DSM RDP time block (no ANALYSIS_DSM)")
    try:
      if has_dsm:
        df_1_dict_dsm_list = []
        df_2_dict_dsm_list = []
        df_3_dict_dsm_list = []

        for dsm in dsm_list:
            df_1_dict = {}
            df_2_dict = {}
            df_3_dict = {}

            df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", dsm)
            df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", dsm)
            df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", dsm)
            df_1_dict["SM"] = rdp_ave(0, 250, 50, "SM", dsm)

            df_1_dict_dsm_list.append(df_1_dict)

            df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", dsm)
            df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", dsm)
            df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", dsm)
            df_2_dict["SM"] = rdp_ave(850, 1100, 50, "SM", dsm)
            df_2_dict_dsm_list.append(df_2_dict)

            df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", dsm)
            df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", dsm)
            df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", dsm)
            df_3_dict["SM"] = rdp_ave(1700, 1950, 50, "SM", dsm)
            df_3_dict_dsm_list.append(df_3_dict)

        # Initialize a dictionary to store the averaged DataFrames
        averaged_dfs_1 = {}
        averaged_dfs_2 = {}
        averaged_dfs_3 = {}

        # Extract the keys (assuming all dictionaries have the same keys)
        keys = df_1_dict_dsm_list[0].keys()

        # Iterate over each key
        for key in keys:
            # Concatenate DataFrames corresponding to the current key from all dictionaries
            concatenated_df_1 = pd.concat([d[key] for d in df_1_dict_dsm_list])
            concatenated_df_2 = pd.concat([d[key] for d in df_2_dict_dsm_list])
            concatenated_df_3 = pd.concat([d[key] for d in df_3_dict_dsm_list])

            # Group by the index and calculate the mean
            averaged_df_1 = concatenated_df_1.groupby(concatenated_df_1.index).mean()
            averaged_df_2 = concatenated_df_2.groupby(concatenated_df_2.index).mean()
            averaged_df_3 = concatenated_df_3.groupby(concatenated_df_3.index).mean()

            # Store the averaged DataFrame in the result dictionary
            averaged_dfs_1[key] = averaged_df_1
            averaged_dfs_2[key] = averaged_df_2
            averaged_dfs_3[key] = averaged_df_3

        df_list = [averaged_dfs_1, averaged_dfs_2, averaged_dfs_3]
        time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]

        # DSM (middle panel): drop BOTH y-tick labels (keep tick marks); still plot the
        # SM (dissolving small molecule) second-axis curve. Left labels are carried by
        # the SG panel, right SM-scale labels by the NDSM panel.
        fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="DSM", base_path=analysis_root, T=args.T,
                             show_ax1_yticks=False, show_ax2_yticks=False, plot_sm=True)
        ax1.set_xlim(0, 400)
        ax1.set_ylim(0, 800)
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_DSM_RDP.png", format="png", dpi=400)
        print("DSM time RDP saved")
    except Exception as exc:
        print(f"DSM time RDP failed: {exc}")

    # NDSM block — skip entirely when no NDSM data is present
    if not has_ndsm:
        print("[INFO] TIME_ANALYSIS: skipping NDSM RDP time block (no ANALYSIS_NDSM)")
    try:
      if has_ndsm:
        df_1_dict_ndsm_list = []
        df_2_dict_ndsm_list = []
        df_3_dict_ndsm_list = []

        for ndsm in ndsm_list:
            df_1_dict = {}
            df_2_dict = {}
            df_3_dict = {}

            df_1_dict["SG"] = rdp_ave(0, 250, 50, "SG", ndsm)
            df_1_dict["Protein"] = rdp_ave(0, 250, 50, "Protein", ndsm)
            df_1_dict["RNA"] = rdp_ave(0, 250, 50, "RNA", ndsm)
            df_1_dict["SM"] = rdp_ave(0, 250, 50, "SM", ndsm)

            df_1_dict_ndsm_list.append(df_1_dict)

            df_2_dict["SG"] = rdp_ave(850, 1100, 50, "SG", ndsm)
            df_2_dict["Protein"] = rdp_ave(850, 1100, 50, "Protein", ndsm)
            df_2_dict["RNA"] = rdp_ave(850, 1100, 50, "RNA", ndsm)
            df_2_dict["SM"] = rdp_ave(850, 1100, 50, "SM", ndsm)
            df_2_dict_ndsm_list.append(df_2_dict)

            df_3_dict["SG"] = rdp_ave(1700, 1950, 50, "SG", ndsm)
            df_3_dict["Protein"] = rdp_ave(1700, 1950, 50, "Protein", ndsm)
            df_3_dict["RNA"] = rdp_ave(1700, 1950, 50, "RNA", ndsm)
            df_3_dict["SM"] = rdp_ave(1700, 1950, 50, "SM", ndsm)
            df_3_dict_ndsm_list.append(df_3_dict)

        # Initialize a dictionary to store the averaged DataFrames
        averaged_dfs_1 = {}
        averaged_dfs_2 = {}
        averaged_dfs_3 = {}

        # Extract the keys (assuming all dictionaries have the same keys)
        keys = df_1_dict_ndsm_list[0].keys()

        # Iterate over each key
        for key in keys:
            # Concatenate DataFrames corresponding to the current key from all dictionaries
            concatenated_df_1 = pd.concat([d[key] for d in df_1_dict_ndsm_list])
            concatenated_df_2 = pd.concat([d[key] for d in df_2_dict_ndsm_list])
            concatenated_df_3 = pd.concat([d[key] for d in df_3_dict_ndsm_list])

            # Group by the index and calculate the mean
            averaged_df_1 = concatenated_df_1.groupby(concatenated_df_1.index).mean()
            averaged_df_2 = concatenated_df_2.groupby(concatenated_df_2.index).mean()
            averaged_df_3 = concatenated_df_3.groupby(concatenated_df_3.index).mean()

            # Store the averaged DataFrame in the result dictionary
            averaged_dfs_1[key] = averaged_df_1
            averaged_dfs_2[key] = averaged_df_2
            averaged_dfs_3[key] = averaged_df_3

        df_list = [averaged_dfs_1, averaged_dfs_2, averaged_dfs_3]
        time_list = ["0.0-0.3 $mu$s", "0.85-1.15$mu$s", "1.7-2.0$mu$s"]
        # NDSM (right panel): drop LEFT y-tick labels (keep tick marks); keep right
        # SM-scale labels.
        fig, ax1, ax2 = plot_rdp(df_list, time_list, sm="NDSM", base_path=analysis_root, T=args.T,
                             show_ax1_yticks=False, show_ax2_yticks=show_ax2_yticks)
        ax1.set_xlim(0, 400)
        ax1.set_ylim(0, 800)
        plt.savefig(f"{output_path}/FIGURES/TIME/TIME_NDSM_RDP.png", format="png", dpi=400)
        print("NDSM time RDP saved")
    except Exception as exc:
        print(f"NDSM time RDP failed: {exc}")
