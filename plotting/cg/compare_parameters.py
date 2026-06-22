"""
Parity plots comparing two coarse-grained parameterizations (JJ vs JK).

Support module for the CG parameterization figures (manuscript Fig S1). Reads
two parameter tables for the same set of Wang-Frenkel interaction parameters
(E, S, V, U, R) produced by independent fits ("JJ" = reference, "JK" = new),
merges them on a shared key, and renders one square JJ-vs-JK scatter per
parameter against the y = x identity line. Each panel is annotated with the
percent RMSE and Pearson correlation of the new fit relative to the reference.

Two modes:
    mix : heterotypic pair-coefficient tables (whitespace-delimited LAMMPS
          ``pair_coeff`` lines, columns index, ID, style, E, S, V, U, R),
          merged on (index, ID) and grouped by small-molecule ID.
    sm  : homotypic small-molecule parameter CSVs (columns Biomolecule,
          Molecule Number, E, S, V, U, R), merged on Biomolecule.

The JK energy column E is rescaled by 4.184 (kcal -> kJ) so both sources share
units before plotting.

Inputs: two parameter files (reference, new); paths supplied by the caller.
Outputs: ``{out_dir}/{mix|sm}_{E,S,V,U,R}_JJ_vs_JK.png`` (400 dpi) when
``plot_all(save=True)``; otherwise shown interactively.

Not runnable as a script; instantiate ``parity_comparer`` from the pipeline
(see cg_pipeline/run_full_pipeline.py and plotting/sm/sm_parameters.py).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error


class parity_comparer:
    """
    Unified parity plotting for mix parameters (JJ vs JK) and
    homotypic small-molecule parameters (JJ vs JK).

    mode: 'mix' or 'sm'
    - mix expects pair_coeff tables: columns index, ID, E,S,V,U,R
    - sm expects CSVs with columns Biomolecule,E,S,V,U,R
    """
    def __init__(self, mode: str, file_jj: str, file_jk: str):
        """Load and merge the reference (JJ) and new (JK) parameter tables.

        Args:
            mode: ``'mix'`` (heterotypic pair-coeff tables) or ``'sm'``
                (homotypic small-molecule CSVs). Case-insensitive.
            file_jj: Path to the reference parameter file.
            file_jk: Path to the new parameter file (its E column is rescaled
                by 4.184 to match JJ units).

        Raises:
            ValueError: if ``mode`` is not ``'mix'`` or ``'sm'``.
        """
        self.mode = mode.lower().strip()
        if self.mode not in {"mix", "sm"}:
            raise ValueError("mode must be 'mix' or 'sm'")
        if self.mode == 'mix':
            self._load_mix(file_jj, file_jk)
        else:
            self._load_sm(file_jj, file_jk)

    def _load_mix(self, file_jj: str, file_jk: str):
        """Load whitespace-delimited pair-coeff tables and merge on (index, ID).

        Populates ``self.df`` (JJ/JK suffixed columns), ``self.group_key``,
        ``self.legend_labels`` (ID -> "SM <ID>") and ``self.colors``.
        """
        cols = ['pair', 'index', 'ID', 'style', 'E', 'S', 'V', 'U', 'R']
        df_jj = pd.read_csv(file_jj, sep=r"\s+", comment='#', names=cols)
        df_jk = pd.read_csv(file_jk, sep=r"\s+", comment='#', names=cols)
        for c in ['index','ID','E','S','V','U','R']:
            df_jj[c] = pd.to_numeric(df_jj[c], errors='coerce')
            df_jk[c] = pd.to_numeric(df_jk[c], errors='coerce')
        # Unit conversion for E
        df_jk['E'] = df_jk['E'] * 4.184
        self.df = pd.merge(df_jj, df_jk, on=['index','ID'], suffixes=('_JJ','_JK'))
        # Group key
        self.group_key = 'ID'
        # Build legend labels SM 1..N by order of unique IDs
        uniq = sorted(self.df['ID'].dropna().unique().tolist())
        # For mix, use the actual ID numeric for label SM <ID>
        self.legend_labels = {ID: f"SM {int(ID)}" for ID in uniq}
        self.colors = {ID: plt.get_cmap('tab20')(i % 20) for i, ID in enumerate(uniq)}

    def _load_sm(self, file_jj: str, file_jk: str):
        """Load homotypic small-molecule CSVs and merge on Biomolecule.

        Accepts headered CSVs or headerless files (falling back to the fixed
        column order). Populates ``self.df``, ``self.group_key``,
        ``self.legend_labels`` (Biomolecule -> "SM <Molecule Number>") and
        ``self.colors``.
        """
        # Prefer headered CSVs; fall back to header=None if needed
        cols = ['Biomolecule', 'Molecule Number', 'E', 'S', 'V', 'U', 'R']
        try:
            df_jj = pd.read_csv(file_jj)
            if 'Biomolecule' not in df_jj.columns:
                raise ValueError
        except Exception:
            df_jj = pd.read_csv(file_jj, header=None, names=cols)
        try:
            df_jk = pd.read_csv(file_jk)
            if 'Biomolecule' not in df_jk.columns:
                raise ValueError
        except Exception:
            df_jk = pd.read_csv(file_jk, header=None, names=cols)

        # Ensure numeric types
        for c in ['E', 'S', 'V', 'U', 'R', 'Molecule Number']:
            if c in df_jj.columns:
                df_jj[c] = pd.to_numeric(df_jj[c], errors='coerce')
            if c in df_jk.columns and c != 'Molecule Number':
                df_jk[c] = pd.to_numeric(df_jk[c], errors='coerce')

        # Unit conversion for E (match mix convention: scale JK)
        if 'E' in df_jk.columns:
            df_jk['E'] = pd.to_numeric(df_jk['E'], errors='coerce') * 4.184

        self.df = pd.merge(
            df_jj[['Biomolecule','Molecule Number','E','S','V','U','R']],
            df_jk[['Biomolecule','E','S','V','U','R']],
            on='Biomolecule',
            suffixes=('_JJ','_JK')
        )
        self.group_key = 'Biomolecule'
        # For SM mode, use Molecule Number from JJ table
        label_series = self.df['Molecule Number']
        label_series = pd.to_numeric(label_series, errors='coerce')
        label_nums = dict(zip(self.df['Biomolecule'], label_series))
        names = self.df['Biomolecule'].astype(str).tolist()
        self.legend_labels = {}
        for i, name in enumerate(names):
            num = label_nums.get(name)
            try:
                lab = f"SM {int(num)}" if pd.notna(num) else f"SM {i+1}"
            except Exception:
                lab = f"SM {i+1}"
            self.legend_labels[name] = lab
        self.colors = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}

    def _apply_style(self):
        """Set the shared Matplotlib/seaborn rc parameters for the panels."""
        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

    def plot_all(self, save: bool = False, out_dir: str = '.', allowed: set[str] | None = None,
                 figsize=None, ax_rect=None):
        """Render one square JJ-vs-JK parity panel per parameter (E, S, V, U, R).

        Each panel scatters new (JK, x) against reference (JJ, y) values on a
        square axis with the y = x line, annotated with percent RMSE and PCC.

        Args:
            save: if True, write each panel to ``out_dir``; else show it.
            out_dir: directory for the saved PNGs.
            allowed: in ``sm`` mode, restrict to these Biomolecule names.
            figsize: figure size in inches (used with ``ax_rect``); defaults to
                a square panel.
            ax_rect: explicit axes rectangle [l, b, w, h] in figure fractions,
                to match other square panels; if None, a default layout is used.
        """
        params = ['E','S','V','U','R']
        df_plot = self.df.copy()
        if self.mode == 'sm' and allowed:
            df_plot = df_plot[df_plot['Biomolecule'].isin(allowed)]
        if df_plot.empty:
            return

        # Build plotting labels/colors for filtered set
        if self.mode == 'sm':
            items = df_plot['Biomolecule'].astype(str).tolist()
            legend_labels = {name: f"SM {i+1}" for i, name in enumerate(items)}
            color_map = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(items)}
        else:
            items = sorted(df_plot['ID'].dropna().unique().tolist())
            legend_labels = {ID: f"SM {i+1}" for i, ID in enumerate(items)}
            color_map = {ID: plt.get_cmap('tab20')(i % 20) for i, ID in enumerate(items)}

        for p in params:
            x = df_plot[f'{p}_JK'].to_numpy(dtype=float, copy=True)
            y = df_plot[f'{p}_JJ'].to_numpy(dtype=float, copy=True)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            x = x[mask]
            y = y[mask]
            maxval = np.nanmax([x.max(), y.max()])
            if not np.isfinite(maxval):
                continue

            # Extend axis to next integer beyond maxval
            max_tick = math.ceil(maxval + 1)
            max_plot = max_tick * 1.05 if max_tick > 0 else 1
            Xm = x.reshape(-1, 1)
            ym = y
            if len(ym)>1 and np.std(Xm)>0 and np.std(ym)>0:
                try:
                    r2 = r2_score(ym, Xm)
                except Exception:
                    r2 = float('nan')
                try:
                    rmse = math.sqrt(mean_squared_error(ym, Xm))
                except Exception:
                    rmse = float('nan')
                rmse_pct = (rmse/abs(np.nanmean(ym))*100) if np.nanmean(ym)!=0 else np.nan
                try:
                    pcc, _ = pearsonr(Xm.flatten(), ym)
                except Exception:
                    pcc = float('nan')
            else:
                r2 = float('nan')
                rmse = float('nan')
                rmse_pct = float('nan')
                pcc = float('nan')

            self._apply_style()
            if ax_rect is not None:
                # explicit axes geometry (e.g. to match other square panels)
                fig = plt.figure(figsize=figsize or (3.30, 3.30))
                ax = fig.add_axes(ax_rect)
            else:
                fig, ax = plt.subplots(figsize=(2.35, 2.35))
                fig.subplots_adjust(left=0.25, right=0.98, bottom=0.25, top=0.95)
            if self.mode == 'sm':
                for _, row in df_plot.iterrows():
                    name = row['Biomolecule']
                    xv = row[f'{p}_JK']
                    yv = row[f'{p}_JJ']
                    if not (np.isfinite(xv) and np.isfinite(yv)):
                        continue
                    lab = legend_labels.get(name, str(name))
                    ax.scatter(xv, yv, color=color_map.get(name, 'C0'), label=lab, s=10, edgecolor='k', linewidth=0.5)
            else:
                for gid, group in df_plot.groupby('ID'):
                    gx = group[f'{p}_JK'].to_numpy(dtype=float, copy=True)
                    gy = group[f'{p}_JJ'].to_numpy(dtype=float, copy=True)
                    valid = np.isfinite(gx) & np.isfinite(gy)
                    if not valid.any():
                        continue
                    ax.scatter(gx[valid], gy[valid], color=color_map.get(gid, 'C0'), s=10, edgecolor='k', linewidth=0.5)

            ax.plot([0, max_plot], [0, max_plot], '--', color='black')
            ax.set_xlim(0, max_plot)
            ax.set_ylim(0, max_plot)
            ax.set_aspect('equal', 'box')
            if np.isfinite(r2):
                ax.text(0.05, 0.95, f'RMSE={rmse_pct:.1f}%\nPCC={pcc:.2f}', transform=ax.transAxes, va='top')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            # Legend intentionally omitted; user can add manually later
            ax.tick_params(left=True, right=True, top=True, bottom=True,
                           labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                           direction='in', length=4, width=2)
            # Ensure matching tick spacing on both axes (5 ticks, rounded labels)
            try:
                # regenerate ticks using extended maximum
                ticks = np.linspace(0, max_tick, 5)
                tick_labels = [f"{t:.2f}" for t in ticks]
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
                ax.set_yticks(ticks)
                ax.set_yticklabels(tick_labels)
            except Exception:
                pass

            fname_prefix = 'mix' if self.mode == 'mix' else 'sm'
            if save:
                fig.savefig(f"{out_dir}/{fname_prefix}_{p}_JJ_vs_JK.png", dpi=400)
                plt.close(fig)
            else:
                plt.show()
