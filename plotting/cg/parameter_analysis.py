"""
Wang-Frenkel homotypic interaction-curve plot for small-molecule parameters.

Support module for the CG parameterization figures (manuscript "MIX_PARAMS"
panel). Reads a fitted small-molecule parameter table (Biomolecule, E, S, V, U,
R) and evaluates the Wang-Frenkel homotypic potential phi(r) for each molecule,
overlaying the curves on one axis. Dissolving small molecules (``Y_`` prefix)
and non-dissolving small molecules (``N_`` prefix) are drawn with distinct
sequential color palettes and legend labels (DSM/NDSM); if no prefixes are
present, all rows are drawn with a single palette and "SM" labels.

Wang-Frenkel form (matches plotting/sm/sm_parameters.py ``_wf_phi``):
    ratio = rc / sig
    alpha = 2*v*ratio^(2mu) * ((1 + 2v) / (2*v*(ratio^(2mu) - 1)))^(2v + 1)
    phi(r) = eps * alpha * ((sig/r)^(2mu) - 1) * ((rc/r)^(2mu) - 1)^(2v)
where (eps, sig, v, mu, rc) = (E, S, V, U, R).

Inputs: ``parameters.csv`` (a sm_parameters.csv with a Biomolecule column) and,
optionally, a molecules CSV (column ``Molecules``) restricting which rows plot.
Output: ``{output_dir}/MIX_PARAMS.png`` (400 dpi).

Entry point: ``run_parameter_analysis()`` (also runnable as a script with the
default arguments via ``main()``); called from cg_pipeline/run_full_pipeline.py.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ["run_parameter_analysis"]


def _configure_style() -> None:
    """Set the shared Matplotlib/seaborn rc parameters for the figure."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("axes", titlesize=10)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=8)
    plt.rc("font", size=10)
    plt.rc("axes", linewidth=2)


def _compute_phi(row: pd.Series, r: np.ndarray) -> np.ndarray:
    """Evaluate the Wang-Frenkel homotypic potential phi(r) for one molecule.

    Args:
        row: parameter row with fields E, S, V, U, R = (eps, sig, v, mu, rc).
        r: radial separations (Angstrom) to evaluate at.

    Returns:
        phi(r) array (same shape as ``r``); zeros where the parameters are
        degenerate (denominator ~ 0) and NaN/inf values are sanitized to 0.
    """
    eps, sig, v, mu, rc = row["E"], row["S"], row["V"], row["U"], row["R"]
    ratio = rc / sig
    alpha = 2 * v * ratio ** (2 * mu)
    denom = 2 * v * (ratio ** (2 * mu) - 1)
    if abs(denom) < 1e-12:
        return np.zeros_like(r)
    alpha *= ((1 + 2 * v) / denom) ** (2 * v + 1)
    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(np.power(rc / r, 2 * mu) - 1, 2 * v)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    return phi


def _plot_curves(df: pd.DataFrame, out_path: Path, interaction_limit: int, label_map: dict[str, str] | None = None) -> None:
    """Overlay Wang-Frenkel phi(r) curves for all molecules and save the figure.

    DSM (``Y_``) and NDSM (``N_``) rows get separate sequential palettes; if no
    prefixed rows exist, a single palette is used. ``interaction_limit`` (when
    > 0) caps how many curves of each group are drawn. ``label_map`` overrides
    the legend label for each index name. No-op if ``df`` is empty.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5), tight_layout=True)
    r = np.linspace(4, 22, 1000)

    rows = list(df.iterrows())
    dissolving = [(name, row) for name, row in rows if name.startswith('Y_')]
    nondissolving = [(name, row) for name, row in rows if name.startswith('N_')]

    plotted_entries: list[tuple[str, pd.Series, tuple[float, float, float]]] = []

    if dissolving or nondissolving:
        if interaction_limit and interaction_limit > 0:
            dissolving = list(islice(dissolving, interaction_limit))
            nondissolving = list(islice(nondissolving, interaction_limit))

        if nondissolving:
            palette_ndense = sns.dark_palette("indigo", n_colors=len(nondissolving))
            plotted_entries.extend((name, row, palette_ndense[idx]) for idx, (name, row) in enumerate(nondissolving))
        if dissolving:
            palette_dense = sns.light_palette("darkgreen", n_colors=len(dissolving))
            plotted_entries.extend((name, row, palette_dense[idx]) for idx, (name, row) in enumerate(dissolving))
    else:
        if not rows:
            return
        palette_single = sns.color_palette("rocket", n_colors=len(rows))
        plotted_entries.extend((name, row, palette_single[idx]) for idx, (name, row) in enumerate(rows))

    for name, row, color in plotted_entries:
        phi = _compute_phi(row, r)
        disp = label_map.get(name, name) if label_map else name
        ax.plot(r, phi, label=disp, color=color, linewidth=1.5)

    ax.tick_params(direction='in', length=4, width=2,
                   top=True, bottom=True, left=True, right=True,
                   labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    ax.set_xlim(4, 16)
    ax.set_ylim(-0.4, 1.0)
    # Legend inside top-right, small font, 2 columns
    ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    fig.savefig(out_path, format="png", dpi=400)
    plt.close(fig)


def _select_molecules(molecules_csv: Path | str | None) -> set[str] | None:
    """Read the optional molecules CSV and return the allowed Biomolecule set.

    Returns None (no filtering) if the path is None, missing, or lacks a
    ``Molecules`` column. Prefers ``N_``/``Y_``-prefixed names when present.
    """
    if molecules_csv is None:
        return None
    molecules_path = Path(molecules_csv)
    if not molecules_path.exists():
        return None
    df_mols = pd.read_csv(molecules_path)
    if 'Molecules' not in df_mols.columns:
        return None
    names = df_mols['Molecules'].astype(str)
    prefixed = names[names.str.startswith(('N_', 'Y_'))]
    if not prefixed.empty:
        return set(prefixed)
    return set(names)


def run_parameter_analysis(parameters_csv: Path | str = 'parameters.csv', output_dir: Path | str = '.', interaction_plot_num: int = 0, molecules_csv: Path | str | None = None) -> None:
    """Read a parameter table and write the MIX_PARAMS Wang-Frenkel curve plot.

    Args:
        parameters_csv: CSV with a Biomolecule column plus E, S, V, U, R.
        output_dir: directory for ``MIX_PARAMS.png`` (created if needed).
        interaction_plot_num: cap on curves drawn per DSM/NDSM group (0 = all).
        molecules_csv: optional CSV (column ``Molecules``) restricting which
            molecules are plotted.

    Prefers small-molecule rows (``N_``/``Y_`` prefixes) when present and
    assigns sequential DSM/NDSM/SM legend labels. No-op if the table lacks a
    Biomolecule column.
    """
    _configure_style()
    parameters_csv = Path(parameters_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_params = pd.read_csv(parameters_csv)
    if 'Biomolecule' not in df_params.columns:
        return
    df_ordered = df_params.drop_duplicates('Biomolecule', keep='first')
    df = df_ordered.set_index('Biomolecule')[['E', 'S', 'V', 'U', 'R']]

    allowed = _select_molecules(molecules_csv)
    if allowed is not None:
        df = df[df.index.isin(allowed)]

    df_sm = df[df.index.str.startswith('N_') | df.index.str.startswith('Y_')]
    df_to_plot = df_sm if not df_sm.empty else df

    # Legend labels sequential to ensure numbering starts at 1
    label_map: dict[str, str] = {}
    for i, name in enumerate(df_to_plot.index, start=1):
        prefix = 'DSM' if str(name).startswith('Y_') else ('NDSM' if str(name).startswith('N_') else 'SM')
        label_map[name] = f"{prefix} {i}"

    _plot_curves(df_to_plot, output_dir / 'MIX_PARAMS.png', interaction_plot_num, label_map)


def main() -> None:
    """Run the parameter analysis with default arguments."""
    run_parameter_analysis()


if __name__ == '__main__':
    main()
