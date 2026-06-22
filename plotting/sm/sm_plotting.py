"""sm_plotting.py — DSM/NDSM small-molecule isolation (pure-SM control) figures.

Loads the per-window CSV outputs of the pure-SM solubility runs (cluster size,
SM radial density profiles, and SM--SM COM RDF) for the DSM and NDSM compound
cohorts, aggregates them (window-averaged within each compound, then mean/SEM
across compounds), and renders the manuscript-style figures.

Purpose: produce the SM self-aggregation panels (cluster fraction / number vs
time, SM number- and mass-density profiles, COM RDF) that show DSM and NDSM
compounds remain dispersed in the absence of biopolymers. ``sm_isolation.py``
reuses the cluster loader (``_load_cluster``) and time downsampler
(``_downsample_time``) defined here.

Inputs (per category directory, e.g. ANALYSIS_DSM / ANALYSIS_NDSM):
    Cluster_SM_<system>_t<start>.csv               (largest-cluster series)
    Density_Profile_SM_Number_<system>_t<start>.csv (SM number density vs r)
    Density_Profile_SM_Mass_<system>_t<start>.csv   (SM mass density vs r)
    RDF_SM_COM_<system>_t<start>.csv                (SM--SM COM g(r))

Outputs (under ``output_dir``): aggregated/, dsm/, ndsm/ subfolders of PNGs
(cluster_time, cluster_number, cluster_violin, rdp_number, rdp_mass, rdf).

CLI:
    python sm_plotting.py --dsm-dir ANALYSIS_DSM --ndsm-dir ANALYSIS_NDSM \
        --output-dir ANALYSIS_SM_PLOTS [--dt-ns 1.0 --window-ns 20.0 \
        --trim-start 40.0 --trim-end 200.0]
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def _configure_style() -> None:
    """Apply the shared seaborn ticks/white house style and font sizes."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("axes", titlesize=12)
    plt.rc("axes", labelsize=11)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=9)
    plt.rc("font", size=10)


def _ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _parse_system_tag(tag: str) -> tuple[str, str, int]:
    """Split a ``<system>_t<start>`` tag into (category, sm_name, start_ns).

    The category is "DSM"/"NDSM" for ``dsm_``/``ndsm_`` prefixes, else the
    uppercased leading token. Raises ValueError if the ``_t<start>`` suffix
    is absent.
    """
    match = re.match(r"(.*)_t(\d+)$", tag)
    if not match:
        raise ValueError(f"Unable to parse system tag '{tag}'. Expected suffix '_t<start>'.")
    full_name = match.group(1)
    start_ns = int(match.group(2))
    if full_name.startswith("ndsm_"):
        category = "NDSM"
        sm_name = full_name.split("_", 1)[1]
    elif full_name.startswith("dsm_"):
        category = "DSM"
        sm_name = full_name.split("_", 1)[1]
    else:
        category = full_name.split("_", 1)[0].upper()
        sm_name = full_name
    return category, sm_name, start_ns


def _collect_files(base_dir: Path, pattern: str) -> list[Path]:
    """Return the sorted list of files in ``base_dir`` matching ``pattern``."""
    return sorted(base_dir.glob(pattern))


def _load_cluster(category_dir: Path, dt_ns: float, window_ns: float,
                  tmin: float, tmax: float) -> pd.DataFrame:
    """Load and concatenate per-window cluster CSVs from one category directory.

    Builds a per-frame time axis (start_ns + frame * dt_ns), keeps frames in
    [tmin, tmax], and tags each row with category/sm/window_start_ns. Returns a
    long DataFrame (empty with the expected columns if nothing matched).
    """
    rows = []
    for path in _collect_files(category_dir, "Cluster_SM_*.csv"):
        system_tag = path.stem.replace("Cluster_SM_", "")
        category, sm_name, start_ns = _parse_system_tag(system_tag)
        window_end = start_ns + window_ns
        if window_end <= tmin or start_ns >= tmax:
            continue
        df = pd.read_csv(path)
        time_ns = start_ns + np.arange(len(df)) * dt_ns
        mask = (time_ns >= tmin) & (time_ns <= tmax)
        if not np.any(mask):
            continue
        sub = df.loc[mask, ["LargestClusterFraction", "LargestClusterSize", "NumClusters"]].copy()
        sub["time_ns"] = time_ns[mask]
        sub["category"] = category
        sub["sm"] = sm_name
        sub["window_start_ns"] = start_ns
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["LargestClusterFraction", "LargestClusterSize", "NumClusters",
                                     "time_ns", "category", "sm", "window_start_ns"])
    return pd.concat(rows, ignore_index=True)


def _load_profile(category_dir: Path, pattern: str, value_col: str,
                  dt_ns: float, window_ns: float, tmin: float, tmax: float) -> pd.DataFrame:
    """Load and concatenate per-window radial-profile/RDF CSVs for one category.

    Selects windows whose [start, start+window_ns) overlaps [tmin, tmax] and
    tags each row with category/sm/window_start_ns. ``value_col`` documents the
    expected y-column; ``dt_ns`` is unused for these per-radius profiles (kept
    for a uniform loader signature). Returns an empty DataFrame if nothing
    matched.
    """
    rows = []
    for path in _collect_files(category_dir, pattern):
        system_tag = path.stem
        system_tag = system_tag.replace("Density_Profile_SM_Number_", "")
        system_tag = system_tag.replace("RDF_SM_COM_", "")
        category, sm_name, start_ns = _parse_system_tag(system_tag)
        window_end = start_ns + window_ns
        if window_end <= tmin or start_ns >= tmax:
            continue
        df = pd.read_csv(path)
        df["category"] = category
        df["sm"] = sm_name
        df["window_start_ns"] = start_ns
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_rdp_number(category_dir: Path, dt_ns: float, window_ns: float,
                     tmin: float, tmax: float) -> pd.DataFrame:
    """Load the per-window SM number-density radial profiles for one category."""
    return _load_profile(category_dir, "Density_Profile_SM_Number_*.csv",
                         "SM number density (1/A^3)", dt_ns, window_ns, tmin, tmax)


def _load_rdp_mass(category_dir: Path, dt_ns: float, window_ns: float,
                   tmin: float, tmax: float) -> pd.DataFrame:
    """Load the per-window SM mass-density radial profiles for one category."""
    return _load_profile(category_dir, "Density_Profile_SM_Mass_*.csv",
                         "SM mass density (mg/mL)", dt_ns, window_ns, tmin, tmax)


def _aggregate_mean_sem(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Group ``df`` by ``group_cols`` and return mean, SEM, and count of ``value_col``.

    SEM uses ddof=1 std / sqrt(count), and is 0 for singleton groups.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "sem", "count"])
    grouped = df.groupby(group_cols)[value_col]
    mean = grouped.mean().rename("mean")
    count = grouped.count().rename("count")
    std = grouped.std(ddof=1).rename("std")
    agg = pd.concat([mean, count, std], axis=1).reset_index()
    agg["sem"] = agg.apply(lambda row: (row["std"] / np.sqrt(row["count"])) if row["count"] > 1 else 0.0, axis=1)
    return agg.drop(columns=["std"])


def _aggregate_by_sm_then_category(df: pd.DataFrame, category_col: str, sm_col: str,
                                   x_col: str, y_col: str) -> pd.DataFrame:
    """Average windows within each SM first, then compute category mean/SEM across SMs.

    This avoids pseudo-replication from multiple windows per SM.
    """
    if df.empty:
        return pd.DataFrame(columns=[category_col, x_col, "mean", "sem", "count"])
    sm_mean = df.groupby([category_col, sm_col, x_col])[y_col].mean().reset_index()
    grouped = sm_mean.groupby([category_col, x_col])[y_col]
    mean = grouped.mean().rename("mean")
    count = grouped.count().rename("count")
    std = grouped.std(ddof=1).rename("std")
    agg = pd.concat([mean, count, std], axis=1).reset_index()
    agg["sem"] = agg.apply(lambda row: (row["std"] / np.sqrt(row["count"])) if row["count"] > 1 else 0.0, axis=1)
    return agg.drop(columns=["std"])


def _format_sm(sm: str) -> str:
    """Format an SM name for display (underscores to spaces, upper-cased)."""
    return sm.replace("_", " ").upper()


def _apply_axis_tick_style(ax) -> None:
    """Apply inward ticks on all four spines (house style)."""
    ax.tick_params(direction="in", top=True, right=True, bottom=True, left=True,
                   length=4, width=1.5)


class _two_decimal_scalar_formatter(ScalarFormatter):
    """ScalarFormatter that pins the mantissa to a fixed number of decimals."""

    def __init__(self, decimals: int = 2) -> None:
        """Build a math-text scientific formatter using ``decimals`` mantissa digits."""
        super().__init__(useMathText=True)
        self._decimals = decimals

    def _set_format(self, *args) -> None:
        """Override the mantissa format string with the fixed-decimal pattern."""
        super()._set_format(*args)
        self.format = f"%.{self._decimals}f"


def _set_axis_tick_formatter(ax, *, axis: str = "y", decimals: int = 2,
                             scientific: bool = False) -> None:
    """Set fixed-decimal tick labels on one axis (math-text scientific if requested)."""
    axis_obj = ax.yaxis if axis == "y" else ax.xaxis
    if scientific:
        formatter = _two_decimal_scalar_formatter(decimals=decimals)
        formatter.set_powerlimits((0, 0))
        formatter.set_scientific(True)
        axis_obj.set_major_formatter(formatter)
        axis_obj.offsetText.set_visible(True)
    else:
        axis_obj.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def _plot_cluster_time(agg_df: pd.DataFrame, palette: dict[str, str],
                       output_path: Path) -> None:
    """Plot the aggregated largest-cluster fraction vs time for each category.

    ``agg_df`` carries per-category ``mean``/``sem`` columns; renders a heavy
    line plus scatter markers and a SEM band, with fixed x (0--200 ns) and y
    (2e-4--4e-4) ranges.
    """
    if agg_df.empty:
        return
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for category in agg_df["category"].unique():
        data = agg_df[agg_df["category"] == category].sort_values("time_ns")
        color = palette.get(category, "grey")
        y = data["mean"]
        ysem = data["sem"]
        ax.plot(data["time_ns"], y, color=color, linewidth=4, label=category, zorder=2)
        ax.scatter(data["time_ns"], y, color=color, edgecolor="k", linewidth=1,
                   s=40, zorder=3, clip_on=False)
        ax.fill_between(data["time_ns"], y - ysem, y + ysem, color=color, alpha=0.2, zorder=1, clip_on=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    ax.legend(frameon=False, loc="upper right")
    # X axis: 0..200 ns with ticks at 0, 50, 100, 150, 200
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    # Y axis: fixed limits 2e-4..4e-4 with 5 ticks; keep scientific formatter
    ax.set_ylim(2e-4, 4e-4)
    ax.set_yticks(list(np.linspace(2e-4, 4e-4, 5)))
    _set_axis_tick_formatter(ax, decimals=2, scientific=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_cluster_number(agg_df: pd.DataFrame, palette: dict[str, str],
                         output_path: Path) -> None:
    """Plot the aggregated number of clusters vs time for each category.

    Same heavy-line + scatter + SEM-band style as ``_plot_cluster_time`` with a
    fixed y range (8240--8340) and the all-monomeric baseline at 8324 (each of
    the 8324 molecules its own cluster).
    """
    if agg_df.empty:
        return
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for category in agg_df["category"].unique():
        data = agg_df[agg_df["category"] == category].sort_values("time_ns")
        color = palette.get(category, "grey")
        y = data["mean"]
        ysem = data["sem"]
        ax.plot(data["time_ns"], y, color=color, linewidth=4, label=category, zorder=2)
        ax.scatter(data["time_ns"], y, color=color, edgecolor="k", linewidth=1,
                   s=40, zorder=3, clip_on=False)
        ax.fill_between(data["time_ns"], y - ysem, y + ysem, color=color, alpha=0.2, zorder=1, clip_on=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    ax.legend(frameon=False, loc="upper right")
    # X axis: 0..200 ns with ticks at 0, 50, 100, 150, 200
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    # Y axis: fixed limits and ticks
    ax.set_ylim(8240, 8340)
    ax.set_yticks(list(np.linspace(8240, 8340, 5)))
    # Scientific notation style consistent with violin plots
    _set_axis_tick_formatter(ax, decimals=2, scientific=True)
    # Baseline for no clustering: each molecule is its own cluster
    ax.axhline(8324, color="grey", linestyle="--", linewidth=1.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_cluster_time_per_sm(df: pd.DataFrame, category: str,
                              palette: list, output_path: Path) -> None:
    """Plot one line per compound: largest-cluster fraction vs time, one category."""
    data = df[df["category"] == category]
    if data.empty:
        return
    mean_df = _aggregate_mean_sem(data, ["sm", "time_ns"], "LargestClusterFraction")
    order = sorted(mean_df["sm"].unique())
    color_cycle = palette if len(palette) >= len(order) else sns.color_palette("husl", len(order))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for sm, color in zip(order, color_cycle):
        sm_data = mean_df[mean_df["sm"] == sm].sort_values("time_ns")
        ax.plot(sm_data["time_ns"], sm_data["mean"], label=_format_sm(sm), color=color, linewidth=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_cluster_number_per_sm(df: pd.DataFrame, category: str,
                                palette: list, output_path: Path) -> None:
    """Plot one line per compound: number of clusters vs time, one category."""
    data = df[df["category"] == category]
    if data.empty:
        return
    mean_df = _aggregate_mean_sem(data, ["sm", "time_ns"], "NumClusters")
    order = sorted(mean_df["sm"].unique())
    color_cycle = palette if len(palette) >= len(order) else sns.color_palette("husl", len(order))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for sm, color in zip(order, color_cycle):
        sm_data = mean_df[mean_df["sm"] == sm].sort_values("time_ns")
        ax.plot(sm_data["time_ns"], sm_data["mean"], label=_format_sm(sm), color=color, linewidth=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_violin(df: pd.DataFrame, output_path: Path, title: str,
                 category_order: list[str] | None = None,
                 palette_map: dict[str, str] | None = None,
                 show_points: bool = True,
                 rotate_xticks: bool = True,
                 scientific_y: bool = False) -> None:
    """Render a per-window LargestClusterFraction violin keyed by ``group``.

    ``df`` must carry a ``group`` column and the y-column LargestClusterFraction;
    ``category_order`` fixes the violin order and ``palette_map`` the per-group
    colors. ``title`` is accepted for call-site symmetry but no title is drawn
    (manuscript panels are titleless).
    """
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(2.0, 3.2))
    palette = None
    if palette_map is not None:
        palette = [palette_map.get(name, "grey") for name in (category_order or df["group"].unique())]
    sns.violinplot(ax=ax, data=df, x="group", y="LargestClusterFraction",
                   order=category_order, inner="box", cut=0,
                   palette=palette, linewidth=2, edgecolor="k")
    if show_points:
        sns.stripplot(ax=ax, data=df, x="group", y="LargestClusterFraction",
                      color="k", size=3, alpha=0.6, order=category_order)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    if rotate_xticks:
        plt.xticks(rotation=30, ha="right")
    else:
        plt.xticks(rotation=0)
    _set_axis_tick_formatter(ax, decimals=2, scientific=scientific_y)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _prepare_violin_data(cluster_df: pd.DataFrame, by_sm: bool) -> pd.DataFrame:
    """Window-average LargestClusterFraction into one ``group`` per violin slot.

    Groups by category (and sm when ``by_sm``) plus window_start_ns so each
    violin point is a window mean; the ``group`` label is the category, or
    ``<category>-<SM>`` when ``by_sm``.
    """
    if cluster_df.empty:
        return pd.DataFrame(columns=["group", "LargestClusterFraction"])
    group_cols = ["category", "sm", "window_start_ns"] if by_sm else ["category", "window_start_ns"]
    grouped = cluster_df.groupby(group_cols)["LargestClusterFraction"].mean().reset_index()
    if by_sm:
        grouped["group"] = grouped.apply(lambda row: f"{row['category']}-{_format_sm(row['sm'])}", axis=1)
    else:
        grouped["group"] = grouped["category"]
    return grouped


def _prepare_violin_num(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Window-average NumClusters into one ``group`` (=category) per violin slot."""
    if cluster_df.empty:
        return pd.DataFrame(columns=["group", "NumClusters"])
    grouped = cluster_df.groupby(["category", "window_start_ns"]).agg({"NumClusters": "mean"}).reset_index()
    grouped["group"] = grouped["category"]
    return grouped


def _aggregate_profile(df: pd.DataFrame, category_col: str,
                       x_col: str, y_col: str) -> pd.DataFrame:
    """Mean/SEM of ``y_col`` per (category, rounded x) for a radial profile."""
    if df.empty:
        return pd.DataFrame(columns=[category_col, x_col, "mean", "sem"])
    df = df.copy()
    df["x_round"] = df[x_col].round(5)
    agg = _aggregate_mean_sem(df, [category_col, "x_round"], y_col)
    agg.rename(columns={"x_round": x_col}, inplace=True)
    return agg


def _plot_profile(agg_df: pd.DataFrame, x_col: str, y_col: str,
                  palette: dict[str, str], output_path: Path,
                  baseline: float | None = None,
                  scientific_y: bool = False,
                  xlim: tuple[float, float] | None = None,
                  xticks: list[float] | None = None,
                  auto_y_limits: bool = False) -> None:
    """Plot per-category mean radial profiles (heavy line + scatter + error bars).

    ``agg_df`` carries per-category ``mean``/``sem`` vs ``x_col``. Optional
    ``baseline`` draws a dashed reference line; ``xlim``/``xticks`` fix the x
    range; ``auto_y_limits`` snaps the y range to the enclosing decade unit.
    """
    if agg_df.empty:
        return
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for category in agg_df["category"].unique():
        data = agg_df[agg_df["category"] == category].sort_values(x_col)
        color = palette.get(category, "grey")
        # SG style: heavy line + scatter markers + error bars without caps
        ax.plot(data[x_col], data["mean"], color=color, linewidth=4, zorder=1)
        ax.scatter(data[x_col], data["mean"], color=color, s=40, edgecolor="k", linewidth=1, zorder=3)
        ax.errorbar(data[x_col], data["mean"], yerr=data["sem"], fmt=".", color=color, zorder=2)
    if baseline is not None:
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.3, zorder=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    if scientific_y:
        _set_axis_tick_formatter(ax, decimals=2, scientific=True)
    # Optional fixed x range and ticks
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    # Optional y-limits rounded to the largest decimal point
    if auto_y_limits:
        lo = (agg_df["mean"] - agg_df["sem"]).min()
        hi = (agg_df["mean"] + agg_df["sem"]).max()
        if np.isfinite(lo) and np.isfinite(hi):
            mag = 10.0 ** np.floor(np.log10(max(abs(lo), abs(hi)) if max(abs(lo), abs(hi)) > 0 else 1.0))
            unit = mag
            y0 = np.floor(lo / unit) * unit
            y1 = np.ceil(hi / unit) * unit
            if y0 == y1:
                # expand a little if degenerate
                y0 -= unit
                y1 += unit
            ax.set_ylim(y0, y1)
    ax.legend(frameon=False)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_profile_per_sm(df: pd.DataFrame, category: str,
                         x_col: str, y_col: str, palette: list,
                         output_path: Path, title: str,
                         baseline: float | None = None,
                         scientific_y: bool = False,
                         xlim: tuple[float, float] | None = None,
                         xticks: list[float] | None = None,
                         auto_y_limits: bool = False) -> None:
    """Plot one mean radial-profile line per compound for a single category.

    Same optional ``baseline``/``xlim``/``xticks``/``auto_y_limits`` controls as
    ``_plot_profile``. ``title`` is accepted for call-site symmetry but no title
    is drawn.
    """
    data = df[df["category"] == category]
    if data.empty:
        return
    agg = _aggregate_mean_sem(data, ["sm", x_col], y_col)
    order = sorted(agg["sm"].unique())
    color_cycle = palette if len(palette) >= len(order) else sns.color_palette("husl", len(order))
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for sm, color in zip(order, color_cycle):
        sm_data = agg[agg["sm"] == sm].sort_values(x_col)
        ax.plot(sm_data[x_col], sm_data["mean"], label=_format_sm(sm), color=color, linewidth=2)
    if baseline is not None:
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.3, zorder=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _apply_axis_tick_style(ax)
    if scientific_y:
        _set_axis_tick_formatter(ax, decimals=2, scientific=True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if auto_y_limits:
        lo = (agg["mean"] - agg["sem"]).min()
        hi = (agg["mean"] + agg["sem"]).max()
        if np.isfinite(lo) and np.isfinite(hi):
            mag = 10.0 ** np.floor(np.log10(max(abs(lo), abs(hi)) if max(abs(lo), abs(hi)) > 0 else 1.0))
            unit = mag
            y0 = np.floor(lo / unit) * unit
            y1 = np.ceil(hi / unit) * unit
            if y0 == y1:
                y0 -= unit
                y1 += unit
            ax.set_ylim(y0, y1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _coarsen_rdp_bins(df: pd.DataFrame, n: int,
                      category_col: str = "category",
                      sm_col: str = "sm",
                      window_col: str = "window_start_ns",
                      x_col: str = "distance",
                      y_col: str = "density",
                      drop_first_bin: bool = False,
                      drop_head_original_bins: int | None = None) -> pd.DataFrame:
    """Coarsen the RDP by averaging every n consecutive bins within each
    (category, sm, window) series. Returns a new DataFrame with coarsened
    midpoints (mean distance) and densities.

    If n <= 1 or df is empty, the input is returned unchanged.
    """
    if df.empty or n <= 1:
        return df

    df = df.copy()
    df.sort_values([category_col, sm_col, window_col, x_col], inplace=True)
    df["_idx"] = df.groupby([category_col, sm_col, window_col]).cumcount()
    # Optionally drop a fixed number of original bins per (category, sm, window)
    if drop_head_original_bins is not None:
        df = df[df["_idx"] >= int(drop_head_original_bins)]
    elif drop_first_bin:
        df = df[df["_idx"] >= 1]
        # recompute running index after drop so coarsening groups remain consecutive
        df["_idx"] = df.groupby([category_col, sm_col, window_col]).cumcount()
    df["_bin"] = (df["_idx"] // n).astype(int)

    agg = (
        df.groupby([category_col, sm_col, window_col, "_bin"], as_index=False)
          .agg({x_col: "mean", y_col: "mean"})
    )
    agg.drop(columns=["_bin"], inplace=True)
    return agg


def _downsample_time(df: pd.DataFrame, time_col: str, stride_ns: float) -> pd.DataFrame:
    """Return rows where time_col falls on multiples of stride_ns.

    Keeps all rows if stride_ns <= 0.
    """
    if df.empty or stride_ns is None or stride_ns <= 0:
        return df
    t = df[time_col].astype(float)
    k = t / float(stride_ns)
    mask = np.isclose(k - np.round(k), 0.0, atol=1e-9)
    return df.loc[mask].copy()


def generate_plots(dsm_dir: Path, ndsm_dir: Path, output_dir: Path,
                   dt_ns: float = 1.0, window_ns: float = 20.0,
                   tmin: float = 40.0, tmax: float = 200.0,
                   coarsen_bins: int = 4,
                   plot_stride_ns: float = 5.0) -> None:
    """Render the full set of DSM/NDSM SM-isolation figures.

    Loads cluster/RDP/RDF CSVs from ``dsm_dir`` and ``ndsm_dir`` (restricted to
    [tmin, tmax]), aggregates per category, and writes aggregated/, dsm/, ndsm/
    PNG panels under ``output_dir``. ``coarsen_bins`` averages every N radial
    bins of the density profiles and ``plot_stride_ns`` thins the cluster time
    series to reduce clutter.
    """
    _configure_style()

    categories = {
        "DSM": Path(dsm_dir),
        "NDSM": Path(ndsm_dir),
    }

    cluster_frames = []
    rdp_number_frames = []
    rdp_mass_frames = []
    rdf_frames = []

    for cat_name, cat_path in categories.items():
        cluster_df = _load_cluster(cat_path, dt_ns, window_ns, tmin, tmax)
        cluster_frames.append(cluster_df)

        rdp_num = _load_rdp_number(cat_path, dt_ns, window_ns, tmin, tmax)
        if not rdp_num.empty:
            rdp_num["category"] = cat_name
            rdp_number_frames.append(rdp_num)

        rdp_mass = _load_rdp_mass(cat_path, dt_ns, window_ns, tmin, tmax)
        if not rdp_mass.empty:
            rdp_mass["category"] = cat_name
            rdp_mass_frames.append(rdp_mass)

        rdf_df = _load_profile(cat_path, "RDF_SM_COM_*.csv",
                               "g(r)", dt_ns, window_ns, tmin, tmax)
        rdf_df["category"] = cat_name
        rdf_frames.append(rdf_df)

    cluster_all = pd.concat(cluster_frames, ignore_index=True)
    rdp_number_all = pd.concat(rdp_number_frames, ignore_index=True) if rdp_number_frames else pd.DataFrame()
    rdp_mass_all = pd.concat(rdp_mass_frames, ignore_index=True) if rdp_mass_frames else pd.DataFrame()
    rdf_all = pd.concat(rdf_frames, ignore_index=True)

    # Match manuscript green palette (dark vs light green)
    palette_category = {
        "DSM": "#40641b",   # dark green
        "NDSM": "#bfe49b",  # light green
    }

    # Aggregated cluster time plot
    cluster_agg = _aggregate_by_sm_then_category(cluster_all, "category", "sm", "time_ns", "LargestClusterFraction")
    cluster_agg = _downsample_time(cluster_agg, "time_ns", plot_stride_ns)
    _plot_cluster_time(cluster_agg, palette_category, output_dir / "aggregated" / "cluster_time.png")

    cluster_num_agg = _aggregate_by_sm_then_category(cluster_all, "category", "sm", "time_ns", "NumClusters")
    cluster_num_agg = _downsample_time(cluster_num_agg, "time_ns", plot_stride_ns)
    _plot_cluster_number(cluster_num_agg, palette_category, output_dir / "aggregated" / "cluster_number.png")

    # Cluster time per category and SM
    # Per-SM color cycles: DSM darker half of rocket, NDSM lighter half
    rocket = sns.color_palette("rocket", 10)
    per_sm_dsm = rocket[0:5]
    per_sm_ndsm = rocket[5:10]
    _plot_cluster_time_per_sm(cluster_all, "DSM", per_sm_dsm, output_dir / "dsm" / "cluster_time.png")
    _plot_cluster_time_per_sm(cluster_all, "NDSM", per_sm_ndsm, output_dir / "ndsm" / "cluster_time.png")
    _plot_cluster_number_per_sm(cluster_all, "DSM", per_sm_dsm, output_dir / "dsm" / "cluster_number.png")
    _plot_cluster_number_per_sm(cluster_all, "NDSM", per_sm_ndsm, output_dir / "ndsm" / "cluster_number.png")

    # Violin plots
    violin_cat = _prepare_violin_data(cluster_all, by_sm=False)
    _plot_violin(violin_cat.assign(group=violin_cat["group"].astype(str)),
                 output_dir / "aggregated" / "cluster_violin.png",
                 "", ["DSM", "NDSM"], palette_category,
                 show_points=False, rotate_xticks=False, scientific_y=True)

    # Violin for number of clusters (aggregated)
    violin_num = _prepare_violin_num(cluster_all)
    _plot_violin(violin_num.rename(columns={"NumClusters": "LargestClusterFraction"}).assign(
                    group=violin_num["group"].astype(str)),
                 output_dir / "aggregated" / "cluster_number_violin.png",
                 "", ["DSM", "NDSM"], palette_category,
                 show_points=False, rotate_xticks=False, scientific_y=True)

    for category in ["DSM", "NDSM"]:
        cat_df = cluster_all[cluster_all["category"] == category]
        violin_sm = _prepare_violin_data(cat_df, by_sm=True)
        order = sorted(violin_sm["group"].unique())
        sub_palette = sns.color_palette("rocket", len(order))
        palette_map = {group: color for group, color in zip(order, sub_palette)}
        _plot_violin(violin_sm, output_dir / category.lower() / "cluster_violin.png",
                     f"{category} cluster distribution", order, palette_map)

    # RDP number density (prefer molecule-number density if present)
    if not rdp_number_all.empty:
        # Determine which column to use as density
        use_molecule_col = "SM molecule number density (1/A^3)" in rdp_number_all.columns
        if use_molecule_col:
            rdp_number_all = rdp_number_all.rename(columns={
                "Distance from center of mass (A)": "distance",
                "SM molecule number density (1/A^3)": "density",
            })
        else:
            # Fall back to bead number density; if beads-per-molecule is present, convert on the fly
            rdp_number_all = rdp_number_all.rename(columns={
                "Distance from center of mass (A)": "distance",
            })
            if "SM number density (1/A^3)" in rdp_number_all.columns and "Beads per molecule" in rdp_number_all.columns:
                rdp_number_all["density"] = (
                    rdp_number_all["SM number density (1/A^3)"] / rdp_number_all["Beads per molecule"].astype(float)
                )
            else:
                rdp_number_all = rdp_number_all.rename(columns={
                    "SM number density (1/A^3)": "density"
                })

        # Drop the first two original bins to avoid near-origin spike
        rdp_number_all = _coarsen_rdp_bins(rdp_number_all, coarsen_bins,
                                           category_col="category", sm_col="sm",
                                           window_col="window_start_ns",
                                           x_col="distance", y_col="density",
                                           drop_head_original_bins=2)
        rdp_number_cat = _aggregate_by_sm_then_category(rdp_number_all, "category", "sm", "distance", "density")
        _plot_profile(rdp_number_cat, "distance", "density", palette_category,
                      output_dir / "aggregated" / "rdp_number.png", scientific_y=True,
                      xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)
        _plot_profile_per_sm(rdp_number_all, "DSM", "distance", "density",
                             per_sm_dsm, output_dir / "dsm" / "rdp_number.png",
                             "", scientific_y=True,
                             xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)
        _plot_profile_per_sm(rdp_number_all, "NDSM", "distance", "density",
                             per_sm_ndsm, output_dir / "ndsm" / "rdp_number.png",
                             "", scientific_y=True,
                             xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)

    # RDP mass density
    if not rdp_mass_all.empty:
        rdp_mass_all = rdp_mass_all.rename(columns={
            "Distance from center of mass (A)": "distance",
            "SM mass density (mg/mL)": "density"
        })
        rdp_mass_all = _coarsen_rdp_bins(rdp_mass_all, coarsen_bins,
                                         category_col="category", sm_col="sm",
                                         window_col="window_start_ns",
                                         x_col="distance", y_col="density",
                                         drop_head_original_bins=2)
        rdp_mass_cat = _aggregate_by_sm_then_category(rdp_mass_all, "category", "sm", "distance", "density")
        _plot_profile(rdp_mass_cat, "distance", "density", palette_category,
                      output_dir / "aggregated" / "rdp_mass.png", scientific_y=True,
                      xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)
        _plot_profile_per_sm(rdp_mass_all, "DSM", "distance", "density",
                             per_sm_dsm, output_dir / "dsm" / "rdp_mass.png",
                             "", scientific_y=True,
                             xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)
        _plot_profile_per_sm(rdp_mass_all, "NDSM", "distance", "density",
                             per_sm_ndsm, output_dir / "ndsm" / "rdp_mass.png",
                             "", scientific_y=True,
                             xlim=(0, 1200), xticks=list(range(0, 1201, 200)), auto_y_limits=True)

    # RDF aggregated
    if not rdf_all.empty:
        rdf_all.rename(columns={"Distance (A)": "distance", "g(r)": "rdf"}, inplace=True)
        rdf_cat = _aggregate_by_sm_then_category(rdf_all, "category", "sm", "distance", "rdf")
        _plot_profile(rdf_cat, "distance", "rdf", palette_category,
                      output_dir / "aggregated" / "rdf.png", baseline=1.0)
        _plot_profile_per_sm(rdf_all, "DSM", "distance", "rdf",
                             per_sm_dsm, output_dir / "dsm" / "rdf.png",
                             "", baseline=1.0)
        _plot_profile_per_sm(rdf_all, "NDSM", "distance", "rdf",
                             per_sm_ndsm, output_dir / "ndsm" / "rdf.png",
                             "", baseline=1.0)


def main():
    """Parse CLI arguments and run ``generate_plots`` for the DSM/NDSM cohorts."""
    parser = argparse.ArgumentParser(description="Generate solubility plots for DSM/NDSM small molecules")
    parser.add_argument("--dsm-dir", type=Path, default=Path("ANALYSIS_DSM"),
                        help="Directory containing DSM analysis CSV files")
    parser.add_argument("--ndsm-dir", type=Path, default=Path("ANALYSIS_NDSM"),
                        help="Directory containing NDSM analysis CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("ANALYSIS_SM_PLOTS"),
                        help="Where to write figures")
    parser.add_argument("--dt-ns", type=float, default=1.0, help="Time spacing in ns for cluster data")
    parser.add_argument("--window-ns", type=float, default=20.0, help="Window length in ns")
    parser.add_argument("--trim-start", type=float, default=40.0, help="Discard data before this time (ns)")
    parser.add_argument("--trim-end", type=float, default=200.0, help="Discard data after this time (ns)")
    args = parser.parse_args()

    generate_plots(args.dsm_dir, args.ndsm_dir, args.output_dir,
                   dt_ns=args.dt_ns, window_ns=args.window_ns,
                   tmin=args.trim_start, tmax=args.trim_end)


if __name__ == "__main__":
    main()
