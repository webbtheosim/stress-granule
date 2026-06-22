"""Self-self domain CONTACT MEMBERSHIP (INTER) maps, each isolated + self-scaled.

The per-pair domain maps from contact_maps.py share ONE universal colour scale, so the
weak self-self blocks (e.g. TDP43-TDP43) wash out to flat pink against the
strongest pair. Here every self-self diagonal block is sliced from the FULL_FULL
membership matrix and rendered on ITS OWN scale (0 -> that block's max), so each
protein's internal self-interaction structure is visible.

Maps are sized so the heatmap axis is 1.747 in -- the SAME axis size as the
figure-S12 maps -- and saved as individual PNGs (assembled into a figure by
hand downstream).

Reuses contact_maps.py's exact plotting machinery (sub-domain colour bars, RNA flip,
house scientific-notation colour bar) so the maps match the rest of the figures.

Paper figure: SI 13 (per-species self-self domain maps).

Inputs (no CLI flags; hard-coded to the 300 K SG correlated run): the FULL_FULL
contact-membership CSV under that run's domain contact-map directory.
Outputs: individual PNGs in FIGURES_SI/self_self_domain_membership/ (one per
species), assembled into the figure by hand downstream.

Run with:
    python si_selfself_domain_maps.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import contact_maps as A

SYS = "SG"
TEMP = 300
S12_AX_IN = 1.747          # heatmap axis size of the S12 maps -> match it exactly

# Set ARRAY's figure width so the square-map branch yields ax = S12_AX_IN.
# (square ax_w = _FIG_MAX_IN - left - right_aux, with right_aux including the
#  y-domain-bar + colourbar pad + colourbar width.)
_RIGHT_AUX = (A._RIGHT_MARGIN_IN + A._DOMAIN_BAR_IN + A._CBAR_PAD_IN + A._CBAR_WIDTH_IN)
A._FIG_MAX_IN = S12_AX_IN + A._LEFT_MARGIN_IN + _RIGHT_AUX

SRC = Path(f"TEMP_{TEMP}/CLASSIFY_CORRELATED_{TEMP}_50_50_2000/FIGURES/DOMAIN_CONTACT_MAPS/{SYS}")
CSV = SRC / "Domain_Contact_Map_FULL_FULL_CONTACT_MEMBERSHIP_INTER.csv"
OUT = Path("FIGURES_SI/self_self_domain_membership"); OUT.mkdir(parents=True, exist_ok=True)

BIO = A.BIOPOLYMER_LIST            # ProteinFUS, ProteinTIA1, ... RNA
LEN = A.BIOPOLYMER_LENGTHS         # 526, 386, 414, 466, 636, 326, 840
OFF = np.cumsum([0] + LEN)

print(f"loading {CSV} ...")
M = pd.read_csv(CSV, header=None, dtype=np.float32).to_numpy()
print(f"  full matrix {M.shape};  ax set to {S12_AX_IN} in (fig {A._FIG_MAX_IN:.3f} in)")
assert M.shape[0] == OFF[-1]


def render_block(bio):
    """Render one self-self block at S12 axis size to an individual PNG."""
    i = BIO.index(bio)
    name = bio.replace("Protein", "")
    s, e = int(OFF[i]), int(OFF[i + 1])
    block = A._flip_rna_axes(M[s:e, s:e].astype(float), bio, bio)
    vmax = float(np.nanmax(block))
    # mirror the RNA sub-domain bar to track the flipped RNA axis (poly-A at the
    # boundary), matching the heatmap -- same fix as the FULL_FULL maps
    bars = A._flip_rna_bars(A._bars_for_biopolymer(bio, block.shape[0]), bio, block.shape[0])
    out = OUT / f"SelfSelf_{name}_MEMBERSHIP_INTER.png"
    # flip_y_bars=False here (unlike the FULL_FULL maps): these single-species
    # blocks pass no block_boundaries, so ARRAY's compensating right-bar invert
    # never fires; with flip_y_bars=True the right bar would render reversed
    # relative to the (y-inverted) heatmap. False makes top and right bars agree.
    A._plot_contact_heatmap(
        block, str(out), symmetric=True, x_bars=bars, y_bars=bars,
        flip_y_bars=False, invert_y=False, apply_neg_log_relative=False,
        cmap_name="Reds", cbar_limits=(0.0, vmax if vmax > 0 else 1.0),
    )
    px = Image.open(out).size
    print(f"  {name:6s} block {block.shape[0]:>3d}^2  vmax={vmax:.2e}  -> {out.name} "
          f"({px[0]}x{px[1]}px, ax {S12_AX_IN} in)")


# individual maps only -- figure assembled by hand downstream
for bio in BIO:
    render_block(bio)
print("done")
