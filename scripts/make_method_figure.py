"""Method overview figure (Figure 1 of the manuscript).

Panel (a) Frame-ISRT  : normalized main path + controlled scale-bearing residual
Panel (b) Node residual: boundary test on a scale-PRESERVING backbone (TranSG)

The colour pair (blue / orange) stays distinguishable under the common forms of
colour-vision deficiency, and every distinction that carries meaning is also
encoded by position and by text.

Writes ../figures/fig_method_overview.pdf (vector) and .png (600 dpi preview).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------
BLUE_F, BLUE_E = "#DCE9F5", "#2F6BA8"      # main path
ORAN_F, ORAN_E = "#FBE3D0", "#C4651F"      # scale-bearing residual path
GREY_F, GREY_E = "#EFEFEF", "#5A5A5A"      # neutral plumbing
GREEN_F, GREEN_E = "#DFF0DE", "#3E7A38"    # outcome

FS = 7.8        # box text
FS_S = 6.8      # small annotations
FS_T = 9.5      # panel titles
FS_EQ = 8.8     # equations


def box(ax, cx, cy, w, h, text, fc=GREY_F, ec=GREY_E, fs=FS, lw=1.1):
    ax.add_patch(
        FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=1.6",
                       linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2)
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#5A5A5A", lw=1.2):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=9, linewidth=lw, color=color,
                        zorder=1, shrinkA=0, shrinkB=0)
    )


def gate(ax, cx, cy, r=2.5):
    """Learnable scalar gate: a circle with a multiply sign, labelled."""
    ax.add_patch(Circle((cx, cy), r, facecolor="white",
                        edgecolor=ORAN_E, linewidth=1.2, zorder=4))
    ax.text(cx, cy, r"$\times$", ha="center", va="center",
            fontsize=FS, color=ORAN_E, zorder=5)
    ax.text(cx, cy - r - 1.2, r"$\sigma(g)$" "\n" "learnable\nscalar gate",
            ha="center", va="top", fontsize=FS_S, color=ORAN_E,
            zorder=5, linespacing=1.3)


# ======================================================================
fig, (axa, axb) = plt.subplots(2, 1, figsize=(7.4, 7.0))

for ax in (axa, axb):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

# ----------------------------------------------------------------------
# Panel (a)
# ----------------------------------------------------------------------
axa.set_title("(a)  Frame-ISRT: controlled scale re-introduction on a "
              "scale-normalizing temporal encoder",
              fontsize=FS_T, weight="bold", loc="left", pad=7)

box(axa, 50, 94, 46, 10,
    "Input skeleton window $X\\in\\mathbb{R}^{T\\times J\\times3}$\n"
    "$T=6$ frames, $J=20$ joints")
arrow(axa, 50, 89, 50, 85.6)

box(axa, 50, 80, 42, 10,
    "Root centering\n$C_{t,j}=X_{t,j}-X_{t,r}$")

arrow(axa, 42, 75, 24, 73, color=BLUE_E)
arrow(axa, 58, 75, 80, 73, color=ORAN_E)

box(axa, 24, 64, 36, 16,
    "$\\mathbf{Main\\ path}$  ---  scale deleted\n"
    "$s=\\mathrm{median}$ bone length,  $N=C/s$\n"
    "$\\mathrm{vec}(N_t)\\rightarrow W_n\\in\\mathbb{R}^{128\\times60}$",
    BLUE_F, BLUE_E)

box(axa, 80, 64, 34, 16,
    "$\\mathbf{Residual\\ path}$  ---  scale bearing\n"
    "$C$ (absolute magnitude),  $\\mathrm{vec}(C_t)$\n"
    "$W_s\\in\\mathbb{R}^{128\\times60}$, zero-init.",
    ORAN_F, ORAN_E)

gate(axa, 52, 67.5)

arrow(axa, 24, 56, 24, 52.8, color=BLUE_E)
arrow(axa, 80, 56, 80, 52.8, color=ORAN_E)

box(axa, 50, 46.5, 70, 13,
    "$z_t=W_n\\,\\mathrm{vec}(N_t)\\;+\\;\\sigma(g)\\,W_s\\,\\mathrm{vec}(C_t)$\n"
    "zero-initialised $W_s$ $\\Rightarrow$ identical to the baseline at step 0",
    "white", "#333333", fs=FS_EQ)
arrow(axa, 50, 40, 50, 34.0)

box(axa, 50, 28.5, 64, 11,
    "Temporal encoder (shared config)\n"
    "GRU  /  Transformer  /  Mamba  $\\rightarrow$  mean pool $+$ $L^2$ norm")
arrow(axa, 50, 23, 50, 17.0)

box(axa, 50, 11.5, 84, 11,
    "$f(X)\\in\\mathbb{R}^{128}$  ---  ArcFace $+\\,0.2\\times$ batch-hard "
    "triplet  ---  cosine retrieval (mAP, Rank-$k$)", GREEN_F, GREEN_E)

# ----------------------------------------------------------------------
# Panel (b)
# ----------------------------------------------------------------------
axb.set_title("(b)  Boundary test: node residual on a scale-preserving "
              "backbone (TranSG)",
              fontsize=FS_T, weight="bold", loc="left", pad=7)

box(axb, 50, 94, 56, 10,
    "Root-centered coordinates $C_{t,j}$\n"
    "$\\mathbf{scale\\ already\\ preserved\\ by\\ the\\ main\\ path}$")
arrow(axb, 42, 89, 24, 84.5, color=BLUE_E)
arrow(axb, 58, 89, 80, 84.5, color=ORAN_E)

box(axb, 24, 76, 36, 16,
    "$\\mathbf{TranSG\\ main\\ path}$\n"
    "$\\mathrm{MLP}(C_{t,j})\\rightarrow$ node token\n"
    "no scale removed", BLUE_F, BLUE_E)

box(axb, 80, 76, 34, 16,
    "$\\mathbf{Node\\ residual}$\n"
    "$W_s\\in\\mathbb{R}^{128\\times3}$, zero-init.\n"
    "bias-free, $+385$ parameters", ORAN_F, ORAN_E)

gate(axb, 52, 79.5)

arrow(axb, 24, 68, 24, 59.8, color=BLUE_E)
arrow(axb, 80, 68, 80, 59.8, color=ORAN_E)

box(axb, 50, 53.5, 70, 12.5,
    "$h_{t,j}=\\mathrm{MLP}(C_{t,j})\\;+\\;\\sigma(g)\\,W_s\\,C_{t,j}$",
    "white", "#333333", fs=FS_EQ)
arrow(axb, 50, 47.2, 50, 41.0)

box(axb, 50, 35, 68, 12,
    "TranSG graph Transformer\n"
    "prototype contrastive learning $+$ structure--trajectory reconstruction")
arrow(axb, 50, 29, 50, 23.0)

box(axb, 50, 17, 84, 12,
    "$f(X)$  ---  Euclidean retrieval (mAP, Rank-$k$)\n"
    "clean target: $\\Delta$mAP $=-0.04$ pp over 12 paired runs",
    GREEN_F, GREEN_E)

# ----------------------------------------------------------------------
fig.text(0.5, 0.013,
         "Both panels use the same residual construction "
         "($W_s$ zero-initialised, magnitude limited by $\\sigma(g)$); "
         "they differ in what the main path does with scale.",
         ha="center", va="bottom", fontsize=FS_S, color="#333333")

fig.subplots_adjust(left=0.015, right=0.985, top=0.955, bottom=0.045,
                    hspace=0.16)

fig.savefig(OUT / "fig_method_overview.pdf", dpi=300)
fig.savefig(OUT / "fig_method_overview.png", dpi=600)
print(f"wrote {OUT / 'fig_method_overview.pdf'} and .png")
