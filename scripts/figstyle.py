"""House style for every figure on the site — publication standard.

Rules applied here:

* vector-quality output: PDF **and** 300 dpi PNG (PNG for the web, PDF for print);
* single-column (8.6 cm) / double-column (17.2 cm) widths, so figures drop straight
  into a paper or a slide without rescaling;
* Computer-Modern maths, 8 pt type, hairline axes, no chartjunk;
* Okabe–Ito colour-blind-safe palette;
* every panel carries an ``(a)`` / ``(b)`` label and an explanatory caption;
* deterministic: seeded RNGs only, so a re-render is byte-stable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.text
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "assets" / "figures"

CM = 1 / 2.54
WIDTH_1COL = 3.4       # inches ~ 8.6 cm
WIDTH_1HALF = 4.6
WIDTH_2COL = 6.8       # inches ~ 17.3 cm
HEIGHT = 2.6

# Okabe–Ito colour-blind-safe palette
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#000000"]

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.prop_cycle": plt.cycler(color=PAL),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.color": "#8b8b8b",
        "grid.linestyle": ":",
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 110,
    }
)

PANEL_LABELS = iter("abcdefgh")


def new_fig(nrows: int = 1, ncols: int = 1, width: float = WIDTH_2COL, ratio: float = 0.42):
    """Figure at publication width, with a fixed number of equal panels."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, width * ratio * max(nrows, 1) ** 0.5),
                             squeeze=False, constrained_layout=True)
    return fig, axes


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.02, 1.05, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom")


def save(fig, stem: str) -> Path:
    """Write ``stem.pdf`` + ``stem.png`` at print quality into ``assets/figures``.

    Every text artist in the figure is checked for non-ASCII first, so a Chinese
    label cannot sneak into a chart (house rule: figures stay English-only).
    """
    ascii_only(" ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text)),
               f"labels of {stem}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png.name} + {pdf.name}")
    return png


# --------------------------------------------------------------------------- #
# House rule (AGENTS.MD): **no CJK characters in charts and plots.**
# They render as tofu boxes in exported figures, so every string that reaches a
# matplotlib text object must be ASCII. ``ascii_only`` enforces it, and ``save``
# checks every text artist in the figure before writing. Chinese belongs in the
# page prose and in figure *captions* (markdown), never in the pixels.
# --------------------------------------------------------------------------- #

def ascii_only(text: str, where: str = "figure text") -> str:
    """Return ``text`` unchanged, or raise if it contains a non-ASCII character."""
    bad = sorted({c for c in str(text) if ord(c) > 127})
    if bad:
        raise ValueError(
            f"{where} must stay ASCII (CJK renders as tofu in exported figures); "
            f"offending characters: {' '.join(bad)}\n  text: {text!r}"
        )
    return str(text)
