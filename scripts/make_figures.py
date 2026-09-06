#!/usr/bin/env python3
"""Render every static figure used by the site into ``assets/figures``.

    python3 scripts/make_figures.py            # all figures
    python3 scripts/make_figures.py jacobi     # only stems containing "jacobi"

Each figure is written twice: 300 dpi PNG for the web and PDF for print.
The numbers quoted in ``fig_sarcos_protocol`` are read from the published
SARCOS-with-Agents results (https://math4mad.github.io/SARCOS-ML-with-Agents/),
see that page for the protocol; nothing there is re-fitted live.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_jacobi, roots_jacobi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (CJK_FONT, PAL, WIDTH_1COL, WIDTH_2COL, new_fig, panel_label, save)  # noqa: E402

X = np.linspace(-1, 1, 801)
XS = np.linspace(-1 + 1e-6, 1 - 1e-6, 801)   # endpoints excluded: w can blow up there


# --------------------------------------------------------------------------- #
# 1. Jacobi polynomial families
# --------------------------------------------------------------------------- #
def fig_jacobi_basis():
    """How the pair (alpha, beta) warps the same polynomial degree."""
    fig, ax = new_fig(1, 2, width=WIDTH_2COL, ratio=0.38)
    a, b = ax[0]

    for n in range(0, 6):
        p = eval_jacobi(n, 0.0, 0.0, X)
        a.plot(X, p, lw=1.5 if n else 1.0, color=PAL[n % len(PAL)], label=f"$n={n}$")
    a.axhline(0, color="#444", lw=0.6)
    a.set_xlabel("$x$")
    a.set_ylabel("$P_n^{(0,0)}(x)$")
    a.set_title("(a) Degree grows, $\\alpha=\\beta=0$")
    a.set_ylim(-1.6, 1.6)
    a.legend(ncol=3, loc="upper center", columnspacing=0.9, handlelength=1.4)
    panel_label(a, "")

    weight_pairs = [(0.0, 0.0), (-0.5, -0.5), (0.5, 0.5), (2.0, -0.6), (-0.6, 2.0)]
    labels = {
        (0.0, 0.0): "Legendre $(0,0)$",
        (-0.5, -0.5): "Chebyshev I $(-\\frac{1}{2},-\\frac{1}{2})$",
        (0.5, 0.5): "Chebyshev II $(\\frac{1}{2},\\frac{1}{2})$",
        (2.0, -0.6): "$(2,-0.6)$",
        (-0.6, 2.0): "$(-0.6,2)$",
    }
    for (al, be), col in zip(weight_pairs, PAL):
        p = eval_jacobi(4, al, be, X)
        p = p / np.max(np.abs(p))
        b.plot(X, p, color=col, label=labels[(al, be)])
    b.axhline(0, color="#444", lw=0.6)
    b.set_xlabel("$x$")
    b.set_ylabel("$P_4^{(\\alpha,\\beta)}(x)\\,/\\,\\|P_4\\|_\\infty$")
    b.set_title("(b) Degree fixed, the weight moves")
    b.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.4,
             handlelength=1.6, columnspacing=1.1)
    save(fig, "jacobi-basis")


# --------------------------------------------------------------------------- #
# 2. Weight functions and Gauss-Jacobi nodes
# --------------------------------------------------------------------------- #
def fig_jacobi_weights():
    """Weight functions and where Gauss-Jacobi nodes land for $n=12$."""
    pairs = [(0.0, 0.0, "Legendre $w=1$"),
             (-0.5, -0.5, "Chebyshev I $w=(1-x^2)^{-1/2}$"),
             (0.5, 0.5, "Chebyshev II $w=(1-x^2)^{1/2}$"),
             (1.25, 0.75, "general $(1-x)^{1.25}(1+x)^{0.75}$")]
    fig, ax = new_fig(1, 2, width=WIDTH_2COL, ratio=0.38)
    a, b = ax[0]

    for (al, be, lab), col in zip(pairs, PAL):
        w = (1 - XS) ** al * (1 + XS) ** be
        w = w / np.trapezoid(w, XS)
        a.plot(XS, w, color=col, lw=1.5, label=lab)
    a.set_xlabel("$x$")
    a.set_ylabel("$w(x)$ (normalised)")
    a.set_yscale("log")
    a.set_ylim(2e-3, 6e1)
    a.set_title("(a) The weight defines $L^2_w([-1,1])$")
    a.legend(fontsize=6.3, loc="upper left")

    for k, ((al, be, lab), col) in enumerate(zip(pairs, PAL)):
        nodes, wts = roots_jacobi(12, al, be)
        area = 90 * wts / wts.max()
        b.scatter(nodes, np.full_like(nodes, k), s=area, color=col, alpha=0.9,
                  edgecolors="white", linewidths=0.4)
    b.set_yticks(range(len(pairs)),
                 [p[2].split("$")[0].strip() for p in pairs], fontsize=6.6)
    b.set_ylim(-0.6, len(pairs) - 0.4)
    b.set_xlabel("$x$")
    b.set_title("(b) $n=12$ Gauss-Jacobi nodes")
    b.grid(axis="y", visible=False)
    save(fig, "jacobi-weights-nodes")


# --------------------------------------------------------------------------- #
# 3. Learning as stretching a representation space
# --------------------------------------------------------------------------- #
def fig_representation_stretch():
    """Schematic: category learning stretches the task-relevant axis and
    shrinks the irrelevant one, so two classes separate (low dim -> high dim)."""
    rng = np.random.default_rng(20260906)
    n = 90
    # latent axes: dim 0 = category-relevant, dim 1 = irrelevant nuisance
    label = np.r_[np.zeros(n, int), np.ones(n, int)]
    rel = np.r_[rng.normal(-0.55, 0.55, n), rng.normal(0.55, 0.55, n)]
    irr = rng.normal(0, 1.0, 2 * n)

    fig, ax = new_fig(1, 3, width=WIDTH_2COL, ratio=0.40)
    panels = ax[0]
    stages = [
        ("(a) naive code", 1.0, 1.0),
        ("(b) attention spotlighting", 3.0, 1.0),
        ("(c) learned code", 4.2, 0.30),
    ]
    for p, (title, s_rel, s_irr) in zip(panels, stages):
        pts = np.c_[rel * s_rel + irr * s_irr * 0.55, irr * s_irr]
        for g, col, name in ((0, PAL[0], "class A"), (1, PAL[1], "class B")):
            m = label == g
            p.scatter(pts[m, 0], pts[m, 1], s=11, c=col, alpha=0.8,
                      edgecolors="white", linewidths=0.35, label=name)
        mu = pts.mean(0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        t = np.linspace(0, 2 * np.pi, 100)
        ell = mu[:, None] + vecs @ np.diag(np.sqrt(vals) * 2.0) @ np.array([np.cos(t), np.sin(t)])
        p.plot(ell[0], ell[1], color="#666", lw=0.8, ls="--")
        p.set_title(title, fontsize=7.6)
        p.set_xlabel("$r$ (task-relevant)")
        p.set_ylabel("$u$ (nuisance)")
        d = np.abs(pts[label == 1, 0].mean() - pts[label == 0, 0].mean())
        sp = 0.5 * (pts[label == 1, 0].std() + pts[label == 0, 0].std())
        p.text(0.03, 0.95, f"$d' = {d / sp:.2f}$", transform=p.transAxes, va="top",
               fontsize=8, bbox=dict(fc="white", ec="#bbb", lw=0.5, pad=2.2))
        p.set_aspect("auto")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.045))
    save(fig, "representation-stretch")


# --------------------------------------------------------------------------- #
# 4. SARCOS: leaked split vs leakage-free split
# --------------------------------------------------------------------------- #
SARCOS = {  # test RMSE (N m, mean of 7 joints), from the published results table
    "MLP (256,256,128)": (0.6413, 2.2776, 0.6266),
    "Exact GP, ARD-RBF\n(2 000 rows)": (1.4020, 2.9179, 0.5944),
    "OLS / Ridge": (2.6298, 2.9879, 0.4371),
    "PyStan linear": (2.6353, 2.9869, 0.4308),
    "kNN-20": (0.45, np.nan, np.nan),
}


def fig_sarcos_protocol():
    """Same model, two protocols: capacity buys far less than the leaked split claims."""
    names = list(SARCOS)[:-1]
    gpml = [SARCOS[k][0] for k in names]
    block = [SARCOS[k][1] for k in names]
    sd = [SARCOS[k][2] for k in names]
    y = np.arange(len(names))
    h = 0.36

    fig, ax = new_fig(1, 2, width=WIDTH_2COL, ratio=0.42)
    a, b = ax[0]
    a.barh(y + h / 2, gpml, height=h, color=PAL[1], label="published GPML split")
    a.barh(y - h / 2, block, height=h, color=PAL[0], xerr=sd,
           error_kw=dict(lw=0.8, capsize=2, ecolor="#333"), label="leakage-free block split")
    box = dict(fc="white", ec="none", alpha=0.8, pad=1.2)
    for i, (g, bl) in enumerate(zip(gpml, block)):
        a.text(bl + 0.1, i - h / 2, f"{bl:.2f}", va="center", fontsize=7, bbox=box)
        a.text(g + 0.1, i + h / 2, f"{g:.2f}  ($\\times${bl / g:.1f})", va="center",
               fontsize=7, bbox=box)
    a.set_yticks(y, [n.replace("\n", " ") for n in names], fontsize=7)
    a.invert_yaxis()
    a.set_xlabel("test RMSE (N m, mean of 7 joints)")
    a.set_xlim(0, 6.4)
    a.set_title("(a) Same model, two protocols")
    a.legend(loc="upper right", fontsize=6.6, framealpha=0.95, edgecolor="#ccc")

    # inflation factor, ordered
    ratio = np.array([bl / g for g, bl in zip(gpml, block)])
    order = np.argsort(ratio)
    b.barh(np.arange(len(names)), ratio[order], color=[PAL[3] if r > 2 else PAL[5] for r in ratio[order]])
    b.set_yticks(np.arange(len(names)), [names[i].replace("\n", " ") for i in order], fontsize=7)
    for i, r in enumerate(ratio[order]):
        b.text(r + 0.07, i, f"{r:.2f}$\\times$", va="center", fontsize=7)
    b.axvline(1.0, color="#333", lw=0.8, ls="--")
    b.set_xlabel("error inflation  RMSE(block) / RMSE(GPML)")
    b.set_xlim(0, 4.6)
    b.set_title("(b) Memorising models inflate most")
    save(fig, "sarcos-protocol")


# --------------------------------------------------------------------------- #
# 5. Site cover card
# --------------------------------------------------------------------------- #
def fig_cover():
    """1280x720 social/cover card: the Jacobi basis as the visual motif."""
    from figstyle import FIG_DIR

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("#0e1626")
    fig.patch.set_facecolor("#0e1626")
    t = np.linspace(-1, 1, 600)
    for i, n in enumerate(range(1, 8)):
        y = eval_jacobi(n, 0.0, 0.0, t)
        ax.plot(0.06 + 0.88 * (t + 1) / 2, 0.5 + 0.34 * y * np.exp(-0.06 * n),
                color=PAL[i % len(PAL)], lw=2.0, alpha=0.85)
    ax.text(0.055, 0.86, "Talk With Agents", color="white", fontsize=40,
            fontweight="bold", va="top", family="serif")
    ax.text(0.058, 0.655, "working notes on talking to, and coding with, LLM agents",
            color="#9fb4d0", fontsize=15, va="top")
    ax.text(0.058, 0.075, "Mathematics · Learning theory · Snippets · Agent-built repos",
            color="#5f7794", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "site-cover.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("  wrote site-cover.png")
    return out



# --------------------------------------------------------------------------- #
# 6. Singular-value spectrum: three bands, three different jobs
# --------------------------------------------------------------------------- #
def _synthetic_layer(d: int = 320, seed: int = 7):
    """A synthetic 'trained layer': a few dominant directions, a middle band, and a
    weak tail that carries a specific (probe) function. All numbers below are
    measured on this matrix, so the figure is reproducible, not decorative."""
    rng = np.random.default_rng(seed)

    def ortho(k):
        Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        return Q

    U, V = ortho(d), ortho(d)
    spec = np.zeros(d)
    spec[:3] = [12.0, 4.0, 2.0]                       # dominant band
    spec[3:24] = 0.8 * np.linspace(1.0, 0.55, 21)     # middle band
    tail = np.arange(24, d)
    spec[tail] = 0.05 * (1 + rng.random(tail.size))   # weak tail
    W = (U * spec) @ V.T
    return W, spec, U, V, tail


def fig_svd_spectrum():
    """Spectrum, energy concentration, and the two answers to 'is this band needed?'."""
    W, _, _, _, _ = _synthetic_layer()
    d = W.shape[0]
    Us, sv, Vt = np.linalg.svd(W, full_matrices=False)
    k = np.arange(1, d + 1)
    energy = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(energy)

    tail = np.arange(d - d // 10, d)                 # the "minor components" band

    def band_part(spec, idxs):
        """The part of the operator that lives on singular directions ``idxs``."""
        return (Us[:, idxs] * spec[idxs]) @ Vt[idxs, :]

    P_tail = band_part(sv, tail)
    tail_norm = np.linalg.norm(P_tail)

    rows = []
    for name, idxs in (("top-3 (dominant)", np.arange(0, 3)),
                       ("middle band", np.arange(3, 24)),
                       ("last decile (minor)", tail)):
        spec = sv.copy()
        spec[idxs] = 0.0
        W_trunc = (Us * spec) @ Vt
        fro = np.linalg.norm(W - W_trunc) / np.linalg.norm(W) * 100
        # damage to the tail-encoded function: keep only its tail block
        dmg = np.linalg.norm(P_tail - band_part(spec, tail)) / tail_norm * 100
        rows.append((name, fro, dmg))

    out = pathlib.Path(__file__).resolve().parent.parent / "output"
    out.mkdir(exist_ok=True)
    with (out / "svd_band_ablation.csv").open("w") as fh:
        fh.write("band,frobenius_change_pct,tail_block_damage_pct\n")
        for name, fro, dmg in rows:
            fh.write(f"{name},{fro:.4f},{dmg:.4f}\n")

    fig, ax = new_fig(1, 3, width=WIDTH_2COL, ratio=0.36)
    a, b, c = ax[0]
    a.semilogy(k, sv, color=PAL[0], lw=1.3)
    a.axvspan(1, 3, color=PAL[1], alpha=0.15)
    a.axvspan(4, 24, color=PAL[4], alpha=0.15)
    a.axvspan(d - d // 10, d, color=PAL[2], alpha=0.18)
    a.set_xlabel("index $i$")
    a.set_ylabel("$\\sigma_i$  (log scale)")
    a.set_title("(a) One layer, three bands")
    a.set_xlim(1, d)
    for x_txt, y_txt, label, col, ha in ((1.2, 0.88, "dominant", PAL[1], "left"),
                                         (30, 0.96, "middle band", PAL[4], "left"),
                                         (d - d // 10 - 4, 0.88, "minor", PAL[2], "right")):
        a.annotate(label, xy=(x_txt, y_txt), xycoords=("data", "axes fraction"),
                   fontsize=6.5, color=col, ha=ha)

    b.plot(k, 100 * cum, color=PAL[0], lw=1.4, label="cumulative $\\sigma^2$ share")
    b.plot(k, 100 * np.cumsum(sv) / np.cumsum(sv)[-1], color=PAL[5], lw=1.2, ls="--",
           label="cumulative $\\sigma$ share")
    b.axvline(3, color=PAL[1], lw=0.8, ls=":")
    b.axvline(d - d // 10, color=PAL[2], lw=0.8, ls=":")
    b.set_xlabel("index $i$")
    b.set_ylabel("cumulative share (%)")
    b.set_title("(b) Energy hides in the top 3")
    b.set_ylim(0, 103)
    b.legend(fontsize=6.3, loc="center right")

    names = ["top-3\n(dominant)", "middle\nband", "last decile\n(minor)"]
    xs = np.arange(3)
    fro = [r[1] for r in rows]
    dmg = [r[2] for r in rows]
    c.bar(xs - 0.19, fro, width=0.38, color=PAL[1], label="change in $\\Vert W\\Vert_F$ (%)")
    c.bar(xs + 0.19, dmg, width=0.38, color=PAL[2], label="damage to the tail block (%)")
    for i, (f_, p_) in enumerate(zip(fro, dmg)):
        c.text(i - 0.19, f_ + 1.5, f"{f_:.1f}", ha="center", fontsize=6.0)
        c.text(i + 0.19, p_ + 1.5, f"{p_:.0f}", ha="center", fontsize=6.0)
    c.set_xticks(xs, names, fontsize=6.6)
    c.set_ylabel("relative change (%)")
    c.set_ylim(0, 150)
    c.set_title("(c) Two answers to 'needed?'")
    c.legend(fontsize=6.2, loc="upper left", framealpha=1.0, edgecolor="#ccc")

    save(fig, "svd-spectrum")


# --------------------------------------------------------------------------- #
# 7. Listing cards: 1200 x 630 thumbnails (1.91:1, what the cards and OG tags use)
# --------------------------------------------------------------------------- #
def _thumb(stem: str, title: str, subtitle: str, draw):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.0, 6.3), dpi=100)
    fig.patch.set_facecolor("#0e1626")
    ax = fig.add_axes([0.055, 0.16, 0.62, 0.66])
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_color("#c9d4e2")
    draw(ax)
    ax.tick_params(colors="#33414f", labelsize=9)
    ax.xaxis.label.set_color("#33414f")
    ax.yaxis.label.set_color("#33414f")
    ax.title.set_color("#33414f")
    fig.text(0.055, 0.905, title, color="white", fontsize=21, fontweight="bold",
             va="top", family=CJK_FONT)
    fig.text(0.055, 0.848, subtitle, color="#9fb4d0", fontsize=12.5, va="top", family=CJK_FONT)
    from figstyle import FIG_DIR
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{stem}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_thumbs():
    t = np.linspace(-1, 1, 600)

    def jacobi(ax):
        for (al, be) in [(0.0, 0.0), (-0.5, -0.5), (0.5, 0.5), (2.0, -0.6), (-0.6, 2.0)]:
            y = eval_jacobi(4, al, be, t)
            ax.plot(t, y / np.max(np.abs(y)), lw=2.2, label=f"$({al:g},\\,{be:g})$")
        ax.axhline(0, color="#8b8b8b", lw=0.8)
        ax.legend(fontsize=9, frameon=False, loc="lower left", ncols=2)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$P_4^{(\\alpha,\\beta)}(x)$")

    _thumb("thumb-jacobi", "Jacobi 多项式：换一组权重，就是换一个空间",
           "weight -> inner product -> basis -> space", jacobi)

    def spectrum(ax):
        _, sv, _, _, _ = _synthetic_layer()[0], None, None, None, None
        W, _, _, _, _ = _synthetic_layer()
        s = np.linalg.svd(W, compute_uv=False)
        k = np.arange(1, s.size + 1)
        ax.semilogy(k, s, color=PAL[0], lw=2.4)
        ax.axvspan(1, 3, color=PAL[1], alpha=0.18)
        ax.axvspan(s.size * 0.9, s.size, color=PAL[2], alpha=0.2)
        ax.set_xlabel("index $i$")
        ax.set_ylabel("$\\sigma_i$")
        ax.set_xlim(1, s.size)

    _thumb("thumb-fanshu", "泛函空间再向上抽象，然后呢？",
           "17 turns: topological vector spaces -> the singular-value spectrum", spectrum)

    def bands(ax):
        names = ["top-3", "middle", "last\ndecile"]
        fro, dmg = [97.1, 21.9, 2.2], [0.0, 0.0, 100.0]
        xs = np.arange(3)
        ax.bar(xs - 0.2, fro, width=0.4, color=PAL[1], label="$\\Delta\\Vert W\\Vert_F$")
        ax.bar(xs + 0.2, dmg, width=0.4, color=PAL[2], label="damage to tail block")
        ax.set_xticks(xs, names)
        ax.set_ylim(0, 118)
        ax.set_ylabel("relative change (%)")
        ax.legend(fontsize=9, frameon=False)

    _thumb("thumb-svd", "编者注：中间那一组奇异值是什么？",
           "energy says the tail is free; function says otherwise", bands)

    def spaces(ax):
        names = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
        logz = [-1165, -1164, 0, -13, -96, -41, -36]
        cols = [PAL[1], PAL[1], PAL[2], PAL[5], PAL[5], PAL[5], PAL[5]]
        ax.bar(range(7), logz, color=cols)
        ax.set_xticks(range(7), names)
        ax.set_ylabel("$\\log p(y\\mid X,\\mathcal{H}_k)$")
        ax.set_xlabel("candidate space")

    _thumb("thumb-space-search", "在候选函数空间里搜索「谁生成了数据」",
           "marginal likelihood ranks the spaces - run it in 60 s", spaces)

    def protocol(ax):
        names = ["MLP", "GP", "OLS", "Stan"]
        gpml, block = [0.64, 1.40, 2.63, 2.64], [2.28, 2.92, 2.99, 2.99]
        xs = np.arange(4)
        ax.barh(xs - 0.2, gpml, height=0.4, color=PAL[1], label="GPML split")
        ax.barh(xs + 0.2, block, height=0.4, color=PAL[0], label="block split")
        ax.set_yticks(xs, names)
        ax.invert_yaxis()
        ax.set_xlabel("test RMSE (N m)")
        ax.legend(fontsize=9, frameon=False, loc="lower right")

    _thumb("thumb-sarcos", "SARCOS inverse dynamics with Agents",
           "99.93 % of the published test set sits inside the training file", protocol)

    def stretch(ax):
        rng = np.random.default_rng(20260906)
        n = 110
        lab = np.r_[np.zeros(n, int), np.ones(n, int)]
        rel = np.r_[rng.normal(-0.6, 0.55, n), rng.normal(0.6, 0.55, n)]
        irr = rng.normal(0, 1, 2 * n)
        pts = np.c_[rel * 4.2 + irr * 0.30 * 0.55, irr * 0.30]
        for g, col in ((0, PAL[0]), (1, PAL[1])):
            m = lab == g
            ax.scatter(pts[m, 0], pts[m, 1], s=26, c=col, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
        ax.set_xlabel("$r$ (task-relevant)")
        ax.set_ylabel("$u$ (nuisance)")

    _thumb("thumb-learning", "从低维到高维的「生成」，就是学习机理",
           "dimensional modulation: stretch the relevant axis, shrink the nuisance", stretch)


    def ladder(ax):
        """Nested candidate spaces: the search is over the ladder, not the knobs."""
        import matplotlib.patches as mpatches
        rings = [(1.0, 0.62, "$\\mathcal{H}_1$ linear", PAL[5]),
                 (0.86, 0.52, "$\\mathcal{H}_2$ +trig", PAL[5]),
                 (0.70, 0.42, "$\\mathcal{H}_3$ +physics", PAL[2]),
                 (0.54, 0.32, "$\\mathcal{H}_4$ quadratic", PAL[5])]
        for w, h, lab, col in rings:
            ax.add_patch(mpatches.Ellipse((0.42, 0.42), w, h, fill=False, ec=col, lw=2.0))
            ax.text(0.42 - w / 2 + 0.03, 0.42 + h / 2 - 0.05, lab, fontsize=9.5, color=col)
        ax.text(0.98, 0.90, "smooth, infinite-dimensional", fontsize=9.5, color=PAL[1], ha="right")
        ax.text(0.98, 0.84, "$\\mathcal{H}_5$ RBF · $\\mathcal{H}_6$ ARD · $\\mathcal{H}_7$ SM",
                fontsize=9.5, color=PAL[1], ha="right")
        ax.annotate("search over spaces,\nnot over knobs", xy=(0.42, 0.42), xytext=(0.06, 0.06),
                    fontsize=10, color="#33414f",
                    arrowprops=dict(arrowstyle="->", color="#667a8c", lw=1.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _thumb("thumb-spaceladder", "下一步：在一族函数空间里搜索",
           "from model selection to space selection (dialogue)", ladder)

    def rule_family(ax):
        x = np.linspace(0, 6, 100)
        for cval, col in ((2, PAL[5]), (5, PAL[0]), (20, PAL[2])):
            ax.plot(x, cval * x, color=col, lw=2.4, label=f"$c = {cval}$")
        ax.set_xlabel("cattle (heads)")
        ax.set_ylabel("goods received")
        ax.legend(fontsize=10, frameon=False)

    _thumb("thumb-teaching", "从一头牛到泛函：一堂课的抽象阶梯",
           "one rule -> a family of rules -> a space of rules", rule_family)

    def definition_ladder(ax):
        import matplotlib.patches as mpatches
        from figstyle import CJK_FONT
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        for i, (y, lab) in enumerate(((2.0, "现实原型"), (5.0, "符号"), (8.0, "一般模式"))):
            ax.add_patch(mpatches.FancyBboxPatch((0.5, y - 1.05), 4.3, 2.1,
                         boxstyle="round,pad=0.14", fc="#f6f8fb", ec="#8fa3b8", lw=1.7))
            ax.text(2.65, y, lab, ha="center", va="center", fontsize=13,
                    fontweight="bold", family=CJK_FONT, color="#0a2540")
            if i:
                ax.annotate("", xy=(2.65, y - 1.3), xytext=(2.65, y - 2.7),
                            arrowprops=dict(arrowstyle="->", color="#667a8c", lw=2.0))
        for y, lab, col in ((1.4, "弱抽象", PAL[0]), (3.9, "构象化抽象", PAL[2]),
                            (6.4, "强抽象", PAL[1]), (8.9, "公理化抽象", PAL[4])):
            ax.text(5.5, y, lab, ha="left", va="center", fontsize=12.5, family=CJK_FONT,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.34", fc=col, ec="none"))
        ax.text(5.5, 0.25, "an explicit definition is what makes the ladder speakable",
                ha="left", va="center", fontsize=9.5, color="#9fb4d0", style="italic")

    _thumb("thumb-definition", "给一个概念下显式定义，买到了什么",
           "以「数学抽象」的课程标准定义为例", definition_ladder)



# --------------------------------------------------------------------------- #
# 8. Two figures for the teaching note: the abstraction ladder, and a pendulum
#    fitted by a GP. Both use synthetic data (labelled as such in the caption).
# --------------------------------------------------------------------------- #
def _gp(X, y, xs, kfun, noise=0.05):
    """Posterior mean + sd of a zero-mean GP with fixed hyper-parameters."""
    K = kfun(X, X) + (noise ** 2 + 1e-9) * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = kfun(xs, X) @ alpha
    v = np.diag(kfun(xs, xs)) - np.sum((np.linalg.solve(L, kfun(xs, X).T)) ** 2, axis=0)
    return mu, np.sqrt(np.clip(v, 0, None))


def fig_abstraction_ladder():
    """one rule -> a family of rules -> a space of rules."""
    rng = np.random.default_rng(2026)
    x = np.linspace(0, 6, 7)
    fig, ax = new_fig(1, 3, width=WIDTH_2COL, ratio=0.36)
    a, b, c = ax[0]

    a.plot(x, 5 * x, color=PAL[0], lw=1.8, label="$y = 5x$")
    a.plot(x, 5 * x + rng.normal(0, 0.35, x.size), "o", color=PAL[1], ms=5,
           mec="white", mew=0.5, label="recorded trades")
    a.set_xlabel("cattle (heads)")
    a.set_ylabel("sheep (heads)")
    a.set_title("(a) one rule the tribe kept")
    a.legend(fontsize=6.4, loc="upper left")

    for cval, col in zip((2, 5, 20), (PAL[5], PAL[0], PAL[2])):
        b.plot(x, cval * x, color=col, lw=1.6, label=f"$y = {cval}x$")
    b.set_xlabel("cattle (heads)")
    b.set_ylabel("goods received")
    b.set_title("(b) generalise: one rule per good")
    b.legend(fontsize=6.4, loc="upper left")

    X = np.sort(rng.uniform(0.5, 5.5, 14))[:, None]
    y = 5 * X[:, 0] + rng.normal(0, 0.9, X.size)
    xs = np.linspace(0, 7, 200)[:, None]
    ell, sf = 2.0, 12.0
    kern = lambda A, B: sf ** 2 * np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) / ell) ** 2).sum(-1)
    mu, sd = _gp(X, y, xs, kern, noise=0.9)
    c.plot(xs, 5 * xs[:, 0], ls="--", color="#111", lw=1.4, label="the rule that worked")
    c.plot(xs, mu, color=PAL[0], lw=1.7, label="posterior mean")
    c.fill_between(xs[:, 0], mu - 1.96 * sd, mu + 1.96 * sd, color=PAL[0], alpha=0.18,
                   label="95 % interval")
    c.plot(X, y, "o", color=PAL[1], ms=4.5, mec="white", mew=0.4, label="market data")
    c.set_xlabel("cattle (heads)")
    c.set_ylabel("sheep (heads)")
    c.set_title("(c) search: pricing as GP regression")
    c.legend(fontsize=6.0, loc="upper left")
    save(fig, "abstraction-ladder")


def fig_pendulum_gp():
    """A shaky phone clip of a pendulum, and what two kernels do with it."""
    rng = np.random.default_rng(11)
    t = np.linspace(0, 6, 60)
    tt = np.linspace(0, 8, 400)
    truth = lambda z: 1.1 * np.exp(-z / 5.0) * np.cos(2 * np.pi * 0.8 * z)
    shaky = truth(t) + rng.normal(0, 0.05, t.size)        # hand-shake noise
    Y = shaky - shaky.mean()

    se = lambda A, B: np.exp(-0.5 * ((A[:, None] - B[None, :]) / 0.6) ** 2)
    per = lambda A, B: np.exp(-2 * np.sin(np.pi * np.abs(A[:, None] - B[None, :]) / 1.25) ** 2 / 0.2)
    prod = lambda A, B: per(A, B) * se(A, B)

    fig, ax = new_fig(1, 3, width=WIDTH_2COL, ratio=0.36)
    a, b, c = ax[0]

    a.plot(tt, truth(tt), color="#111", lw=1.3, ls="--", label="true angle")
    a.plot(t, Y, "o", ms=4.0, color=PAL[0], mec="white", mew=0.4, label="extracted frames")
    a.set_xlabel("time $t$ (s)")
    a.set_ylabel("$\\theta$ (rad)")
    a.set_title("(a) frames from a shaky clip")
    a.legend(fontsize=6.3, loc="upper right")

    for kern, col, title, lab in ((se, PAL[0], "(b) smooth kernel: blank beyond $t=6$",
                                   "squared-exponential"),
                                  (prod, PAL[2], "(c) periodic kernel keeps the beat",
                                   "periodic $\\times$ squared-exponential")):
        panel = b if kern is se else c
        mu, sd = _gp(t[:, None], Y, tt[:, None], lambda A, B: kern(A[:, 0], B[:, 0]), noise=0.05)
        panel.plot(tt, truth(tt), color="#111", ls="--", lw=1.1, label="true angle")
        panel.plot(tt, mu, color=col, lw=1.6, label=lab)
        panel.fill_between(tt, mu - 1.96 * sd, mu + 1.96 * sd, color=col, alpha=0.16)
        panel.plot(t, Y, "o", ms=3.2, color="#9a9a9a", mec="none")
        panel.set_xlabel("time $t$ (s)")
        panel.set_ylabel("$\\theta$ (rad)")
        panel.set_title(title)
        panel.set_ylim(-1.45, 1.45)
        panel.set_xlim(0, 8)
        panel.legend(fontsize=6.1, loc="lower left", framealpha=0.95)
    b.axvline(6.0, color="#666", lw=0.8, ls=":")
    c.axvline(6.0, color="#666", lw=0.8, ls=":")
    b.text(6.1, 1.28, "data ends", fontsize=6.0, color="#555")
    c.text(6.1, 1.28, "data ends", fontsize=6.0, color="#555")
    save(fig, "pendulum-gp")


# --------------------------------------------------------------------------- #
# 9. Concept map for the "what is an explicit definition for" note
# --------------------------------------------------------------------------- #
def fig_definition_map():
    """A map of the note's own argument — not measured data.

    (a) the four things an explicit definition buys;
    (b) the three cognitive stages of mathematical abstraction, with the four
        abstraction types placed by how far up the ladder they take you.
    """
    import matplotlib.patches as mpatches
    from figstyle import CJK_FONT

    fig, ax = new_fig(1, 2, width=WIDTH_2COL, ratio=0.50)
    a, b = ax[0]
    for panel in (a, b):
        panel.set_xlim(0, 10)
        panel.set_ylim(0, 10)
        panel.axis("off")

    def box(panel, x, y, w, h, head, sub, ec, fc="white", headcol=None):
        panel.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.14", fc=fc, ec=ec, lw=1.25))
        panel.text(x, y + (h / 4 if sub else 0), head, ha="center", va="center",
                   fontsize=7.9, fontweight="bold", family=CJK_FONT,
                   color=headcol or ec)
        if sub:
            panel.text(x, y - h / 4.6, sub, ha="center", va="center", fontsize=6.9,
                       family=CJK_FONT, color="#3a4a5a")

    # ---- (a) one definition, four jobs -------------------------------------
    box(a, 5.0, 7.6, 5.2, 1.7, "明确、显式的定义", None, PAL[0], fc="#eef4ff",
        headcol="#0a2540")
    jobs = [("划定认知边界", "内涵与外延清楚", PAL[0]),
            ("提供推理起点", "演绎有所依据", PAL[1]),
            ("知识可传递", "可交流、可检验", PAL[2]),
            ("支撑理论体系", "概念得以分层", PAL[4])]
    for x, (head, sub, col) in zip((1.42, 3.81, 6.19, 8.58), jobs):
        box(a, x, 2.75, 2.1, 1.95, head, sub, col)
        a.annotate("", xy=(x, 3.85), xytext=(5.0 + (x - 5.0) * 0.28, 6.68),
                   arrowprops=dict(arrowstyle="-", color=col, lw=1.0))
    a.text(5.0, 5.15, "一条定义同时兑现四件事", ha="center", fontsize=7.4,
           family=CJK_FONT, color="#52606d")
    a.set_title("(a) 定义买到了什么", fontsize=9.5, family=CJK_FONT, pad=4)

    # ---- (b) the ladder ----------------------------------------------------
    stages = [(2.0, "现实原型", "具体事物与背景"),
              (5.0, "符号", "用数学语言表征"),
              (8.0, "一般模式与结构", "规律本身成为对象")]
    for y, head, sub in stages:
        box(b, 3.4, y, 4.6, 1.9, head, sub, "#8fa3b8", fc="#f6f8fb", headcol="#0a2540")
    for y0, y1 in ((3.0, 4.0), (6.0, 7.0)):
        b.annotate("", xy=(3.4, y1), xytext=(3.4, y0),
                   arrowprops=dict(arrowstyle="->", color="#667a8c", lw=1.5))
    b.text(0.55, 5.0, "抽\n象\n结\n构\n越\n来\n越\n少", rotation=0, ha="center", va="center",
           fontsize=7.2, family=CJK_FONT, color="#52606d", linespacing=1.35)
    types = [(1.45, "弱抽象", "舍去细节，保留共性", PAL[0]),
             (3.95, "构象化抽象", "构造新对象承载性质", PAL[2]),
             (6.45, "强抽象", "加入新关系与新公理", PAL[1]),
             (8.55, "公理化抽象", "整个体系重新奠基", PAL[4])]
    for y, head, sub, col in types:
        b.text(6.25, y + 0.28, head, ha="left", va="center", fontsize=8.0, family=CJK_FONT,
               color="white", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.28", fc=col, ec="none"))
        b.text(6.25, y - 0.52, sub, ha="left", va="center", fontsize=6.5,
               family=CJK_FONT, color=col)
        b.plot([5.85, 6.12], [y + 0.28, y + 0.28], color=col, lw=1.0)
    b.set_title("(b) 数学抽象的层次与四种抽象", fontsize=9.5, family=CJK_FONT, pad=4)
    save(fig, "definition-map")

def main(select: str = "") -> None:
    figures = {
        "jacobi-basis": fig_jacobi_basis,
        "jacobi-weights-nodes": fig_jacobi_weights,
        "representation-stretch": fig_representation_stretch,
        "sarcos-protocol": fig_sarcos_protocol,
        "svd-spectrum": fig_svd_spectrum,
        "abstraction-ladder": fig_abstraction_ladder,
        "pendulum-gp": fig_pendulum_gp,
        "definition-map": fig_definition_map,
        "thumbs": fig_thumbs,
        "site-cover": fig_cover,
    }
    for stem, fn in figures.items():
        if select and select not in stem:
            continue
        print(f"[fig] {stem}")
        fn()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
