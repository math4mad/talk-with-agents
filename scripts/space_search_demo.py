#!/usr/bin/env python3
"""Space search demo: can the marginal likelihood find the space that generated data?

The dialogue in ``coding/repos/sarcos-space-search.qmd`` proposes, for the SARCOS
inverse-dynamics problem, to stop committing to one hypothesis space and instead
score a *family* of candidate spaces

    H1 linear            span{1, q, dq, ddq}
    H2 linear + trig     span{1, q, dq, ddq, sin q, cos q}
    H3 linear + physics  span{..., q*ddq, q*dq, dq*ddq}     (Coriolis-like terms)
    H4 full quadratic    all degree-2 monomials in (q, dq, ddq)
    H5 RBF (isotropic)   infinite-dimensional, smooth
    H6 RBF-ARD           one length-scale per input
    H7 spectral mixture  sum of Gaussian spectral components (Wilson & Adams 2013)

Each space is a GP prior with kernel ``k(x,x') = phi(x)^T diag(theta) phi(x')``
(explicit bases) or a classic stationary kernel, hyper-parameters are fitted by
maximising the log marginal likelihood, and every space is then scored three ways:

* ``logZ``   — log marginal likelihood (Occam factor built in),
* ``loo``    — closed-form GP leave-one-out RMSE,
* ``test``   — RMSE on a fresh, leakage-free sample,
* ``dof``    — effective degrees of freedom ``tr(K_y^{-1} K)``.

Run::

    python3 scripts/space_search_demo.py            # ~10 s, CPU only

Outputs ``assets/figures/space-search.{png,pdf}`` and ``output/space_search_results.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import PAL, WIDTH_2COL, new_fig, save  # noqa: E402

RNG = np.random.default_rng(7)
NOISE = 0.06          # observation noise sd (N m in the SARCOS analogy)
N_TRAIN, N_TEST = 320, 2000


# --------------------------------------------------------------------------- #
# the "physical" data-generating model:  torque ~ inertia*ddq + Coriolis + gravity
# --------------------------------------------------------------------------- #
def truth(x: np.ndarray) -> np.ndarray:
    q, dq, ddq = x[:, 0], x[:, 1], x[:, 2]
    return 1.2 * ddq + 0.9 * q * ddq + 0.7 * np.sin(q) + 0.25 * dq


def sample(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-3, 3, n),
                            rng.uniform(-4, 4, n)])


# --------------------------------------------------------------------------- #
# candidate feature maps (the "spaces")
# --------------------------------------------------------------------------- #
def f_linear(x):
    return np.column_stack([np.ones(len(x)), x])


def f_trig(x):
    return np.column_stack([f_linear(x), np.sin(x[:, 0]), np.cos(x[:, 0])])


def f_physics(x):
    q, dq, ddq = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([f_trig(x), q * ddq, q * dq, dq * ddq])


def f_quad(x):
    q, dq, ddq = x[:, 0], x[:, 1], x[:, 2]
    extra = [q * q, dq * dq, ddq * ddq, q * ddq, q * dq, dq * ddq]
    return np.column_stack([f_trig(x)] + extra)


SPACES = {
    "H1 linear": ("basis", f_linear),
    "H2 +trig": ("basis", f_trig),
    "H3 +physics": ("basis", f_physics),
    "H4 quadratic": ("basis", f_quad),
    "H5 RBF": ("rbf", None),
    "H6 RBF-ARD": ("ard", None),
    "H7 spec-mix": ("sm", None),
}


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def gram(kind: str, a: np.ndarray, b: np.ndarray, theta: np.ndarray, space: str) -> np.ndarray:
    """Kernel of candidate space ``space`` evaluated on rows of ``a`` x rows of ``b``."""
    if kind == "basis":                                      # finite basis, ARD over features
        phi = SPACES[space][1](a) * theta
        psi = SPACES[space][1](b) * theta
        return phi @ psi.T
    tau = a[:, None, :] - b[None, :, :]
    if kind == "rbf":                                        # one shared length-scale
        return theta[1] ** 2 * np.exp(-0.5 * ((tau / theta[0]) ** 2).sum(-1))
    if kind == "ard":                                        # one length-scale per input
        return theta[-1] ** 2 * np.exp(-0.5 * ((tau / theta[:-1]) ** 2).sum(-1))
    if kind == "sm":                                         # Wilson & Adams (2013), Q = 2
        w, mu, v = theta[:2], theta[2:8].reshape(2, 3), theta[8:14].reshape(2, 3)
        k = np.zeros(tau.shape[:2])
        for j in range(2):
            quad = (tau ** 2) @ v[j]
            k += w[j] ** 2 * np.exp(-2 * np.pi ** 2 * quad) * np.cos(2 * np.pi * (tau @ mu[j]))
        return k
    raise ValueError(kind)


def n_params(kind: str, space: str, d: int) -> int:
    if kind == "basis":
        return SPACES[space][1](np.zeros((1, d))).shape[1]
    return {"rbf": 2, "ard": d + 1, "sm": 14}[kind]


def neg_logml(params: np.ndarray, X: np.ndarray, y: np.ndarray, kind: str, space: str) -> float:
    theta = np.exp(params[:-1])
    noise = np.exp(params[-1])
    K = gram(kind, X, X, theta, space) + (noise ** 2 + 1e-8) * np.eye(len(X))
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        return 1e6
    return 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(y) * np.log(2 * np.pi)


def fit(X: np.ndarray, y: np.ndarray, kind: str, space: str) -> dict:
    """Maximise the log marginal likelihood (multi-start L-BFGS-B), then score the space."""
    d = X.shape[1]
    p0 = np.r_[np.zeros(n_params(kind, space, d)) - 1.0, np.log(NOISE)]
    best = None
    for seed in range(3):                                      # logZ is not convex
        rng = np.random.default_rng(seed)
        start = p0 + rng.normal(0, 0.6, p0.shape)
        r = minimize(neg_logml, start, args=(X, y, kind, space), method="L-BFGS-B",
                     options=dict(maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    theta, noise = np.exp(best.x[:-1]), np.exp(best.x[-1])
    K = gram(kind, X, X, theta, space)
    K_y = K + (noise ** 2 + 1e-9) * np.eye(len(X))
    K_inv = np.linalg.inv(K_y)
    loo_mu = y - (K_inv @ y) / np.diag(K_inv)
    return dict(theta=theta, noise=noise, kind=kind, space=space,
                logml=-best.fun,
                loo_rmse=float(np.sqrt(np.mean((y - loo_mu) ** 2))),
                dof=float(np.trace(K_inv @ K)))


def predict(model: dict, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    Ks = gram(model["kind"], Xte, Xtr, model["theta"], model["space"])
    K_inv = np.linalg.inv(gram(model["kind"], Xtr, Xtr, model["theta"], model["space"])
                          + (model["noise"] ** 2 + 1e-9) * np.eye(len(Xtr)))
    return Ks @ (K_inv @ ytr)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, dict]:
    Xtr, Xte = sample(N_TRAIN, RNG), sample(N_TEST, np.random.default_rng(11))
    ytr = truth(Xtr) + RNG.normal(0, NOISE, N_TRAIN)
    yte = truth(Xte)
    rows, models = [], {}
    for space, (kind, _) in SPACES.items():
        m = fit(Xtr, ytr, kind, space)
        pred = predict(m, Xtr, ytr, Xte)
        nb = SPACES[space][1](Xtr[:1]).shape[1] if kind == "basis" else np.inf
        rows.append(dict(space=space, logml=m["logml"], loo_rmse=m["loo_rmse"],
                         test_rmse=float(np.sqrt(np.mean((pred - yte) ** 2))),
                         dof=m["dof"], n_basis=nb))
        models[space] = (m, pred)
        print(f"  {space:14s} logZ={m['logml']:9.2f}  LOO={rows[-1]['loo_rmse']:.4f}  "
              f"test={rows[-1]['test_rmse']:.4f}  dof={m['dof']:6.2f}")
    return rows, Xtr, ytr, models


def figure(rows: list[dict], Xtr: np.ndarray, ytr: np.ndarray, models: dict) -> None:
    """Three panels: evidence, honest error, and what the winner actually fits."""
    names = [r["space"] for r in rows]
    logml = np.array([r["logml"] for r in rows])
    test = np.array([r["test_rmse"] for r in rows])
    dof = np.array([r["dof"] for r in rows])
    best, worst = names[int(np.argmax(logml))], names[int(np.argmin(logml))]
    cols = [PAL[2] if n == best else (PAL[1] if n == worst else PAL[5]) for n in names]
    idx = np.arange(len(rows))

    fig, ax = new_fig(1, 3, width=WIDTH_2COL, ratio=0.36)
    a, b, c = ax[0]

    a.bar(idx, logml - logml.max(), color=cols)
    a.set_xticks(idx, names, rotation=38, ha="right", fontsize=6.4)
    a.set_ylabel("$\\log p(y\\,|\\,X,\\mathcal{H}_k)$  (relative)")
    a.set_title("(a) Evidence orders the spaces")
    a.tick_params(axis="x", pad=1)
    for i, v in enumerate(logml - logml.max()):
        inside = v < -400
        a.text(i, v - (30 if not inside else -60), f"{v:,.0f}", ha="center",
               va="top" if not inside else "bottom", fontsize=6.0,
               color="white" if inside else "#333")
    a.set_ylim((logml - logml.max()).min() - 240, 40)

    b.barh(idx, np.maximum(test, 1e-4), color=cols)
    b.set_yticks(idx, [f"{n} ({d:.1f} dof)" for n, d in zip(names, dof)], fontsize=6.4)
    b.invert_yaxis()
    b.set_xscale("log")
    b.set_xlabel("honest test RMSE  (noise floor $\\sigma = 0.06$)")
    b.set_title("(b) Honest error, same order")
    for i, v in enumerate(test):
        b.text(v * 1.3, i, f"{v:.4f}", va="center", fontsize=6.0)
    b.set_xlim(2e-3, 40)

    t = np.linspace(-2, 2, 200)
    slice_q = np.column_stack([t, np.zeros_like(t), np.full_like(t, 1.5)])
    c.scatter(Xtr[:, 0], ytr, s=5, color="#b4b4b4", alpha=0.6, label="training rows")
    c.plot(t, truth(slice_q), color="#111", lw=2.4, ls="--", label="data-generating model")
    for name, col in ((worst, PAL[1]), (best, PAL[2])):
        c.plot(t, predict(models[name][0], Xtr, ytr, slice_q), lw=1.5, color=col, label=name)
    c.set_xlabel("$q$   ($\\dot q = 0$, $\\ddot q = 1.5$)")
    c.set_ylabel("torque (N m)")
    c.set_title("(c) The winner fits the curvature")
    c.legend(fontsize=6.2, loc="lower right", framealpha=1.0, edgecolor="#ccc")
    save(fig, "space-search")


def write_csv(rows: list[dict]) -> Path:
    out = Path(__file__).resolve().parent.parent / "output"
    out.mkdir(exist_ok=True)
    import csv

    path = out / "space_search_results.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(path.parents[1])}")
    return path


if __name__ == "__main__":
    rows, Xtr, ytr, models = run()
    write_csv(rows)
    figure(rows, Xtr, ytr, models)
