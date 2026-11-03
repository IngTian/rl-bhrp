"""Competitive baselines (issue #3 Sec. 11.1, the subset that needs no external code). Each maps the panel at
decision t to target weights over all N names (0 = ineligible); `project` then holds every method, RL included, to
the same capital mandate."""
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

from bhrp import risk_budget
from data import SECTOR_OF

NAME_CAP, SECTOR_CAP = 0.10, 0.35  # common capital mandate (Sec. 6)
MV_GAMMA = 3.0                     # same risk aversion as the evaluation utility


def _active(P, t):
    a = P["act"][t]
    return a, P["S"][t][np.ix_(a, a)], SECTOR_OF[a]


def _full(a, w):
    out = np.zeros(len(a))
    out[a] = w
    return out


def _solve(obj, grad, sec, w0):
    """min obj(w) s.t. sum w = 1, 0 <= w <= NAME_CAP, sector sums <= SECTOR_CAP (SLSQP; objective pre-scaled)."""
    A = np.eye(sec.max() + 1)[sec].T
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": lambda w: SECTOR_CAP - A @ w, "jac": lambda w: -A}]
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=[(0, NAME_CAP)] * len(w0), constraints=cons,
                   options=dict(ftol=1e-12, maxiter=1000))
    w = np.clip(res.x, 0, None)
    w /= w.sum()
    # status 8 = line search stalled at round-off; accept only a feasible point no worse than the start
    assert res.success or (res.status == 8 and feasible(w, sec, 1e-7) and obj(w) <= obj(w0) + 1e-12), res.message
    return w


def feasible(w, sec, tol=1e-9):
    return w.max() <= NAME_CAP + tol and np.bincount(sec, w).max() <= SECTOR_CAP + tol


def project(P, t, w):
    """Mandate projection argmin (x - w)' S (x - w) (the Sec. 6 executable reference without cost/participation
    limits). Identity when w already satisfies the mandate. Returns (weights, whether the projection bound)."""
    a, S, sec = _active(P, t)
    v = w[a]
    if feasible(v, sec):
        return w, False
    Sn = S / np.diag(S).mean()
    x = _solve(lambda x: (x - v) @ Sn @ (x - v), lambda x: 2 * Sn @ (x - v), sec, np.full(len(v), 1 / len(v)))
    return _full(a, x), True


def ew(P, t, **_):
    a = P["act"][t]
    return a / a.sum()


def sector_ew(P, t, **_):
    a = P["act"][t]
    n = np.bincount(SECTOR_OF, a)
    return a / n[SECTOR_OF] / len(n)


def erc(P, t, **_):
    a, S, _ = _active(P, t)
    return _full(a, risk_budget(S, np.full(a.sum(), 1 / a.sum())))


def bhrp(P, t, **_):
    return P["w_bhrp"][t]


def hrp(P, t, **_):
    """Lopez de Prado (2016): correlation distance, Euclidean distance between its columns, single linkage,
    quasi-diagonal order, recursive bisection with inverse-variance cluster variances."""
    a, S, _ = _active(P, t)
    sd = np.sqrt(np.diag(S))
    d = np.sqrt(np.clip((1 - S / np.outer(sd, sd)) / 2, 0, None))
    order = leaves_list(linkage(pdist(d), "single"))
    w = np.ones(len(sd))

    def cvar(c):
        iv = 1 / np.diag(S)[c]
        iv /= iv.sum()
        return iv @ S[np.ix_(c, c)] @ iv

    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) > 1:
                l, r = c[: len(c) // 2], c[len(c) // 2:]
                al = 1 - cvar(l) / (cvar(l) + cvar(r))
                w[l] *= al
                w[r] *= 1 - al
                nxt += [l, r]
        clusters = nxt
    return _full(a, w / w.sum())


def minvar(P, t, **_):
    a, S, sec = _active(P, t)
    Sn = S / np.diag(S).mean()
    return _full(a, _solve(lambda w: w @ Sn @ w, lambda w: 2 * Sn @ w, sec, np.full(a.sum(), 1 / a.sum())))


def mean_variance(P, t, kappa, **_):
    """max mu'w - (gamma/2) w'Sw - (kappa/2) ||w - 1/n||^2 with the posterior mean mu_hat (same forecast the RL state
    sees). The L2 anchor regularises concentration; kappa is chosen on each fold's validation window."""
    a, S, sec = _active(P, t)
    mu, n = P["mu"][t][a], a.sum()
    s = np.diag(S).mean()
    f = lambda w: -(mu @ w - MV_GAMMA / 2 * w @ S @ w - kappa / 2 * ((w - 1 / n) ** 2).sum()) / s
    g = lambda w: -(mu - MV_GAMMA * S @ w - kappa * (w - 1 / n)) / s
    return _full(a, _solve(f, g, sec, np.full(n, 1 / n)))


def momentum(P, t, mom, **_):
    """12-1 momentum, equal weight in the top third of eligible names (prespecified, no tuning)."""
    a = P["act"][t]
    m = np.where(a, mom[t], -np.inf)
    top = np.argsort(-m)[: int(np.ceil(a.sum() / 3))]
    w = np.zeros(len(a))
    w[top] = 1 / len(top)
    return w


METHODS = dict(ew=ew, sector_ew=sector_ew, erc=erc, bhrp=bhrp, hrp=hrp, minvar=minvar, mean_variance=mean_variance,
               momentum=momentum)
