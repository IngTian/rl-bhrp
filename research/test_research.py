"""Checks for the research lane. Run: uv run python -m research.test_research"""
import numpy as np

from bhrp import bhrp
from data import SECTOR_OF
from research.evaluate import _drawdown, mandate_deviation, sharpe_test, simulate, stationary_indices

rng = np.random.default_rng(0)


def test_ledger_matches_dollar_positions():
    D, N = 60, 4
    R = rng.normal(0.0005, 0.01, (D, N))
    rf = np.full(D, 1e-4)
    W = rng.dirichlet(np.ones(N), 3)
    ex, cost = np.array([2, 20, 40]), 7e-4
    nav, log = simulate(W, ex, R, rf, cost, start_pos=0, end_pos=D - 1)
    pos, cash = np.zeros(N), 1.0  # independent dollar ledger
    for d in range(D):
        if d > 0:
            pos, cash = pos * (1 + R[d]), cash * (1 + rf[d - 1])
        if d in ex:
            A = pos.sum() + cash
            w = W[list(ex).index(d)]
            k = 1.0
            for _ in range(100):  # fixed point k = 1 - cost * ||k w A - pos|| / A
                k = 1 - cost * np.abs(k * w * A - pos).sum() / A
            C = cost * np.abs(k * w * A - pos).sum()
            pos, cash = k * w * A, A - k * A - C
            assert abs(cash) < 1e-9
        if d == D - 1:
            assert np.isclose(nav[-1], pos.sum() + cash, rtol=1e-10)
    assert np.isclose(nav[1], 1 + rf[0])  # in cash until the first execution
    for t in log:
        assert np.isclose(t["x"].sum(), 1) and np.isclose(t["cost"], cost * t["turnover"])


def test_partial_investment_keeps_cash():
    D, N = 10, 3
    R = rng.normal(0, 0.01, (D, N))
    rf = np.full(D, 2e-4)
    w = np.array([0.5, 0.3, 0.15])  # 5% cash by design
    nav, log = simulate(w[None], np.array([0]), R, rf, 1e-3, 0, D - 1)
    x = log[0]["x"]
    assert np.allclose(x, w)  # post-trade risky weights = target, the rest is cash
    pos, cash = x * nav[0], (1 - w.sum()) * nav[0]
    for d in range(1, D):
        pos, cash = pos * (1 + R[d]), cash * (1 + rf[d - 1])
    assert np.isclose(nav[-1], pos.sum() + cash, rtol=1e-12)


def test_zero_cost_rebalance_equals_weighted_returns():
    D, N = 30, 3
    R = rng.normal(0, 0.01, (D, N))
    w = np.array([0.5, 0.3, 0.2])
    ex = np.arange(0, D - 1)  # rebalance every day at zero cost -> NAV = prod(1 + w'R)
    nav, _ = simulate(np.tile(w, (len(ex), 1)), ex, R, np.zeros(D), 0.0, 0, D - 1)
    assert np.allclose(nav, np.cumprod(np.r_[1, 1 + R[1:] @ w]))


def test_drawdown_counts_initial_capital():
    dd, length = _drawdown(np.array([1.0, 0.9, 0.95, 1.01, 0.8]))
    assert np.isclose(dd, 1 - 0.8 / 1.01) and length is None
    dd, length = _drawdown(np.array([1.0, 0.8, 1.0, 1.1]))
    assert np.isclose(dd, 0.2) and length == 2


def test_bhrp_meets_its_own_mandate():
    sec = SECTOR_OF
    F = np.c_[rng.uniform(0.5, 1.5, len(sec)), 0.8 * np.eye(sec.max() + 1)[sec]]
    S = (F @ F.T + np.diag(rng.uniform(0.5, 2, len(sec)))) * 1e-3
    act = np.ones(len(sec), bool)
    vs, vi, breach = mandate_deviation(bhrp(S, sec), S, act)
    assert vs < 1e-12 and vi < 1e-12 and not breach
    vs, vi, breach = mandate_deviation(np.full(len(sec), 1 / len(sec)), S, act)
    assert breach and vs + vi > 0


def test_bootstrap_and_sharpe_test():
    idx = stationary_indices(120, 2000, 6)
    assert idx.min() >= 0 and idx.max() < 120
    runs = np.diff(idx, axis=1) == 1
    assert abs(1 / (1 - runs.mean()) - 6) < 0.6  # mean block length ~ 6
    e = rng.normal(0.008, 0.04, 240)
    assert sharpe_test(e, e.copy(), reps=300)[1] == 1.0
    assert sharpe_test(e, rng.normal(0.008, 0.04, 240), reps=300)[1] > 0.05  # same SR: no rejection
    assert sharpe_test(e + 0.02, e - 0.02, reps=300)[1] < 0.01


def test_baselines():
    from research import baselines as bl
    N, T = len(SECTOR_OF), 1
    var = rng.uniform(0.8, 1.2, N) * 1e-3
    P = {"act": np.ones((T, N), bool), "S": np.diag(var)[None], "mu": np.zeros((T, N)), "w_bhrp": np.zeros((T, N))}
    w = bl.minvar(P, 0)
    assert np.allclose(w, (1 / var) / (1 / var).sum(), atol=1e-6)  # caps not binding
    assert np.allclose(bl.mean_variance(P, 0, kappa=1e6), 1 / N, atol=1e-6)
    h = bl.hrp(P, 0)
    assert np.isclose(h.sum(), 1) and (h > 0).all()
    v = np.zeros(N)
    v[:3] = [0.5, 0.3, 0.2]  # breaks both caps (all in sector 0)
    x, bound = bl.project(P, 0, v)
    assert bound and bl.feasible(x, SECTOR_OF, 1e-7)
    assert bl.project(P, 0, h)[1] is False
    mom = rng.normal(size=(T, N))
    m = bl.momentum(P, 0, mom=mom)
    assert np.isclose(m.sum(), 1) and set(np.flatnonzero(m)) == set(np.argsort(-mom[0])[:16])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
