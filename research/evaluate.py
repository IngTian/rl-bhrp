"""Research-lane evaluation (issue #3): drift-aware daily NAV ledger (Sec. 6), Sec. 15.2 metrics, paired
stationary-bootstrap contrasts and a studentized Sharpe-difference test (Sec. 15.4). Reproduction results keep the
paper's conventions in run.py; nothing here changes them."""
import numpy as np
import pandas as pd

from data import SECTOR_OF

GAMMA = 3.0      # local quadratic utility u(e) = e - GAMMA e^2 / 2 (Sec. 10.4)
TOL_REL = 0.25   # mandate band: risk share within +-25% of its target (sector 1/G, asset 1/(G n_g)); declared, not tuned


def simulate(targets, exec_pos, R, rf, cost, start_pos, end_pos):
    """Daily NAV with drifted holdings. targets (K, N) are executed at the close of daily rows exec_pos (K,), each
    against the holdings drifted to that close: x = kappa w with kappa + cost * ||kappa w - h||_1 = 1, i.e. costs are
    paid from NAV and post-trade risky weights equal w (1 - 1'w stays in cash; 0 for fully invested targets).
    R (D, N): close-to-close simple returns; rf (D,): cash
    return over (d, d+1]. Starts with capital 1.0 in cash at the close of start_pos. Returns post-trade NAV for rows
    start_pos..end_pos and a trade log."""
    exec_at = {int(p): k for k, p in enumerate(exec_pos)}
    assert min(exec_at) >= start_pos
    N = R.shape[1]
    h, cash, nav = np.zeros(N), 1.0, 1.0
    navs, log = [], []
    for d in range(start_pos, end_pos + 1):
        if d > start_pos:
            r = R[d]
            held = h != 0
            assert np.isfinite(r[held]).all(), f"held asset without a return on row {d}"
            g = np.where(held, h * (1 + np.nan_to_num(r)), 0.0)
            c = cash * (1 + rf[d - 1])
            growth = g.sum() + c
            nav *= growth
            h, cash = g / growth, c / growth
        if d in exec_at:
            w = targets[exec_at[d]]
            lo, hi = 0.0, 1.0  # k = 1 - C(k): post-trade risky holdings are w of post-trade NAV
            for _ in range(60):
                k = (lo + hi) / 2
                lo, hi = (k, hi) if k + cost * np.abs(k * w - h).sum() < 1 else (lo, k)
            x = lo * w
            turn = np.abs(x - h).sum()
            C = cost * turn
            nav *= 1 - C
            h, cash = x / (1 - C), (1 - x.sum() - C) / (1 - C)  # residual cash c0 = 1 - 1'x - C (Sec. 6)
            log.append(dict(pos=d, turnover=turn, cost=C, x=h.copy()))
        navs.append(nav)
    return np.array(navs), log


def _drawdown(nav):
    """Max drawdown of a NAV path whose first entry is the initial capital, and its length in trading days from
    peak to recovery (None if never recovered, i.e. censored)."""
    peak = np.maximum.accumulate(nav)
    dd = 1 - nav / peak
    i = int(dd.argmax())
    p = int(np.flatnonzero(nav[: i + 1] == peak[i])[0])
    rec = np.flatnonzero(nav[i:] >= peak[i])
    return dd[i], (i + int(rec[0]) - p) if len(rec) else None


def mandate_deviation(x, S, act):
    """Sec. 15.1 fixed-mandate excess deviation at executed holdings x with the dated covariance S: sector risk shares
    vs 1/G and within-sector contrasts (asset share - sector share / n_g) vs 0, bands TOL_REL * target, weights
    1/G per sector and 1/(G n_g) per asset. Returns (sector part, within part, any node outside its band)."""
    x, S, sec = x[act], S[np.ix_(act, act)], SECTOR_OF[act]
    G = sec.max() + 1
    rc = x * (S @ x) / (x @ S @ x)
    n = np.bincount(sec, minlength=G)
    share = np.bincount(sec, rc, G)
    dev_s = np.abs(share - 1 / G) - TOL_REL / G
    dev_i = np.abs(rc - share[sec] / n[sec]) - TOL_REL / (G * n[sec])
    vs = (np.maximum(dev_s, 0) / G).sum()
    vi = (np.maximum(dev_i, 0) / (G * n[sec])).sum()
    return vs, vi, bool((dev_s > 1e-12).any() or (dev_i > 1e-12).any())


def metrics(nav, days, bench_nav, rf, log, month_end, panels=None, act=None):
    """A subset of Sec. 15.2 on complete months (no factor regressions, liquidation cost or execution/capacity rows).
    nav/bench_nav: post-trade daily NAV from simulate() on `days`, which start at a month-end with initial capital 1.0
    (so initial trading costs count); rf: daily cash return on `days`; month_end: bool mask of month-end rows (row 0
    included); panels: {label: list of covariances, one per trade in log} for the mandate metrics, each graded
    identically for every method (Sec. 15.1 keeps point-estimate and realized-proxy panels separate); act: eligibility
    per trade."""
    days = pd.DatetimeIndex(days)
    me = np.flatnonzero(month_end)
    assert me[0] == 0
    path = lambda x: np.r_[1.0, x[me[1:]]]  # month-end NAV, starting from the initial capital
    r, b = np.diff(path(nav)) / path(nav)[:-1], np.diff(path(bench_nav)) / path(bench_nav)[:-1]
    rf_m = np.array([np.prod(1 + rf[a:z]) - 1 for a, z in zip(me[:-1], me[1:])])
    e, eb = r - rf_m, b - rf_m
    years = (days[me[-1]] - days[0]).days / 365.25
    A = nav[me[-1]]
    daily = np.r_[1.0, nav[: me[-1] + 1]]
    dd, rec = _drawdown(daily)
    dloss = -(daily[1:] / daily[:-1] - 1)
    var_d = np.quantile(dloss, 0.95, method="inverted_cdf")
    var_m = np.quantile(-r, 0.95, method="inverted_cdf")
    a = r - b
    te = np.sqrt(12) * a.std(ddof=1)
    beta = np.cov(e, eb, ddof=1)[0, 1] / eb.var(ddof=1)
    turn = np.array([t["turnover"] for t in log])
    costs = np.array([t["cost"] for t in log])
    xs = np.array([t["x"] / t["x"].sum() for t in log])
    sec = SECTOR_OF if xs.shape[1] == len(SECTOR_OF) else np.arange(xs.shape[1])  # ETF universe: one ETF per sector
    sec_w = np.stack([np.bincount(sec, x, sec.max() + 1) for x in xs])
    down = np.sqrt(np.mean(np.minimum(e, 0) ** 2))
    out = {
        "Months": len(r),
        "CAGR": A ** (1 / years) - 1,
        "Log growth (12 x mean)": 12 * np.mean(np.log1p(r)),
        "Excess return (12 x mean)": 12 * e.mean(),
        "Volatility": np.sqrt(12) * r.std(ddof=1),
        "Excess Sharpe": np.sqrt(12) * e.mean() / e.std(ddof=1),
        "Sortino (MAR = rf)": np.sqrt(12) * e.mean() / down if down > 0 else np.nan,
        "Utility (12 x mean u_3)": 12 * np.mean(utility(e)),
        "Max drawdown (daily, from start)": dd,
        "Max drawdown recovered (1 = yes, 0 = censored)": float(rec is not None),
        "Max drawdown length, peak to recovery (days)": np.nan if rec is None else float(rec),
        "Daily VaR 95% (loss)": var_d,
        "Daily ES 95% (loss)": dloss[dloss >= var_d].mean(),
        "Daily tail observations": float((dloss >= var_d).sum()),
        "Monthly VaR 95% (loss)": var_m,
        "Monthly ES 95% (loss)": (-r)[-r >= var_m].mean(),
        "Monthly tail observations": float((-r >= var_m).sum()),
        "Tracking error": te,
        "Information ratio": 12 * a.mean() / te if te > 0 else np.nan,
        "Beta (excess)": beta,
        "Turnover per year (sum |x - h|)": turn.sum() / years,
        "Turnover per rebalance, median": np.median(turn),
        "Turnover per rebalance, p95": np.quantile(turn, 0.95),
        "Cost per year (NAV bps)": 1e4 * costs.sum() / years,
        "Max name weight (mean)": xs.max(1).mean(),
        "Max sector weight (mean)": sec_w.max(1).mean(),
        "Effective N (mean)": (1 / (xs ** 2).sum(1)).mean(),
        "Top-15 share (contemporaneous, mean)": np.sort(xs, 1)[:, -15:].sum(1).mean(),
    }
    for label, covs in (panels or {}).items():
        md = np.array([mandate_deviation(t["x"], S, a_) for t, S, a_ in zip(log, covs, act)])
        for j, part in ((0, "sector"), (1, "within")):
            for q, f in (("mean", np.mean), ("median", np.median), ("p95", lambda v: np.quantile(v, 0.95)), ("max", np.max)):
                out[f"Mandate deviation, {part}, {q} ({label})"] = f(md[:, j])
        out[f"Mandate breach frequency ({label})"] = md[:, 2].mean()
    return pd.Series(out), r, e


def stationary_indices(n, reps, block, seed=0):
    """Politis-Romano stationary bootstrap: mean block length `block`, wrap-around; one index matrix shared by all
    methods and seeds (paired resampling)."""
    rng = np.random.default_rng(seed)
    idx = np.empty((reps, n), int)
    idx[:, 0] = rng.integers(n, size=reps)
    jump = rng.random((reps, n)) < 1 / block
    fresh = rng.integers(n, size=(reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def utility(e):
    return e - GAMMA * e ** 2 / 2


def contrast(e1, e2, idx):
    """Paired stationary bootstrap of A - B on aligned monthly excess returns: annualised mean, excess Sharpe and
    utility differences. e1 may be (seeds, n): A is then the expected training algorithm (Sec. 15.4), each statistic
    computed from every seed's own returns and then averaged over seeds, not a traded ensemble. Returns estimates,
    95% percentile intervals and centred two-sided p-values. The Sharpe row is descriptive (no p-value): its
    inference is sharpe_test, the studentized test Sec. 15.4 asks for."""
    m = lambda x: 12 * x.mean(-1)
    sh = lambda x: np.sqrt(12) * x.mean(-1) / x.std(-1, ddof=1)
    u = lambda x: 12 * utility(x).mean(-1)
    seeds = e1.ndim == 2
    agg = (lambda v: v.mean(0)) if seeds else (lambda v: v)
    stats = lambda a, b: np.stack([agg(m(a)) - m(b), agg(sh(a)) - sh(b), agg(u(a)) - u(b)])
    point = stats(e1, e2)
    boot = stats(e1[:, idx] if seeds else e1[idx], e2[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=1)
    p = (np.abs(boot - point[:, None]) >= np.abs(point)[:, None]).mean(1)
    p[1] = np.nan
    names = ["Excess return diff", "Excess Sharpe diff (descriptive interval)", "Utility diff"]
    return pd.DataFrame({"estimate": point, "ci_low": lo, "ci_high": hi, "p_value": p}, index=names)


def sharpe_test(e1, e2, block=6, reps=5000, seed=0):
    """Ledoit-Wolf (2008)-style studentized circular block bootstrap for H0: SR1 = SR2 (monthly excess returns).
    SE by the delta method on (mean, second moment) of both series; HAC (Bartlett, lag block-1) in the sample,
    block-sum estimator in the bootstrap world. Returns (annualised SR difference, p-value)."""
    def delta_se(a, b, boot=False):
        m1, m2, g1, g2 = a.mean(), b.mean(), (a ** 2).mean(), (b ** 2).mean()
        grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                         -m1 / (2 * (g1 - m1 ** 2) ** 1.5), m2 / (2 * (g2 - m2 ** 2) ** 1.5)])
        y = np.c_[a - m1, b - m2, a ** 2 - g1, b ** 2 - g2]
        if boot:
            z = y[: len(y) // block * block].reshape(-1, block, 4).sum(1) / np.sqrt(block)
            psi = z.T @ z / len(z)
        else:
            psi = y.T @ y / len(y)
            for j in range(1, block):
                gj = y[j:].T @ y[:-j] / len(y)
                psi += (1 - j / block) * (gj + gj.T)
        d = m1 / np.sqrt(g1 - m1 ** 2) - m2 / np.sqrt(g2 - m2 ** 2)
        return d, np.sqrt(grad @ psi @ grad / len(a))

    n = len(e1)
    d, se = delta_se(e1, e2)
    scale = np.sqrt(12 * (n - 1) / n)  # report on the ddof=1 scale used everywhere else; the test is scale-free
    if not se > 0:  # identical series
        return scale * d, 1.0
    rng = np.random.default_rng(seed)
    nb = n // block
    hits = 0
    for _ in range(reps):
        starts = rng.integers(n, size=nb)
        ix = ((starts[:, None] + np.arange(block)) % n).ravel()
        db, seb = delta_se(e1[ix], e2[ix], boot=True)
        hits += abs(db - d) / seb >= abs(d) / se
    return scale * d, (hits + 1) / (reps + 1)
