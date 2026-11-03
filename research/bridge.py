"""Correction bridge and competitive comparison for the research lane (issue #3 Sec. 11, 13-15).
Needs results/research/rl_targets.npz and rl_targets_v1ext.npz from research.walkforward.
uv run python -m research.bridge"""
import json

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

import data
import panel
import run
from data import ETFS, TICKERS, month_ends
from research import baselines as bl
from research.evaluate import contrast, metrics, sharpe_test, simulate, stationary_indices
from research.walkforward import OUT, START, folds

COST, LAG = 5e-4, 1                        # main protocol: 5 bps all-in per dollar traded, fill at next session's close
COSTS = (0.0, 5e-4, 1e-3, 2.5e-3, 5e-3)    # Sec. 14 panel 1
KAPPAS = (0.01, 0.1, 1.0)                  # mean-variance anchor, chosen per fold on its validation window
BLOCK, BLOCKS, REPS = 6, (3, 6, 12), 5000  # stationary bootstrap (months), Sec. 15.4
CUTOFF = pd.Timestamp("2025-07-31")        # last complete month; Aug 1-13 2025 reported separately
REALIZED_DAYS = 21                         # realized-proxy covariance: the 21 sessions after each fill
POINT, REAL = "point estimate S_t", "realized proxy"
HEAD = ["CAGR", "Excess Sharpe", "Sortino (MAR = rf)", "Max drawdown (daily, from start)", "Tracking error",
        "Information ratio", "Turnover per year (sum |x - h|)", "Months"]
CONV = {"B0": "paper: rf = 0, population SD, drawdown excl. start (monthly), Sortino std(min(r,0)), target turnover",
        "B1": "research definitions at monthly resolution, target-to-target turnover",
        "B2": "research definitions at monthly resolution, target-to-target turnover",
        "B3": "daily ledger, turnover on drifted holdings",
        "B4": "daily ledger, turnover on drifted holdings",
        "B5": "daily ledger, turnover on drifted holdings"}


class Market:
    def __init__(self):
        self.P = P = panel.build(START)
        adj, _, _ = data.load(START)
        self.days = adj.index
        self.R = adj[TICKERS].pct_change(fill_method=None).to_numpy()
        self.Re = adj[ETFS].pct_change(fill_method=None).to_numpy()
        self.etf_ok = adj[ETFS].notna().to_numpy()
        self.rf = data.risk_free(self.days, START).to_numpy()
        self.me_set = set(month_ends(self.days))
        self.dec = self.days.get_indexer(P["dates"])
        px = adj[TICKERS].to_numpy()
        self.mom = px[self.dec - 21] / px[self.dec - 252] - 1  # 12-1 momentum known at each decision close
        self.F = list(folds(P))
        self.t0, self.T = self.F[0][2], len(P["dates"])      # test decisions 2019-12-31 .. 2025-07-31
        self._real = {}

    def bench(self, t0, t1):
        ok = self.etf_ok[self.dec[t0:t1]]
        return ok / ok.sum(1, keepdims=True)

    def realized(self, t, row):
        """Ex-post proxy covariance for the holdings filled at `row`: Ledoit-Wolf on the next REALIZED_DAYS sessions.
        No method optimises against it, unlike S_t (which BHRP solves exactly and the RL penalty uses)."""
        if (t, row) not in self._real:
            a = self.P["act"][t]
            S = np.zeros((len(a), len(a)))
            S[np.ix_(a, a)] = ledoit_wolf(self.R[row + 1: row + 1 + REALIZED_DAYS][:, a])[0]
            self._real[(t, row)] = S
        return self._real[(t, row)]

    def evaluate(self, W, t0, t1, cost=COST, lag=LAG, etf=False, bench_nav=None):
        """Ledger + metrics for targets W decided at panel dates t0..t1-1 (complete months up to CUTOFF)."""
        start = self.dec[t0]
        end = self.dec[t1] if t1 < self.T else len(self.days) - 1
        rows = self.dec[t0:t1] + lag
        nav, log = simulate(W, rows, self.Re if etf else self.R, self.rf, cost, start, end)
        if bench_nav is None:
            bench_nav = nav if etf else self.evaluate(self.bench(t0, t1), t0, t1, cost, lag, etf=True)[0]
        days = self.days[start:end + 1]
        me = np.array([d in self.me_set and d <= CUTOFF for d in days])
        k = int((self.days[rows] <= CUTOFF).sum())  # trades filled inside the complete-month window
        ts = range(t0, t0 + k)
        panels = None if etf else {POINT: self.P["S"][t0:t0 + k], REAL: [self.realized(t, r) for t, r in zip(ts, rows)]}
        m, r, e = metrics(nav, days, bench_nav, self.rf[start:end + 1], log[:k], me, panels, self.P["act"][t0:t0 + k])
        m["Aug 1-13 2025 return (partial, not in the stats)"] = nav[-1] / nav[np.flatnonzero(me)[-1]] - 1 if t1 == self.T else np.nan
        return nav, m, r, e

    def targets(self, fn, t0, t1, **kw):
        out = [bl.project(self.P, t, fn(self.P, t, mom=self.mom, **kw)) for t in range(t0, t1)]
        return np.array([w for w, _ in out]), np.mean([b for _, b in out])


def monthly_metrics(r, b, start, ends, W, prev, cost, rf_d):
    """Research definitions on a monthly return series (bridge steps B1-B2, before the daily ledger; drawdown and
    tail metrics therefore at monthly resolution). rf_d: daily cash return Series on trading days."""
    days = pd.DatetimeIndex([start, *ends])
    rf_m = np.array([np.prod(1 + rf_d[(rf_d.index >= a) & (rf_d.index < z)]) - 1 for a, z in zip(days[:-1], days[1:])])
    turn = np.abs(W - np.vstack([prev, W[:-1]])).sum(1)
    log = [dict(turnover=t, cost=cost * t, x=w) for t, w in zip(turn, W)]
    nav, bnav = np.r_[1, np.cumprod(1 + r)], np.r_[1, np.cumprod(1 + b)]
    return metrics(nav, days, bnav, np.r_[rf_m, 0], log, np.ones(len(days), bool))[0]


def md(df, fmt="{:.4f}"):
    cols = list(df.columns)
    rows = ["| | " + " | ".join(map(str, cols)) + " |", "|---|" + "---:|" * len(cols)]
    for i, row in df.iterrows():
        rows.append(f"| {i} | " + " | ".join("—" if pd.isna(v) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in row) + " |")
    return "\n".join(rows)


def main():
    mk = Market()
    P, t0, T = mk.P, mk.t0, mk.T
    z = np.load(OUT / "rl_targets.npz", allow_pickle=True)
    zx = np.load(OUT / "rl_targets_v1ext.npz", allow_pickle=True)
    for d in (z["wf_dates"], z["v1_dates"], zx["dates"]):
        assert (d == P["dates"][t0:T]).all()
    out = {}

    # ---- competitive comparison, main protocol ---------------------------------------------------------------------
    bench_nav, bench_m, _, bench_e = mk.evaluate(mk.bench(t0, T), t0, T, etf=True)
    W, bound = {}, {}
    for name, fn in bl.METHODS.items():
        if name == "mean_variance":
            parts, picks = [], {}
            for Y, tr, va, te in mk.F:
                val = {k: mk.evaluate(mk.targets(fn, tr, va, kappa=k)[0], tr, va)[1]["Excess Sharpe"] for k in KAPPAS}
                picks[Y] = max(val, key=val.get)
                parts.append(mk.targets(fn, va, te, kappa=picks[Y]))
            W[name], bound[name] = np.concatenate([p[0] for p in parts]), np.mean([p[1] for p in parts])
            out["mean_variance_kappa"] = picks
        else:
            W[name], bound[name] = mk.targets(fn, t0, T)
    rl = {}
    for key, arr in (("v1", z["v1"]), ("v1ext", zx["v1ext"]), ("wf", z["wf"])):
        proj = [[bl.project(P, t0 + i, w) for i, w in enumerate(seed)] for seed in arr]
        rl[key] = np.array([[w for w, _ in s] for s in proj])
        bound["rl_" + key] = np.mean([[b for _, b in s] for s in proj])

    labels = dict(ew="Asset EW", sector_ew="Sector EW", erc="ERC", bhrp="BHRP", hrp="HRP (Lopez de Prado)",
                  minvar="Min variance (LW)", mean_variance="Mean-variance (L2 anchor)", momentum="12-1 momentum")
    rl_labels = {"v1": "RL-BHRP v1 policy (10 seeds, fit 2015-2019)", "wf": "RL-BHRP walk-forward (20 seeds)"}
    res, E, NAV = {"Benchmark (EW sector ETFs)": bench_m}, {"Benchmark (EW sector ETFs)": bench_e}, {}
    for name in bl.METHODS:
        NAV[name], res[labels[name]], _, E[labels[name]] = mk.evaluate(W[name], t0, T, bench_nav=bench_nav)
        res[labels[name]]["Mandate projection binding (share of dates)"] = bound[name]
    seeds = {}
    for key in ("v1", "v1ext", "wf"):
        runs = [mk.evaluate(w, t0, T, bench_nav=bench_nav) for w in rl[key]]
        seeds[key] = pd.DataFrame([m for _, m, _, _ in runs])
        E["rl_" + key] = np.array([e for _, _, _, e in runs])
        NAV["rl_" + key] = np.array([n for n, _, _, _ in runs])
        if key in rl_labels:
            res[rl_labels[key]] = seeds[key].mean()
            res[rl_labels[key]]["Mandate projection binding (share of dates)"] = bound["rl_" + key]
            E[rl_labels[key]] = E["rl_" + key]
    comp = pd.DataFrame(res)

    # ---- paired contrasts vs BHRP -------------------------------------------------------------------------------------
    eb = E["BHRP"]
    idx = stationary_indices(len(eb), REPS, BLOCK)
    rows = []
    for lab, e in E.items():
        if lab == "BHRP" or lab.startswith("rl_"):
            continue
        for stat, r in contrast(e, eb, idx).iterrows():
            rows.append(dict(method=lab, stat=stat, **r))
        path = e[0] if e.ndim == 2 else e  # RL: the predeclared first seed (1001 walk-forward, 0 for v1) as one path
        d, p = sharpe_test(path, eb, block=BLOCK, reps=REPS)
        suffix = "" if e.ndim == 1 else (", seed 1001 path" if "walk" in lab else ", seed 0 path")
        rows.append(dict(method=lab + suffix, stat="Excess Sharpe diff (studentized test)", estimate=d, p_value=p))
    ewf = E["rl_wf"]
    for b in BLOCKS:
        for stat, r in contrast(ewf, eb, stationary_indices(len(eb), REPS, b, seed=b)).iterrows():
            rows.append(dict(method=f"RL-BHRP walk-forward, block {b}", stat=stat, **r))
        d, p = sharpe_test(ewf[0], eb, block=b, reps=REPS)
        rows.append(dict(method=f"RL-BHRP walk-forward, block {b}, seed 1001 path",
                         stat="Excess Sharpe diff (studentized test)", estimate=d, p_value=p))
    contrasts = pd.DataFrame(rows)

    # ---- cost panel: policies and targets fixed (RL trained at 5 bps), Sec. 14 panel 1 -----------------------------------
    cost_rows = []
    for c in COSTS:
        bn = mk.evaluate(mk.bench(t0, T), t0, T, cost=c, etf=True)
        cost_rows.append(dict(cost_bps=1e4 * c, method="Benchmark (EW sector ETFs)", CAGR=bn[1]["CAGR"],
                              sharpe=bn[1]["Excess Sharpe"]))
        for name in bl.METHODS:
            m = mk.evaluate(W[name], t0, T, cost=c, bench_nav=bn[0])[1]
            cost_rows.append(dict(cost_bps=1e4 * c, method=labels[name], CAGR=m["CAGR"], sharpe=m["Excess Sharpe"]))
        ms = pd.DataFrame([mk.evaluate(w, t0, T, cost=c, bench_nav=bn[0])[1] for w in rl["wf"]])
        cost_rows.append(dict(cost_bps=1e4 * c, method=rl_labels["wf"], CAGR=ms["CAGR"].mean(), sharpe=ms["Excess Sharpe"].mean()))
    costs = pd.DataFrame(cost_rows).pivot(index="method", columns="cost_bps")

    # ---- correction bridge: one change per step ---------------------------------------------------------------------------
    P0 = panel.build()
    pub = pd.read_csv("results/shared-global/table2.csv", index_col=0)
    pw = np.load("results/shared-global/weights.npz", allow_pickle=True)
    pr = pd.read_csv("results/shared-global/returns.csv", index_col=0)
    tt = int(np.searchsorted(P0["dates"], np.datetime64("2020-01-31")))
    tc = tt - 1                                                   # 2019-12-31
    ew0 = lambda t: P0["act"][t] / P0["act"][t].sum()
    rf_d = pd.Series(mk.rf, mk.days)
    bridge = []

    def row(step, lab, m, extra=None):
        bridge.append(dict(step=step, strategy=lab, **{h: m[h] for h in HEAD}, **(extra or {}), conventions=CONV[step[:2]]))

    def rl_row(step, lab, ms):
        row(step, lab, ms.mean(), {"CAGR seed range": f"{ms.CAGR.min():.4f} to {ms.CAGR.max():.4f}"})

    paper = {"BHRP": "BHRP", "RL-BHRP v1 (10 seeds)": "RL-BHRP (seed mean)", "Benchmark": "Benchmark"}
    turn0 = {"BHRP": 12 * np.abs(np.diff(np.vstack([ew0(tt), pw["bhrp"]]), axis=0)).sum(1).mean(),
             "RL-BHRP v1 (10 seeds)": np.mean([12 * np.abs(np.diff(np.vstack([ew0(tt), w]), axis=0)).sum(1).mean()
                                               for w in pw["rl"]]), "Benchmark": 0.0}
    for lab, col in paper.items():
        row("B0 v1.0.0 as published", lab,
            {"CAGR": pub.loc["CAGR", col], "Excess Sharpe": pub.loc["Sharpe", col], "Sortino (MAR = rf)": pub.loc["Sortino", col],
             "Max drawdown (daily, from start)": -pub.loc["Max Drawdown", col], "Tracking error": pub.loc["Tracking Error", col],
             "Information ratio": pub.loc["Information Ratio", col], "Turnover per year (sum |x - h|)": turn0[lab], "Months": 67})
    ends0 = pd.DatetimeIndex(P0["ret_dates"][tt:])
    for lab, cols, Ws in (("BHRP", ["bhrp"], [pw["bhrp"]]), ("RL-BHRP v1 (10 seeds)", [f"rl_seed{s}" for s in range(10)], pw["rl"])):
        ms = pd.DataFrame([monthly_metrics(pr[c].to_numpy(), pr["benchmark"].to_numpy(), P0["dates"][tt], ends0, w, ew0(tt),
                                           run.C, rf_d) for c, w in zip(cols, Ws)])
        (rl_row if len(Ws) > 1 else row)("B1 + research metric definitions (excess of T-bill, RMS Sortino, NAV from start)",
                                         lab, *([ms] if len(Ws) > 1 else [ms.mean()]))
    ends2 = pd.DatetimeIndex(P0["ret_dates"][tc:-1])              # Jan 2020 .. Jul 2025
    b2 = P0["bench"][tc:-1]
    for lab, Ws in (("BHRP", [P0["w_bhrp"][tc:]]), ("RL-BHRP v1 (10 seeds)", z["v1"])):
        ms = pd.DataFrame([monthly_metrics(run.hold(P0, tc, len(P0["dates"]) - 1, w[:-1]), b2, P0["dates"][tc], ends2, w[:-1],
                                           ew0(tc), run.C, rf_d) for w in Ws])
        (rl_row if len(Ws) > 1 else row)("B2 + complete months only (Jan 2020 - Jul 2025)", lab,
                                         *([ms] if len(Ws) > 1 else [ms.mean()]))
    for step, lag in (("B3 + drift-aware daily ledger, start in cash, same-close fills", 0),
                      ("B4 + next-session fills (main protocol)", 1)):
        bn = mk.evaluate(mk.bench(t0, T), t0, T, lag=lag, etf=True)
        row(step, "Benchmark", bn[1])
        wb = np.array([bl.project(P, t0 + i, w)[0] for i, w in enumerate(P0["w_bhrp"][tc:])])
        row(step, "BHRP", mk.evaluate(wb, t0, T, lag=lag, bench_nav=bn[0])[1])
        rl_row(step, "RL-BHRP v1 (10 seeds)", pd.DataFrame([mk.evaluate(w, t0, T, lag=lag, bench_nav=bn[0])[1] for w in rl["v1"]]))
    rl_row("B5a + 2009-history panel (training starts 2012), v1 protocol: one fit, lambda 1e6, 300 iters",
           "RL-BHRP v1 protocol (10 seeds)", seeds["v1ext"])
    row("B5a + 2009-history panel (training starts 2012), v1 protocol: one fit, lambda 1e6, 300 iters", "BHRP", res["BHRP"])
    rl_row("B5b + nested annual walk-forward retraining and tuning (20 seeds)", "RL-BHRP walk-forward (20 seeds)", seeds["wf"])
    bridge = pd.DataFrame(bridge)

    # ---- outputs -----------------------------------------------------------------------------------------------------
    comp.to_csv(OUT / "comparison.csv")
    contrasts.to_csv(OUT / "contrasts.csv", index=False)
    costs.to_csv(OUT / "costs.csv")
    bridge.to_csv(OUT / "bridge.csv", index=False)
    seeds["wf"].assign(seed=z["wf_seeds"]).to_csv(OUT / "rl_seeds.csv", index=False)
    out |= dict(folds=json.loads((OUT / "rl_folds.json").read_text()), projection_binding=bound)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(OUT / "nav.npz", days=np.array(mk.days[mk.dec[t0]:]), rl_wf=NAV["rl_wf"], bhrp=NAV["bhrp"], bench=bench_nav)

    key = ["CAGR", "Excess return (12 x mean)", "Volatility", "Excess Sharpe", "Sortino (MAR = rf)", "Utility (12 x mean u_3)",
           "Max drawdown (daily, from start)", "Tracking error", "Information ratio", "Turnover per year (sum |x - h|)",
           "Cost per year (NAV bps)", "Effective N (mean)", "Mandate projection binding (share of dates)"]
    key += [f"Mandate {s} ({p})" for p in (POINT, REAL) for s in ("deviation, sector, mean", "deviation, within, mean",
                                                                  "breach frequency")]
    sel = contrasts[~contrasts.stat.str.startswith("Excess return")]
    disclose = "\n".join([
        "- Main protocol: decision at month-end close, fill at the next session's close, 5 bps per dollar traded on drifted",
        "  holdings, start in cash on 2019-12-31, cash at the 13-week T-bill, complete months Jan 2020 - Jul 2025. Terminal",
        "  liquidation is not charged (at most ~1 bp/yr, the same for every method).",
        "- Retrospective: the 756-day window, the shared actor, the penalty centering and alpha were fixed in the v1.0.0",
        "  reproduction, which had already looked at 2020-2025. The walk-forward removes look-ahead from lambda and the",
        "  iteration count only.",
        "- The RL validation loss is v1.0.0's reproduction Sharpe (rf = 0, target-to-target costs); mean-variance kappa is",
        "  chosen on the ledger's excess Sharpe. Other methods have no tuned parameters.",
        f"- Mandate metrics: '{POINT}' is the Ledoit-Wolf covariance BHRP solves exactly and the RL penalty uses, so BHRP",
        f"  meets it by construction; '{REAL}' (Ledoit-Wolf on the {REALIZED_DAYS} sessions after each fill) is graded",
        "  identically for every method and optimised by none.",
        "- RL rows are the expected training algorithm: every statistic is computed from each seed's own ledger, then",
        "  averaged. Studentized Sharpe tests use one predeclared path (seed 1001; seed 0 for v1). The Sharpe interval in",
        "  the bootstrap contrast is descriptive only.",
        "- The universe is the paper's fixed list of 48 large caps chosen in 2025: survivorship bias favours anything",
        "  that chases past winners (momentum, mean-variance on trailing means). A point-in-time universe is required",
        "  before any of these differences can be read as evidence.",
        "- BHRP already solves the exact risk-budget fixed point (b_i = 1/(G n_g)), so the issue's 'corrected risk",
        "  budgeting' bridge step changes nothing here and is omitted."])
    report = "\n\n".join([
        "# Research lane results\n\nGenerated by `uv run python -m research.bridge`; see README section \"Research lane (issue #3)\".",
        "## Protocol and disclosures\n\n" + disclose,
        "## Correction bridge (one change per step)\n\n" + md(bridge.set_index(["step", "strategy"]).reset_index(level=1)),
        "## Competitive comparison (main protocol)\n\n" + md(comp.loc[key].T),
        "## Paired contrasts vs BHRP (stationary bootstrap, 5,000 resamples; studentized Sharpe test)\n\n" + md(sel.set_index("method")),
        "## Cost panel (policies fixed; CAGR / excess Sharpe)\n\n" + md(costs),
        "## RL walk-forward seeds\n\n" + md(seeds["wf"][["CAGR", "Excess Sharpe", "Max drawdown (daily, from start)",
                                                          "Tracking error", "Information ratio"]].describe().T)])
    (OUT / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
