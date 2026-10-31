"""Sec. 4 backtest: RL-BHRP vs BHRP vs equal-weight sector-ETF benchmark, Table 2 metrics, weight diagnostics.
uv run python run.py [--actor flat|shared] [--centering global|sector] [--seeds 10]"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import panel
from data import TICKERS

C = 5e-4          # 5 bps one-way (Sec. 4.2)
ALPHA = 0.5       # within/across split; not reported in the paper
LAMS = (1e5, 1e6, 1e7, 1e8)
ITERS = (25, 50, 100, 200, 300)
VAL_MONTHS = 24   # last 24 of the 59 training decisions (2017-12 .. 2019-11) for calibration

PAPER = pd.DataFrame({  # Table 2
    "RL-BHRP": [1.199612, .151637, .157213, .173748, .904829, 1.648382, -.203308, .745850, .036883, .686752, .982509, .027636, -.074066, -.102409, .641791],
    "BHRP": [1.013672, .133563, .139742, .165164, .846076, 1.534682, -.191023, .699201, .035790, .219576, .935146, .016412, -.067241, -.097049, .641791],
    "Benchmark": [.914063, .123310, .131883, .172838, .763042, 1.368210, -.183210, .673053, 0, np.nan, 1, 0, -.079889, -.103351, .626866],
}, index=["Total Cumulative Return", "CAGR", "Annual Return (Mean x12)", "Annual Volatility", "Sharpe", "Sortino",
          "Max Drawdown", "Calmar", "Tracking Error", "Information Ratio", "Beta", "Jensen Alpha", "VaR 5% (period)",
          "CVaR 5% (period)", "Hit Rate (>0)"])


def metrics(r, b):
    """Table 2, monthly returns, rf = 0, 12 periods/yr, using the paper's conventions (recovered by matching
    its Benchmark column exactly): population std; drawdown peaks exclude the starting wealth 1.0; Sortino
    denominator = std(min(r, 0)). The last two rows give the standard drawdown and Sortino."""
    r, b = np.asarray(r), np.asarray(b)
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    mdd = (wealth / peak - 1).min()
    cagr = wealth[-1] ** (12 / len(r)) - 1
    ann, vol = 12 * r.mean(), np.sqrt(12) * r.std()
    te = np.sqrt(12) * (r - b).std()
    beta = np.cov(r, b, ddof=0)[0, 1] / b.var()
    var5 = np.quantile(r, 0.05)
    down = np.minimum(r, 0)
    return pd.Series([wealth[-1] - 1, cagr, ann, vol, ann / vol, ann / (np.sqrt(12) * down.std()),
                      mdd, cagr / -mdd, te, 12 * (r - b).mean() / te if te > 0 else np.nan, beta,
                      12 * (r.mean() - beta * b.mean()), var5, r[r <= var5].mean(), (r > 0).mean(),
                      (wealth / np.maximum(peak, 1) - 1).min(), ann / np.sqrt(12 * (down ** 2).mean())],
                     index=[*PAPER.index, "Max Drawdown (standard, from 1.0)", "Sortino (standard, RMS downside)"])


def hold(P, t0, t1, W):
    """Net-of-cost returns of a weight path W (t1-t0, N), w_{-1} = equal weight (same cost model as RL)."""
    a = P["act"][t0]
    prev = np.vstack([a / a.sum(), W[:-1]])
    return np.einsum("tn,tn->t", W, P["R"][t0:t1]) - C * np.abs(W - prev).sum(1)


def diagnostics(W):
    """Sec. 4.2: Top-15 share, L1 turnover, cosine persistence."""
    top = np.argsort(-W.mean(0))[:15]
    share = W[:, top].sum(1)
    tau = np.abs(np.diff(W, axis=0)).sum(1)
    rho = (W[1:] * W[:-1]).sum(1) / np.linalg.norm(W[1:], axis=1) / np.linalg.norm(W[:-1], axis=1)
    return dict(top15=[TICKERS[i] for i in top], top15_share_mean=share.mean(), top15_share_median=np.median(share),
                top15_share_range=[share.min(), share.max()], turnover_mean=tau.mean(), turnover_p95=np.quantile(tau, .95),
                cosine_mean=rho.mean(), names_traded=int((W > 1e-4).any(0).sum()))


def _tune(job):
    import torch, rl
    torch.set_num_threads(1)
    P, (lam, seed, t0, t1, t2, centering, actor) = panel.build(), job
    kw = dict(c=C, lam=lam, alpha=ALPHA, centering=centering)
    out = {}

    def on_iter(it, net, stats):
        if it in ITERS:
            r, _ = rl.evaluate(net, stats, P, t1, t2, kw)
            out[it] = metrics(r, P["bench"][t1:t2])["Sharpe"]
    rl.train(P, t0, t1, seed=seed, env_kw=kw, actor=actor, iters=max(ITERS), on_iter=on_iter)
    return lam, seed, out


def _final(job):
    import torch, rl
    torch.set_num_threads(1)
    P, (lam, iters, seed, t1, t2, centering, actor) = panel.build(), job
    kw = dict(c=C, lam=lam, alpha=ALPHA, centering=centering)
    net, m, s = rl.train(P, 0, t1, seed=seed, env_kw=kw, actor=actor, iters=iters)
    return rl.evaluate(net, (m, s), P, t2, len(P["dates"]), kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="flat", choices=["flat", "shared"])
    ap.add_argument("--centering", default="global", choices=["global", "sector"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tune-seeds", type=int, default=3)
    args = ap.parse_args()
    out = Path("results") / f"{args.actor}-{args.centering}"
    out.mkdir(parents=True, exist_ok=True)

    P = panel.build()
    T = len(P["dates"])
    t_train = int(np.searchsorted(P["ret_dates"], np.datetime64("2020-01-01")))  # returns realised by 2019-12-31
    t_test = int(np.searchsorted(P["dates"], np.datetime64("2020-01-31")))       # first return: Feb 2020
    t_val = t_train - VAL_MONTHS
    bench = P["bench"][t_test:]
    ret_dates = pd.DatetimeIndex(P["ret_dates"][t_test:]).to_period("M").to_timestamp("M")

    with Pool(min(12, len(LAMS) * args.tune_seeds)) as pool:
        tuned = pool.map(_tune, [(l, s, 0, t_val, t_train, args.centering, args.actor)
                                 for l in LAMS for s in range(args.tune_seeds)])
    val = pd.DataFrame([{"lam": l, "seed": s, "iters": i, "val_sharpe": v} for l, s, o in tuned for i, v in o.items()])
    val_bhrp = metrics(hold(P, t_val, t_train, P["w_bhrp"][t_val:t_train]), P["bench"][t_val:t_train])["Sharpe"]
    grid = val.pivot_table(index="lam", columns="iters", values="val_sharpe", aggfunc="mean")
    lam, iters = grid.stack().idxmax()
    print(f"validation Sharpe (mean of {args.tune_seeds} seeds; BHRP = {val_bhrp:.3f}):\n{grid.round(3)}\n-> lam={lam:g}, iters={iters}")

    with Pool(min(12, args.seeds)) as pool:
        runs = pool.map(_final, [(lam, int(iters), s, t_train, t_test, args.centering, args.actor)
                                 for s in range(args.seeds)])
    R = np.array([r for r, _ in runs])
    Ws = np.array([w for _, w in runs])
    seed_m = pd.DataFrame([metrics(r, bench) for r in R])
    med = int(np.argsort(seed_m["CAGR"].to_numpy())[len(R) // 2])  # median-CAGR seed for the full table/figures

    W_bhrp = P["w_bhrp"][t_test:]
    r_bhrp = hold(P, t_test, T, W_bhrp)
    table = pd.DataFrame({
        "RL-BHRP (seed mean)": seed_m.mean(), "RL-BHRP (seed min)": seed_m.min(), "RL-BHRP (seed max)": seed_m.max(),
        "BHRP": metrics(r_bhrp, bench), "Benchmark": metrics(bench, bench),
        "paper RL-BHRP": PAPER["RL-BHRP"], "paper BHRP": PAPER["BHRP"], "paper Benchmark": PAPER["Benchmark"]}
    ).reindex(seed_m.columns)
    pd.set_option("display.width", 200)
    print(f"\ntest {ret_dates[0].date()} .. {ret_dates[-1].date()} ({len(bench)} periods), actor={args.actor}, centering={args.centering}\n")
    print(table.round(4).to_string())
    beats = (seed_m["CAGR"] > table.loc["CAGR", "BHRP"]).sum()
    print(f"\nseeds beating BHRP on CAGR: {beats}/{len(R)}; on Sharpe: {(seed_m['Sharpe'] > table.loc['Sharpe', 'BHRP']).sum()}/{len(R)}")
    diag = {"RL-BHRP (median seed)": diagnostics(Ws[med]), "BHRP": diagnostics(W_bhrp)}
    print(json.dumps(diag, indent=1, default=float))

    table.to_csv(out / "table2.csv")
    seed_m.to_csv(out / "seeds.csv")
    val.to_csv(out / "validation.csv")
    pd.DataFrame(np.c_[R.T, r_bhrp, bench], index=ret_dates,
                 columns=[f"rl_seed{s}" for s in range(len(R))] + ["bhrp", "benchmark"]).to_csv(out / "returns.csv")
    np.savez_compressed(out / "weights.npz", rl=Ws, bhrp=W_bhrp, dates=P["dates"][t_test:], tickers=TICKERS)
    (out / "summary.json").write_text(json.dumps(dict(actor=args.actor, centering=args.centering, lam=lam,
                                                      iters=int(iters), alpha=ALPHA, c=C, median_seed=med,
                                                      diagnostics=diag), indent=1, default=float))


if __name__ == "__main__":
    main()
