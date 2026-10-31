"""Is the RL edge over BHRP robust to the unreported choices? Test-period outcomes over a grid of
(centering, lambda, iterations) x seeds. A robustness report, NOT a selection step (that would be test snooping).
uv run python sensitivity.py"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import panel
from run import ALPHA, C, hold, metrics

GRID = [(cen, lam) for cen in ("global", "sector") for lam in (1e5, 1e6, 1e7)]
CHECK = (50, 100, 200)
SEEDS = 10


def _job(job):
    import torch, rl
    torch.set_num_threads(1)
    P, (cen, lam, seed, t1, t2) = panel.build(), job
    kw = dict(c=C, lam=lam, alpha=ALPHA, centering=cen)
    out = []

    def on_iter(it, net, stats):
        if it in CHECK:
            r, _ = rl.evaluate(net, stats, P, t2, len(P["dates"]), kw)
            m = metrics(r, P["bench"][t2:])
            out.append(dict(centering=cen, lam=lam, iters=it, seed=seed, CAGR=m["CAGR"], Sharpe=m["Sharpe"],
                            TE=m["Tracking Error"]))
    rl.train(P, 0, t1, seed=seed, env_kw=kw, iters=max(CHECK), on_iter=on_iter)
    return out


if __name__ == "__main__":
    P = panel.build()
    t1 = int(np.searchsorted(P["ret_dates"], np.datetime64("2020-01-01")))
    t2 = int(np.searchsorted(P["dates"], np.datetime64("2020-01-31")))
    base = metrics(hold(P, t2, len(P["dates"]), P["w_bhrp"][t2:]), P["bench"][t2:])
    with Pool(12) as pool:
        rows = sum(pool.map(_job, [(c, l, s, t1, t2) for c, l in GRID for s in range(SEEDS)]), [])
    df = pd.DataFrame(rows)
    df.to_csv("results/sensitivity.csv", index=False)
    g = df.groupby(["centering", "lam", "iters"])
    summary = pd.DataFrame({
        "CAGR mean": g["CAGR"].mean(), "CAGR sd": g["CAGR"].std(),
        "Sharpe mean": g["Sharpe"].mean(), "TE mean": g["TE"].mean(),
        "P(CAGR > BHRP)": g["CAGR"].apply(lambda x: (x > base["CAGR"]).mean()),
        "P(Sharpe > BHRP)": g["Sharpe"].apply(lambda x: (x > base["Sharpe"]).mean()),
    })
    print(f"BHRP test: CAGR {base['CAGR']:.4f}, Sharpe {base['Sharpe']:.4f} | paper RL-BHRP: CAGR 0.1516, Sharpe 0.9048, TE 0.0369\n")
    print(summary.round(4).to_string())
    print(f"\nall {len(df)} runs: P(CAGR > BHRP) = {(df.CAGR > base['CAGR']).mean():.2f}, "
          f"P(Sharpe > BHRP) = {(df.Sharpe > base['Sharpe']).mean():.2f}, "
          f"P(CAGR >= paper 15.16%) = {(df.CAGR >= 0.1516).mean():.2f}")
    summary.to_csv("results/sensitivity_summary.csv")
