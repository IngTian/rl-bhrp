"""RL-BHRP under the issue #3 Sec. 13 protocol: nested annual walk-forward (inner train -> validation -> refit ->
deploy one year, holdings/targets carried across folds), 20 final seeds 1001-1020. Also re-trains the v1.0.0
policies (seeds 0-9) so the correction bridge can re-evaluate them from the first complete test month.
Validation loss: the reproduction Sharpe of run.metrics (rf = 0, target-to-target costs), i.e. v1.0.0's own
tuning rule, not the research ledger's excess Sharpe. Window (756 d), actor (shared), centering and alpha are fixed
from the v1.0.0 reproduction, which had already inspected 2020-2025, so these results are retrospective evidence.
Heavy (about 30 min on 12 processes); writes results/research/rl_targets.npz.
uv run python -m research.walkforward          # walk-forward + v1 re-evaluation
uv run python -m research.walkforward v1ext    # bridge step B5a: v1 protocol, one fit, on the 2009-history panel"""
import sys
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import panel
import run

OUT = Path("results/research")
START = "2009-01-01"
TEST_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
TUNE_SEEDS = (1, 2, 3)
FINAL_SEEDS = tuple(range(1001, 1021))
ACTOR, CENTERING = "shared", "global"   # the v1.0.0 headline configuration
V1 = dict(lam=1e6, iters=300, seeds=range(10))  # results/shared-global/summary.json


def kw(lam):
    return dict(c=run.C, lam=lam, alpha=run.ALPHA, centering=CENTERING)


def folds(P):
    """(test year, end of inner train, end of validation, end of test) as panel indices; decisions are assigned to
    the calendar year in which their holding-period return is realised."""
    y = pd.DatetimeIndex(P["ret_dates"]).year
    for Y in TEST_YEARS:
        yield Y, *(int(np.searchsorted(y, v)) for v in (Y - 2, Y, Y + 1))


def _tune(job):
    import torch
    import rl
    torch.set_num_threads(1)
    Y, lam, seed, tr, va = job
    P, out = panel.build(START), {}

    def on_iter(it, net, stats):
        if it in run.ITERS:
            r, _ = rl.evaluate(net, stats, P, tr, va, kw(lam))
            out[it] = run.metrics(r, P["bench"][tr:va])["Sharpe"]
    rl.train(P, 0, tr, seed=seed, env_kw=kw(lam), actor=ACTOR, iters=max(run.ITERS), on_iter=on_iter)
    return Y, lam, seed, out


def _fit(job):
    import torch
    import rl
    torch.set_num_threads(1)
    start, lam, iters, seed, t1 = job
    P = panel.build(start) if start else panel.build()
    net, m, s = rl.train(P, 0, t1, seed=seed, env_kw=kw(lam), actor=ACTOR, iters=iters)
    return net.state_dict(), m, s


def _net(state):
    import rl
    net = rl.ActorCritic(rl.N * 7, rl.G + rl.N, ACTOR)
    net.load_state_dict(state)
    return net


def main():
    import rl
    OUT.mkdir(parents=True, exist_ok=True)
    P = panel.build(START)
    P0 = panel.build()
    t0_train = int(np.searchsorted(P0["ret_dates"], np.datetime64("2020-01-01")))  # v1 training end = 2019-12-31 decision
    F = list(folds(P))
    with Pool(12) as pool:
        tuned = pool.map(_tune, [(Y, l, s, tr, va) for Y, tr, va, _ in F for l in run.LAMS for s in TUNE_SEEDS])
        v1 = pool.map(_fit, [(None, V1["lam"], V1["iters"], s, t0_train) for s in V1["seeds"]])
        grid = pd.DataFrame([dict(year=Y, lam=l, seed=s, iters=i, val_sharpe=v) for Y, l, s, o in tuned for i, v in o.items()])
        pick = {Y: g.groupby(["lam", "iters"]).val_sharpe.mean().idxmax() for Y, g in grid.groupby("year")}
        fits = pool.map(_fit, [(START, pick[Y][0], int(pick[Y][1]), s, va) for Y, _, va, _ in F for s in FINAL_SEEDS])
    grid.to_csv(OUT / "rl_tuning.csv", index=False)
    (OUT / "rl_folds.json").write_text(json.dumps(
        {int(Y): dict(train=[str(P["dates"][0])[:10], str(P["dates"][tr - 1])[:10]],
                      validation=[str(P["dates"][tr])[:10], str(P["dates"][va - 1])[:10]],
                      test=[str(P["dates"][va])[:10], str(P["dates"][te - 1])[:10]],
                      lam=pick[Y][0], iters=int(pick[Y][1])) for Y, tr, va, te in F}, indent=1))

    # v1 policies: must reproduce the published test weights before being re-used from 2019-12-31
    pub = np.load("results/shared-global/weights.npz", allow_pickle=True)["rl"]
    v1_w = []
    for (state, m, s), seed in zip(v1, V1["seeds"]):
        net = _net(state)
        _, w_pub = rl.evaluate(net, (m, s), P0, t0_train + 1, len(P0["dates"]), kw(V1["lam"]))
        assert np.abs(w_pub - pub[seed]).max() < 1e-9, f"v1 seed {seed} does not reproduce"
        v1_w.append(rl.evaluate(net, (m, s), P0, t0_train, len(P0["dates"]), kw(V1["lam"]))[1])

    # walk-forward deployment: each seed's last target enters the next fold's state
    wf_w = []
    for k, seed in enumerate(FINAL_SEEDS):
        ws, prev = [], None
        for f, (Y, _, va, te) in enumerate(F):
            state, m, s = fits[f * len(FINAL_SEEDS) + k]
            _, w = rl.evaluate(_net(state), (m, s), P, va, te, kw(pick[Y][0]), w0=prev)
            ws.append(w)
            prev = w[-1]
        wf_w.append(np.concatenate(ws))
    np.savez_compressed(OUT / "rl_targets.npz", v1=np.array(v1_w), v1_dates=P0["dates"][t0_train:],
                        wf=np.array(wf_w), wf_dates=P["dates"][F[0][2]:F[-1][3]], wf_seeds=np.array(FINAL_SEEDS))
    print(json.dumps({int(Y): pick[Y] for Y in pick}, default=float))


def v1ext():
    """Bridge step B5a: the unchanged v1 protocol (lambda 1e6, 300 iterations, seeds 0-9, one fit on returns through
    2019) on the research panel, whose history starts in 2012 instead of 2015."""
    import rl
    P = panel.build(START)
    t_train = int(np.searchsorted(P["ret_dates"], np.datetime64("2020-01-01")))
    t0 = next(folds(P))[2]
    with Pool(10) as pool:
        fits = pool.map(_fit, [(START, V1["lam"], V1["iters"], s, t_train) for s in V1["seeds"]])
    w = [rl.evaluate(_net(st), (m, s), P, t0, len(P["dates"]), kw(V1["lam"]))[1] for st, m, s in fits]
    np.savez_compressed(OUT / "rl_targets_v1ext.npz", v1ext=np.array(w), dates=P["dates"][t0:])


if __name__ == "__main__":
    v1ext() if sys.argv[1:] == ["v1ext"] else main()
