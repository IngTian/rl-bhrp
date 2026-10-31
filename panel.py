"""Monthly decision panel: at each month-end t, Bayesian posterior + BHRP weights from data up to t,
and the realised next-period return (Sec. 3.2: weights chosen at t, held over [t, t+1])."""
from pathlib import Path

import numpy as np
import pandas as pd

from bhrp import bhrp, posterior
from data import ETFS, SECTOR_OF, TICKERS, load, month_ends, tradable

WINDOW = 252      # rolling window W_t, trading days
LAGS = 3          # k lagged period returns in the state
TAU2 = 0.001**2   # prior variance of a daily mean around its sector mean (Alg. 3, fixed exogenously)
H = 21            # trading days per period: daily moments -> monthly
CACHE = Path(__file__).parent / "data" / "panel.npz"


def build():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    adj, raw, dv = load()
    daily = adj[TICKERS].pct_change(fill_method=None)
    ok = tradable(raw, dv)
    me = month_ends(adj.index)
    monthly = adj.loc[me].pct_change(fill_method=None)
    pos = adj.index.get_indexer(me)
    T, N = len(me) - 1, len(TICKERS)  # last month-end (2025-08-13) has no next period
    out = {k: np.zeros(s) for k, s in dict(S=(T, N, N), mu=(T, N), vol=(T, N), lags=(T, N, LAGS), R=(T, N),
                                           w_bhrp=(T, N)).items()}
    out["act"] = np.zeros((T, N), bool)
    out["bench"] = monthly[ETFS].shift(-1).mean(axis=1).to_numpy()[:T]  # equal-weight ETFs, held over [t, t+1]
    for t in range(T):
        if pos[t] < WINDOW:
            continue
        win = daily.iloc[pos[t] - WINDOW + 1: pos[t] + 1].to_numpy()
        lag = monthly[TICKERS].iloc[max(t - LAGS + 1, 0): t + 1].to_numpy()[::-1].T
        act = ok.iloc[pos[t]].to_numpy() & ~np.isnan(win).any(0) & (lag.shape[1] == LAGS) & ~np.isnan(lag).any(1)
        assert set(SECTOR_OF[act]) == set(SECTOR_OF), f"{me[t].date()}: a sector has no tradable name"
        a = np.flatnonzero(act)
        mu, S = posterior(win[:, a], SECTOR_OF[a], TAU2)
        S *= H
        out["act"][t] = act
        out["mu"][t, a] = mu * H
        out["S"][t][np.ix_(a, a)] = S
        out["vol"][t, a] = np.sqrt(np.diag(S))
        out["lags"][t, a] = lag[a]
        out["R"][t, a] = monthly[TICKERS].iloc[t + 1].to_numpy()[a]
        assert np.isfinite(out["R"][t, a]).all()
        out["w_bhrp"][t, a] = bhrp(S, SECTOR_OF[a])
    first = np.argmax(out["act"].any(1))
    out = {k: v[first:] for k, v in out.items()}
    out["dates"] = np.array(me[first:T])           # decision dates t
    out["ret_dates"] = np.array(me[first + 1:T + 1])  # end of the period each return is realised over
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    P = build()
    d = pd.DatetimeIndex(P["dates"])
    print(len(d), "decisions", d[0].date(), "->", d[-1].date(), "| last return ends", pd.Timestamp(P["ret_dates"][-1]).date())
    print("active names per date: min", P["act"].sum(1).min(), "max", P["act"].sum(1).max())
