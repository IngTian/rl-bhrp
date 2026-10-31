"""Universe (Table 1), Yahoo download + cache, month-end grid, quarterly tradability screen (Def. 3.1.1)."""
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = {  # Table 1: sector ETF -> constituents
    "XLK": ["AAPL", "MSFT", "NVDA", "ADBE", "ORCL", "CSCO", "INTC", "AVGO", "AMD", "AMAT"],
    "XLC": ["GOOGL", "META", "T", "VZ"],
    "XLY": ["AMZN", "HD", "NKE", "LOW"],
    "XLP": ["PG", "KO", "PEP", "COST", "PM"],
    "XLF": ["JPM", "BAC", "WFC", "V", "MA"],
    "XLE": ["XOM", "CVX"],
    "XLV": ["LLY", "UNH", "ABBV", "TMO", "PFE", "MRK"],
    "XLI": ["CAT", "BA", "MMM", "GE", "HON"],
    "XLU": ["DUK", "SO", "D"],
    "XLRE": ["PLD", "SPG"],
    "XLB": ["LIN", "APD"],
}
ETFS = list(SECTORS)
TICKERS = [t for v in SECTORS.values() for t in v]
SECTOR_OF = np.array([g for g, v in enumerate(SECTORS.values()) for _ in v])  # g: I -> G
START, END = "2012-01-01", "2025-08-14"  # yfinance end is exclusive; last bar 2025-08-13
CACHE = Path(__file__).parent / "data" / "yahoo.pkl"


def load():
    """Daily adjusted close (returns), raw close (price screen), dollar volume (ADV screen)."""
    if not CACHE.exists():
        import yfinance as yf
        df = yf.download(TICKERS + ETFS, start=START, end=END, auto_adjust=False, actions=True,
                         progress=False, threads=True)
        CACHE.parent.mkdir(exist_ok=True)
        df.to_pickle(CACHE)
    df = pd.read_pickle(CACHE)
    adj, close, vol = df["Adj Close"], df["Close"], df["Volume"]
    # Yahoo "Close" is split-adjusted; undo later splits so the $3 screen sees the price traded that day.
    splits = df["Stock Splits"].replace(0, 1).fillna(1)
    future = splits[::-1].cumprod()[::-1].shift(-1).fillna(1)
    raw_close = close * future
    return adj, raw_close, close * vol  # split-adjusted close * split-adjusted volume = raw dollar volume


def month_ends(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def tradable(raw_close, dollar_vol, adv_min=10e6, px_min=3.0):
    """Def. 3.1.1: screened at each quarter-end, frozen for the following quarter. Returns daily bool mask."""
    adv = dollar_vol[TICKERS].rolling(20).mean()
    ok = (adv >= adv_min) & (raw_close[TICKERS] >= px_min)
    qe = month_ends(ok.index)
    qe = qe[qe.month % 3 == 0]
    return ok.loc[qe].reindex(ok.index).shift(1).ffill().fillna(False).astype(bool)


if __name__ == "__main__":
    adj, raw, dv = load()
    print(adj.shape, adj.index[0].date(), adj.index[-1].date())
    print("first valid:", adj[TICKERS].apply(lambda s: s.first_valid_index().date()).sort_values().tail(4).to_dict())
    m = tradable(raw, dv)
    print("names failing screen at some month-end:", sorted(set(m.columns[(~m.loc[month_ends(m.index)][m.index[300]:]).any()])))
    print("NVDA raw close 2012-06:", raw.loc["2012-06", "NVDA"].iloc[0].round(2), "AMD:", raw.loc["2015-10", "AMD"].iloc[0].round(2))
