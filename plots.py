"""Figures 1-3 (Sec. 4) from results/<run>/. uv run python plots.py flat-global"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RL, BHRP, BENCH = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "legend.frameon": False})


def end_labels(ax, x, series, gap=11):
    """Direct labels at line ends, pushed apart vertically (in points) so they never collide."""
    ax.figure.canvas.draw()
    to_pt = lambda v: ax.transData.transform((0, v))[1] * 72 / ax.figure.dpi
    items = sorted(((to_pt(y[-1]), y[-1], t, c) for y, t, c in series), reverse=True)
    pos = []
    for p, *_ in items:
        pos.append(min(p, pos[-1] - gap) if pos else p)
    for (p, v, t, c), q in zip(items, pos):
        ax.plot(x[-1], v, "o", ms=5, color=c, mec=SURFACE, mew=2)
        ax.annotate(t, (x[-1], v), xytext=(8, q - p), textcoords="offset points", va="center", color=INK, fontsize=8)


def main(run="flat-global"):
    d = Path("results") / run
    ret = pd.read_csv(d / "returns.csv", index_col=0, parse_dates=True)
    med = pd.read_json(d / "summary.json", typ="series")["median_seed"]
    z = np.load(d / "weights.npz", allow_pickle=True)
    rl = ret.filter(like="rl_seed")
    wealth = lambda r: np.r_[1, np.cumprod(1 + r.to_numpy())]
    x = np.r_[[ret.index[0] - pd.offsets.MonthEnd(1)], ret.index]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    W = np.array([wealth(rl[c]) for c in rl])
    ax.fill_between(x, W.min(0), W.max(0), color=RL, alpha=0.15, lw=0, label=f"RL-BHRP, range over {len(rl.columns)} seeds")
    lines = [(W[med], "RL-BHRP", RL), (wealth(ret["bhrp"]), "BHRP", BHRP), (wealth(ret["benchmark"]), "Benchmark", BENCH)]
    for y, lab, c in lines:
        ax.plot(x, y, color=c, lw=2, label=lab + (" (median seed)" if c == RL else ""))
    ax.set_ylabel("Cumulative wealth")
    ax.set_title(f"Fig. 1 — test-period cumulative wealth (actor-centering: {run})", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=200))
    end_labels(ax, x, lines)
    fig.tight_layout()
    fig.savefig(d / "fig1_wealth.png", dpi=160)

    w = z["rl"][med]
    tick = list(z["tickers"])
    top = np.argsort(-w.mean(0))[:15]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(w[:, top].T, aspect="auto", cmap=BLUES, interpolation="nearest",
                   extent=[0, len(w), 15, 0])
    ax.set_yticks(np.arange(15) + 0.5, [tick[i] for i in top])
    yrs = pd.DatetimeIndex(z["dates"])
    jan = np.flatnonzero(yrs.month == 1)
    ax.set_xticks(jan, yrs[jan].year)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="weight", fraction=0.03, pad=0.02)
    ax.set_title("Fig. 2 — RL-BHRP weights, top 15 names by average weight (median seed)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "fig2_heatmap.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    lines = [(ws[:, np.argsort(-ws.mean(0))[:15]].sum(1), lab, c) for ws, lab, c in [(w, "RL-BHRP", RL), (z["bhrp"], "BHRP", BHRP)]]
    for s, lab, c in lines:
        ax.plot(yrs, s, color=c, lw=2, label=lab)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-15 share of capital")
    ax.set_title("Fig. 3 — concentration: weight in each strategy's own top 15 names", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(yrs[0], yrs[-1] + pd.Timedelta(days=200))
    end_labels(ax, yrs, lines)
    fig.tight_layout()
    fig.savefig(d / "fig3_top15_share.png", dpi=160)


if __name__ == "__main__":
    main(*sys.argv[1:])
