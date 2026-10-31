# RL-BHRP — reproduction

A from-scratch reimplementation of Kang & Tian (2025), *Optimal Portfolio Construction — A Reinforcement
Learning Embedded Bayesian Hierarchical Risk Parity (RL-BHRP) Approach*, [arXiv:2508.11856](https://arxiv.org/abs/2508.11856).
The original code was lost; everything here is rebuilt from the paper's formulation.

> **Status: preliminary.** An independent review of the code against the paper is in progress; numbers below may change.

## Run

```sh
uv sync
uv run python test_bhrp.py            # numerical checks of the paper's propositions (Sec. 3.1, 3.2.5, 3.2.7, App. A)
uv run python run.py                  # tune on 2017-12..2019-11, 10 seeds, Table 2 -> results/global/
uv run python run.py --centering sector
uv run python plots.py global         # Figs. 1-3
uv run python sensitivity.py          # test outcomes over (centering, lambda, iterations) x 10 seeds
```

The first run downloads Yahoo Finance data (2012-01-01 .. 2025-08-13) into `data/` (not committed).

| file | paper |
|---|---|
| `data.py` | Table 1 universe, Def. 3.1.1 quarterly screen (ADV20 >= $10M, price >= $3) |
| `bhrp.py` | Alg. 3 posterior, Def. 3.1.10 / 3.2.3 risk contributions + penalty, Alg. 2/5 log-Newton, Alg. 1 two-level fixed point, Def. 3.2.4 factorized softmax |
| `panel.py` | monthly decision panel (posterior, Sigma, BHRP weights, next-period returns) |
| `rl.py` | Sec. 3.2 MDP, reward eq. (4), average-reward PPO (Alg. 4) |
| `run.py` | Sec. 4 backtest, Table 2 metrics, Sec. 4.2 weight diagnostics |

## Where the paper is silent or inconsistent

- **Alg. 1 within-sector step.** The literal update eta <- 1/m is a period-2 map (on a diagonal block it alternates between
  uniform and 1/sigma^2 and never converges). We use eta <- sqrt(eta/m), which has the same fixed point. The sector step uses Alg. 2.
  The two-level fixed point is exactly risk budgeting with b_i = 1/(G n_g) (checked in `test_bhrp.py`).
- **Alg. 5 start y = 0** requires Sigma·1 > 0, which holds for positively correlated equities but not for every PD Sigma.
- **Penalty centering.** Def. 3.2.3 (and the 3.2.7 gradient) centre V_within over all N assets, so the penalty does not vanish at BHRP
  when sectors have unequal sizes; Alg. 1's RCVAR_within centres within each sector. `--centering global` (default) is the
  formula as written, `--centering sector` the within-sector reading.
- **Actor.** Not specified. Ours outputs residual logits on top of BHRP's own logits (zero-initialised, so it starts at BHRP);
  diagonal Gaussian over logits; deterministic mean at evaluation.
- **Unreported hyperparameters.** lambda and iterations picked on the last 24 training months; alpha = 0.5; 252-day window,
  tau = 0.1%/day, k = 3 lags; PPO: 16 rollouts/iter, 10 epochs, lr 3e-4, clip 0.2.
- **Metric conventions** (recovered by matching the paper's Benchmark column exactly): population std; drawdown peaks exclude the
  starting wealth (this omits the Feb-2020 loss); Sortino denominator std(min(r, 0)). `run.py` reports these plus the
  standard drawdown and Sortino as extra rows.

## Results (test: Feb 2020 – Aug 2025, 67 months, 5 bps one-way)

| | CAGR | Sharpe | Max DD | TE | note |
|---|---|---|---|---|---|
| Benchmark (EW 11 SPDR ETFs) | 12.33% | 0.763 | -18.32% | — | identical to paper |
| BHRP | 13.64% | 0.868 | -17.80% | 3.94% | paper: 13.36%, 0.846 |
| RL-BHRP, global centering, 10 seeds | 12.74% (10.5–15.4) | 0.795 | -20.0% | 6.5% | 2/10 seeds beat BHRP CAGR |
| RL-BHRP, sector centering, 10 seeds | 14.36% (8.7–22.7) | 0.850 | -20.8% | 9.3% | 6/10 seeds beat BHRP CAGR |
| paper RL-BHRP | 15.16% | 0.905 | -20.33% | 3.69% | |

Full tables: `results/<centering>/table2.csv`; per-seed: `seeds.csv`; figures: `fig*.png`.
