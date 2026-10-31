# RL-BHRP

Code reproducing Kang & Tian (2025), *Optimal Portfolio Construction — A Reinforcement Learning Embedded Bayesian
Hierarchical Risk Parity (RL-BHRP) Approach*, [arXiv:2508.11856](https://arxiv.org/abs/2508.11856).

## Run

```sh
uv sync
uv run python test_bhrp.py                                  # numerical checks of the paper's propositions
uv run python run.py --actor shared --centering global      # tune, 10 seeds, Table 2 -> results/shared-global/
uv run python run.py --actor flat   --centering global      # (also: --centering sector)
uv run python plots.py shared-global                        # Figs. 1-3
uv run python sensitivity.py                                # test outcomes over actor x centering x lambda x iters x 10 seeds
```

The first run downloads Yahoo Finance data (2012-01-01 .. 2025-08-13) into `data/` (not committed).

| file | paper |
|---|---|
| `data.py` | Table 1 universe; Def. 3.1.1 quarterly screen (ADV20 >= $10M, price >= $3) |
| `bhrp.py` | Alg. 3 posterior; Def. 3.1.10 / 3.2.3 risk contributions + penalty; Alg. 2/5 log-Newton; Alg. 1 two-level fixed point; Def. 3.2.4 factorized softmax |
| `panel.py` | monthly decision panel: posterior, Sigma, BHRP weights, next-period returns |
| `rl.py` | Sec. 3.2 MDP, reward eq. (4), actors, average-reward PPO (Alg. 4) |
| `run.py` | Sec. 4 backtest, Table 2 metrics, Sec. 4.2 weight diagnostics |
| `test_bhrp.py` | Prop. 3.1.8, Lemma 3.1.10, Thm. 3.1.11, Alg. 1 = risk budgeting with b_i = 1/(G n_g), App. A, the 3.2.5 Jacobian and 3.2.7 gradient |

## Results (test: Feb 2020 – Aug 2025, 67 months, 5 bps one-way; RL = 10 seeds, mean and range)

| | CAGR | Sharpe | Max DD | TE | IR | seeds > BHRP (CAGR / Sharpe) |
|---|---|---|---|---|---|---|
| Benchmark (EW 11 SPDR ETFs) | 12.33% | 0.763 | -18.32% | — | — | |
| BHRP | 13.36% | 0.846 | -19.12% | 3.58% | 0.220 | |
| RL-BHRP, shared actor | **14.55%** (14.23–14.88) | **0.891** (0.873–0.904) | -19.7% | 3.29% | 0.58 | 10/10 / 10/10 |
| RL-BHRP, flat actor | 14.30% (13.15–15.80) | 0.876 (0.813–0.953) | -19.3% | 5.07% | 0.33 | 9/10 / 6/10 |
| *paper RL-BHRP* | *15.16%* | *0.905* | *-20.33%* | *3.69%* | *0.687* | |

**Robustness** (`sensitivity.py`). This covers 640 runs over actor x centering x lambda in {1e5..1e8} x iterations
in {25..200} x 10 seeds, scored on the test period. It is a robustness report and was not used to select anything.
- **Shared actor:** RL-BHRP beats BHRP on CAGR and Sharpe for every seed in every cell with lambda <= 1e6, at every
  iteration count. The edge grows with training (lambda = 1e5, 200 iterations: 14.96% CAGR, Sharpe 0.911, TE 3.3%),
  and the seed sd is 0.1–0.3 pp. lambda = 1e8 pins the policy to BHRP.
- **Flat actor:** the edge is positive on average but noisier (seed sd up to 2.9 pp, TE up to 9%).

All seeds share one 67-month test path, so this is robustness to training randomness, not statistical significance:
an IR of 0.58 over 5.6 years is t ≈ 1.4 (the paper's 0.687 is t ≈ 1.6).

Penalty centering as written in Def. 3.2.3 (`global`). The within-sector reading (`--centering sector`) gives 14.54% / 0.890
with the shared actor and 13.62% / 0.848 with the flat one. The Benchmark and BHRP rows match the paper's Table 2 on all
15 metrics (BHRP to within 6e-4, Benchmark to within 1e-6). Per-metric tables: `results/<actor>-<centering>/table2.csv`;
per seed: `seeds.csv`; robustness grid: `results/sensitivity_summary.csv`.

## Implementation choices

Where the paper leaves a choice open, it was fixed as follows.

- **Rolling window: 756 trading days.** Not stated in the paper. With 756 days, the BHRP column of Table 2 is reproduced
  on all 15 rows; 252 or 504 days are visibly off. Daily moments are scaled x21 to the monthly horizon. The prior sd
  is tau = 0.1%/day, and the state uses k = 3 lags.
- **Metric conventions.** These were recovered by matching the paper's Benchmark column exactly: population std; drawdown
  peaks exclude the starting wealth 1.0 (so the Feb-2020 loss is not counted); Sortino denominator std(min(r, 0)).
  `run.py` also reports the standard drawdown and Sortino as extra rows. The paper's Benchmark VaR (-0.079889) is a typo
  for -0.079989 (its text says -8.00%).
- **Actor.** Not specified in the paper. Both actors output residual logits added to BHRP's own logits (log W_g,
  log eta_i|g). They are zero-initialised, so training starts exactly at BHRP. The stochastic policy is a diagonal
  Gaussian over the logits; evaluation uses the mean action.
  - `flat` is an MLP over the whole state (29.6k params). With 59 training months it can memorise dates.
  - `shared` is one small MLP applied to each asset's own row of the state (434 params). It outputs phi_i, and its
    second output is averaged over each sector to give psi_g.
- **Penalty centering.** Def. 3.2.3 and the 3.2.7 gradient centre V_within over all N assets (`global`). Alg. 1's
  RCVAR_within centres within each sector (`sector`), and only that version vanishes exactly at BHRP when sectors have
  unequal sizes. In practice the two give nearly identical results.
- **Alg. 1.** The within-sector update is damped to eta <- sqrt(eta/m), which has the same fixed point as the paper's
  eta <- 1/m; the sector step uses Alg. 2. The literal update is a period-2 map on a block-diagonal Sigma, but on these
  covariances it converges to the same weights. Alg. 5's start at y = 0 needs Sigma·1 > 0, which holds for equity
  covariances.
- **Tuning.** lambda in {1e5, 1e6, 1e7, 1e8} and the iteration count are picked by mean Sharpe over 3 seeds on the last
  24 training months (2017-12 .. 2019-11). alpha = 0.5. PPO uses 16 rollouts/iter, 10 epochs x 4 minibatches,
  lr 3e-4 and clip 0.2.
- **Screen and costs.** The $3 screen uses raw (split-reconstructed) close. The cost is c·||w_t − w_{t−1}||₁ on target
  weights, as in the paper. BHRP pays the same cost; the benchmark is cost-free.

## License

MIT
