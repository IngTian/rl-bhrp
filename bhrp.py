"""BHRP math: Bayesian hierarchical update (Alg. 3), risk contributions + hierarchical RP penalty
(Def. 3.1.10, 3.2.3), log-Newton risk budgeting (Alg. 2/5), two-level fixed point (Alg. 1),
factorized softmax (Def. 3.2.4)."""
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf


def posterior(R, sec, tau2):
    """Alg. 3. R: (n, N) window of returns, sec: sector id per column. Returns (mu_hat, Sigma)."""
    n = len(R)
    rbar, s2 = R.mean(0), R.var(0, ddof=1)
    mu_g = pd.Series(rbar).groupby(sec).transform("mean").to_numpy()
    mu = (mu_g / tau2 + n * rbar / s2) / (1 / tau2 + n / s2)
    return mu, ledoit_wolf(R)[0]


def rc(w, S):
    """Asset risk contributions RC_i = w_i (S w)_i; batched over leading dims of w."""
    return w * (w @ S)


def _groups(sec):
    """Start offset and contiguous 0..G'-1 id of each sector run (sec must be sorted)."""
    new = np.r_[True, sec[1:] != sec[:-1]]
    return np.flatnonzero(new), np.cumsum(new) - 1


def penalty(w, S, sec, alpha, centering="global"):
    """Def. 3.2.3: alpha*V_within + (1-alpha)*V_across. w: (B, N), sec sorted & contiguous.
    centering="global" is the paper's P_N = I - 11'/N; "sector" centres RC within each sector instead,
    so both terms vanish exactly at the two-level RP portfolio (the paper's Alg. 1 RCVAR_within)."""
    r = rc(w, S)
    starts, gid = _groups(sec)
    r_sec = np.add.reduceat(r, starts, axis=1)  # S @ RC
    if centering == "global":
        v_within = r.var(1)
    else:
        v_within = ((r - (r_sec / np.bincount(gid))[:, gid]) ** 2).mean(1)
    return alpha * v_within + (1 - alpha) * r_sec.var(1)


def risk_budget(S, b, tol=1e-12, max_iter=100):
    """Alg. 2/5: log-domain Newton on x o (S x) = nu b with 1'x = 1. Unique for S > 0, b > 0."""
    n = len(b)
    y, c = np.zeros(n), 0.0

    def resid(y, c):
        u = np.exp(y)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.log(S @ u) + y - np.log(b) - c
        return r, u.sum() - 1

    r, s = resid(y, c)
    # The paper's start y=0 needs S 1 > 0 (log z); true for positively correlated long-only equity books.
    assert np.isfinite(r).all(), "risk_budget: S @ 1 has a non-positive entry; y=0 start undefined"
    for _ in range(max_iter):
        if max(np.abs(r).max(), abs(s)) <= tol:
            break
        u = np.exp(y)
        J = np.eye(n) + S * u / (S @ u)[:, None]  # I + Z^-1 S U, eq. (9)
        A = np.block([[J, -np.ones((n, 1))], [u[None, :], np.zeros((1, 1))]])
        d = np.linalg.solve(A, -np.r_[r, s])  # eq. (10)
        f0, a = r @ r + s * s, 1.0
        while True:  # Armijo backtracking on ||r||^2 + s^2; non-finite (S u <= 0) is rejected
            r1, s1 = resid(y + a * d[:n], c + a * d[n])
            f1 = r1 @ r1 + s1 * s1
            if (np.isfinite(f1) and f1 <= (1 - 1e-4 * a) * f0) or a < 1e-10:
                break
            a /= 2
        y, c, r, s = y + a * d[:n], c + a * d[n], r1, s1
    else:
        raise RuntimeError(f"risk_budget: no convergence, |r|={np.abs(r).max():.2e}")
    return np.exp(y)


def bhrp(S, sec, tol=1e-10, max_iter=1000):
    """Alg. 1: two-level fixed point -> equal RC across sectors and within each sector.
    Sector step is Alg. 2. Within-sector step uses eta <- sqrt(eta / m), same fixed point as the
    paper's eta <- 1/m. The literal update is a period-2 map on block-diagonal Sigma (uniform <->
    1/sigma^2); on real equity covariances the cross-sector terms dominate m and it converges to the
    same weights, so the damping is a safeguard, not a correction."""
    N, G = len(sec), sec.max() + 1
    eta = np.zeros((N, G))
    eta[np.arange(N), sec] = 1
    eta /= eta.sum(0)  # column g holds eta^(g) on I_g
    W = np.full(G, 1 / G)
    w = eta @ W
    for _ in range(max_iter):
        for g in range(G):  # Gauss-Seidel over sectors
            ix = sec == g
            m = S[ix] @ w  # m^(g)(W, eta) = (S w) restricted to I_g
            e = np.sqrt(eta[ix, g] / np.maximum(m, 1e-12))
            eta[ix, g] = e / e.sum()
            w[ix] = W[g] * eta[ix, g]
        W = risk_budget(eta.T @ S @ eta, np.full(G, 1 / G))  # composite covariance, eq. (1)
        w_new = eta @ W
        done = np.abs(w_new - w).sum() <= tol
        w = w_new
        if done:
            break
    return w


def factorized_softmax(psi, phi, sec):
    """Def. 3.2.4: w_i = softmax(psi)_{g(i)} * softmax_{I_g}(phi)_i. psi: (B, G), phi: (B, N);
    phi = -inf masks an asset out. sec sorted & contiguous, every sector with >= 1 finite logit."""
    starts, gid = _groups(sec)
    W = np.exp(psi - psi.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    e = np.exp(phi - np.maximum.reduceat(phi, starts, axis=1)[:, gid])
    return W[:, gid] * e / np.add.reduceat(e, starts, axis=1)[:, gid]
