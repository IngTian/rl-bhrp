"""Numerical checks of the paper's propositions and of bhrp.py. Run: uv run python test_bhrp.py"""
import numpy as np

from bhrp import bhrp, factorized_softmax, penalty, posterior, rc, risk_budget

rng = np.random.default_rng(0)
sec = np.repeat(np.arange(4), [5, 2, 3, 2])  # unequal sector sizes, as in Table 1
N, G = len(sec), 4
F = np.c_[rng.uniform(0.5, 1.5, N), 0.8 * np.eye(G)[sec]]  # market + sector factors, equity-like
S = (F @ F.T + np.diag(rng.uniform(0.5, 2, N))) * 1e-3


def test_bijection_and_variance_decomposition():  # Prop. 3.1.8, Lemma 3.1.10, Thm. 3.1.11
    W = rng.dirichlet(np.ones(G))
    eta = np.zeros((N, G))
    for g in range(G):
        eta[sec == g, g] = rng.dirichlet(np.ones((sec == g).sum()))
    w = eta @ W
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert np.allclose(np.bincount(sec, w), W)
    assert np.allclose(w / W[sec], eta.sum(1))
    H = eta.T @ S @ eta
    assert np.isclose(rc(w, S).sum(), w @ S @ w)
    assert np.isclose(w @ S @ w, W @ H @ W)  # eq. (2)
    assert np.allclose(np.bincount(sec, rc(w, S)), W * (H @ W))  # eq. (3)


def test_risk_budget():  # Alg. 2/5, Appendix A
    b = rng.dirichlet(np.ones(N))
    x = risk_budget(S, b)
    assert np.isclose(x.sum(), 1) and np.allclose(rc(x, S) / (x @ S @ x), b, atol=1e-10)
    d = np.diag(S)  # diagonal special case: x ∝ sqrt(b)/sigma
    assert np.allclose(risk_budget(np.diag(d), b), np.sqrt(b / d) / np.sqrt(b / d).sum())


def test_two_level_fixed_point_is_risk_budgeting():  # Alg. 1 == RB with b_i = 1/(G n_g)
    w = bhrp(S, sec)
    r = rc(w, S) / (w @ S @ w)
    n = np.bincount(sec)
    assert np.allclose(r, 1 / (G * n[sec]), atol=1e-8)
    assert np.allclose(w, risk_budget(S, 1 / (G * n[sec])), atol=1e-8)
    assert penalty(w[None], S, sec, 0.5, "sector")[0] < 1e-18  # sector-centred penalty vanishes at BHRP
    assert penalty(w[None], S, sec, 0.5, "global")[0] > 1e-12  # the paper's global P_N one does not


def test_paper_literal_reciprocal_update_cycles():  # on a block-diagonal Sigma; why bhrp() damps with sqrt
    d = np.array([1.0, 4.0, 9.0])
    e = np.ones(3) / 3
    seen = []
    for _ in range(6):
        e = (1 / (d * e)) / (1 / (d * e)).sum()
        seen.append(e)
    assert np.allclose(seen[0], seen[2]) and np.allclose(seen[1], seen[3]) and not np.allclose(seen[0], seen[1])


def test_penalty_closed_form_gradient():  # Sec. 3.2.7 (global centering, as written)
    w = rng.dirichlet(np.ones(N))
    J = np.diag(S @ w) + np.diag(w) @ S
    P_N, P_G = np.eye(N) - 1 / N, np.eye(G) - 1 / G
    Sg = np.eye(G)[sec].T
    r = rc(w, S)
    grad = 0.3 * (2 / N) * J.T @ P_N @ r + 0.7 * (2 / G) * J.T @ Sg.T @ P_G @ Sg @ r
    eps = 1e-7
    fd = [(penalty((w + eps * e)[None], S, sec, 0.3)[0] - penalty((w - eps * e)[None], S, sec, 0.3)[0]) / (2 * eps)
          for e in np.eye(N)]
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-14)


def test_factorized_softmax():  # Def. 3.2.4, 3.2.5 (representability + Jacobian)
    w = rng.dirichlet(np.ones(N))
    W = np.bincount(sec, w)
    assert np.allclose(factorized_softmax(np.log(W)[None] + 3.0, np.log(w / W[sec])[None], sec)[0], w)
    psi, phi = rng.normal(size=(1, G)), rng.normal(size=(1, N))
    w = factorized_softmax(psi, phi, sec)[0]
    eta = w / np.bincount(sec, w)[sec]
    eps = 1e-6
    for g in range(G):
        d = np.zeros((1, G)); d[0, g] = eps
        fd = (factorized_softmax(psi + d, phi, sec) - factorized_softmax(psi - d, phi, sec))[0] / (2 * eps)
        assert np.allclose(fd, w * ((sec == g) - np.bincount(sec, w)[g]), atol=1e-8)
    j = 4  # an asset in sector 0
    d = np.zeros((1, N)); d[0, j] = eps
    fd = (factorized_softmax(psi, phi + d, sec) - factorized_softmax(psi, phi - d, sec))[0] / (2 * eps)
    h = sec[j]
    analytic = np.where(sec == h, np.bincount(sec, w)[h] * eta * ((np.arange(N) == j) - eta[j]), 0)
    assert np.allclose(fd, analytic, atol=1e-8)
    masked = phi.copy(); masked[0, 0] = -np.inf  # masking: weight 0, still on the simplex
    wm = factorized_softmax(psi, masked, sec)[0]
    assert wm[0] == 0 and np.isclose(wm.sum(), 1)


def test_posterior_shrinks_to_sector_mean():  # Alg. 3
    R = rng.normal(0.001, 0.02, size=(252, N)) + rng.normal(0, 0.002, N)
    mu, Sig = posterior(R, sec, tau2=1e-6)
    mu_flat, _ = posterior(R, sec, tau2=1e6)
    mu_g = np.array([R.mean(0)[sec == g].mean() for g in range(G)])[sec]
    assert np.allclose(mu_flat, R.mean(0), atol=1e-9)
    assert (np.abs(mu - mu_g) <= np.abs(R.mean(0) - mu_g) + 1e-15).all()
    assert np.allclose(Sig, Sig.T) and np.linalg.eigvalsh(Sig).min() > 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
