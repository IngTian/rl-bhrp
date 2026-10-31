"""RL layer (Sec. 3.2): MDP over the monthly panel, factorized-softmax Gaussian policy, average-reward PPO (Alg. 4)."""
import numpy as np
import torch
import torch.nn as nn

from bhrp import factorized_softmax, penalty
from data import SECTOR_OF

G, N = SECTOR_OF.max() + 1, len(SECTOR_OF)


class Env:
    """Replays panel dates t0..t1-1 for B parallel rollouts. State (Def. 3.2.2): k lagged returns,
    mu_hat, diag(Sigma) (as vol), w_{t-1}. Action: logit residuals added to the BHRP logits at t,
    so a = 0 reproduces BHRP. Reward: eq. (4)."""

    def __init__(self, P, t0, t1, *, c, lam, alpha, centering, stats):
        self.P, self.t0, self.t1 = P, t0, t1
        self.c, self.lam, self.alpha, self.centering = c, lam, alpha, centering
        self.m, self.s = stats
        W = np.stack([np.bincount(SECTOR_OF, w, G) for w in P["w_bhrp"]])
        with np.errstate(divide="ignore"):
            self.psi0, self.phi0 = np.log(W), np.log(P["w_bhrp"] / W[:, SECTOR_OF])  # -inf masks inactive

    def reset(self, B):
        self.t = self.t0
        a = self.P["act"][self.t0]
        self.w = np.tile(a / a.sum(), (B, 1))  # w_{-1} = equal weight (Alg. 4)
        return self.obs()

    def obs(self):
        x = (features(self.P, self.t) - self.m) / self.s * self.P["act"][self.t][:, None]
        return np.concatenate([np.broadcast_to(x.ravel(), (len(self.w), x.size)), self.w * N], 1)

    def step(self, a):
        P, t = self.P, self.t
        w = factorized_softmax(self.psi0[t] + a[:, :G], self.phi0[t] + a[:, G:], SECTOR_OF)
        act = P["act"][t]
        net = w @ P["R"][t] - self.c * np.abs(w - self.w).sum(1)
        pen = penalty(w[:, act], P["S"][t][np.ix_(act, act)], SECTOR_OF[act], self.alpha, self.centering)
        self.w, self.t = w, t + 1
        return net - self.lam * pen, net, w

    @property
    def done(self):
        return self.t == self.t1


def features(P, t):
    return np.concatenate([P["lags"][t], P["mu"][t][:, None], P["vol"][t][:, None]], 1)


def feature_stats(P, t0, t1):
    """Per-feature z-score stats over active (asset, date) pairs in the training dates only."""
    x = np.concatenate([features(P, t)[P["act"][t]] for t in range(t0, t1)])
    return x.mean(0), x.std(0) + 1e-12


class ActorCritic(nn.Module):
    def __init__(self, d, m, h=64, std0=0.3):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, m))
        self.v = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        nn.init.zeros_(self.pi[-1].weight)  # initial mean action 0 == BHRP
        nn.init.zeros_(self.pi[-1].bias)
        self.log_std = nn.Parameter(torch.full((m,), float(np.log(std0))))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())


def train(P, t0, t1, *, seed, env_kw, iters=300, B=16, epochs=10, mb=256, lr=3e-4, clip=0.2, ent=1e-3,
          gae=0.95, scale=100.0, log_every=0, on_iter=None):
    """Alg. 4. Average-reward GAE: delta_t = r_t - rho + V(s_{t+1}) - V(s_t), rho = batch mean reward."""
    torch.manual_seed(seed)
    env = Env(P, t0, t1, **env_kw, stats=feature_stats(P, t0, t1))
    net = ActorCritic(env.reset(1).shape[1], G + N)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    f32 = lambda x: torch.as_tensor(x, dtype=torch.float32)
    for it in range(iters):
        o, O, A, LP, R = env.reset(B), [], [], [], []
        while not env.done:
            with torch.no_grad():
                d = net.dist(f32(o))
                a = d.sample()
            r, _, _ = env.step(a.double().numpy())
            O.append(f32(o)), A.append(a), LP.append(d.log_prob(a).sum(-1)), R.append(f32(r * scale))
            o = env.obs()  # at t1 this is s_T, which bootstraps the truncated episode
        O, A, LP, R = torch.stack(O), torch.stack(A), torch.stack(LP), torch.stack(R)
        with torch.no_grad():
            V = net.v(torch.cat([O, f32(o)[None]])).squeeze(-1)
        delta = R - R.mean() + V[1:] - V[:-1]
        adv = torch.zeros_like(R)
        g = torch.zeros(B)
        for k in reversed(range(len(R))):
            g = delta[k] + gae * g
            adv[k] = g
        ret = (adv + V[:-1]).flatten()
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).flatten()
        O, A, LP = O.flatten(0, 1), A.flatten(0, 1), LP.flatten()
        for _ in range(epochs):
            for j in torch.randperm(len(adv)).split(mb):
                d = net.dist(O[j])
                ratio = (d.log_prob(A[j]).sum(-1) - LP[j]).exp()
                pi_loss = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
                v_loss = (net.v(O[j]).squeeze(-1) - ret[j]).pow(2).mean()
                loss = pi_loss + 0.5 * v_loss - ent * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if log_every and it % log_every == 0:
            print(f"seed {seed} it {it:4d} mean reward {R.mean().item() / scale:+.5f} std {net.log_std.exp().mean().item():.3f}")
        if on_iter:
            on_iter(it + 1, net, (env.m, env.s))
    return net, env.m, env.s


def evaluate(net, stats, P, t0, t1, env_kw):
    """Deterministic (mean-action) rollout. Returns net-of-cost returns (T,) and weights (T, N)."""
    env = Env(P, t0, t1, **env_kw, stats=stats)
    o, rets, ws = env.reset(1), [], []
    while not env.done:
        with torch.no_grad():
            a = net.pi(torch.as_tensor(o, dtype=torch.float32)).double().numpy()
        _, r, w = env.step(a)
        rets.append(r[0]), ws.append(w[0])
        if not env.done:
            o = env.obs()
    return np.array(rets), np.array(ws)
