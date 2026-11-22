"""Steering cost for natural-policy-gradient agents in two-action coordination games.
Agents sit on a graph. x_i = probability of the target action H, theta_i = logit of x_i.
Game advantage of H over G for agent i:  g_i = c * (m_i - xs),  m_i = mean of x over neighbours.
Update (Definition 4.1 of Huang et al. 2025, two actions, payment v_i >= 0 on H only):
    theta_i <- theta_i + alpha * (g_i + v_i - lam * theta_i)        (lam = 0 is exact NPG)
Steering cost of one step (paper): sum_i x_i * v_i.   alpha-weighted cost (released code): alpha times that.
"""
import numpy as np
from numba import njit

def sig(t): return 1.0/(1.0+np.exp(-t))
def L(t): return np.logaddexp(0.0, t)          # L(theta) = -log(1-x)
def logit(x): return np.log(x/(1.0-x))

@njit(cache=True)
def neigh_mean(x, kind, m):
    N = x.shape[0]
    if kind == 0:                               # complete graph
        s = x.sum()
        for i in range(N): m[i] = (s - x[i])/(N-1)
    else:                                       # ring, nearest neighbours
        for i in range(N): m[i] = 0.5*(x[(i+1) % N] + x[(i-1) % N])

@njit(cache=True)
def _drift(th, kind, c, xs, x, m, out):
    N = th.shape[0]
    for i in range(N): x[i] = 1.0/(1.0+np.exp(-th[i]))
    neigh_mean(x, kind, m)
    for i in range(N): out[i] = c*(m[i]-xs)

@njit(cache=True)
def basin(th0, kind, c, xs, dt=0.05, tmax=20000000):
    """1 if the unsteered dynamics from logits th0 converge to all-H, else 0.
    Integration: classical Runge-Kutta, step dt.
    Exits use monotonicity: all m_i > xs implies convergence to all-H; all m_i < xs implies all-G.
    On the ring, two adjacent agents with x > 2*xs also imply all-H (Proposition 3),
    and two adjacent agents with x < 2*xs-1 can never rise, so all-H is not reached (proof of Proposition 4)."""
    N = th0.shape[0]; th = th0.copy(); x = np.empty(N); m = np.empty(N)
    k1 = np.empty(N); k2 = np.empty(N); k3 = np.empty(N); k4 = np.empty(N); tmp = np.empty(N)
    for t in range(tmax):
        _drift(th, kind, c, xs, x, m, k1)
        up = True; down = True
        for i in range(N):
            if m[i] <= xs: up = False
            if m[i] >= xs: down = False
        if up: return 1
        if down: return 0
        if kind == 1 and xs < 0.5:
            for i in range(N):
                if x[i] > 2*xs and x[(i+1) % N] > 2*xs: return 1
        if kind == 1 and xs > 0.5:
            for i in range(N):
                if x[i] < 2*xs-1 and x[(i+1) % N] < 2*xs-1: return 0
        for i in range(N): tmp[i] = th[i]+0.5*dt*k1[i]
        _drift(tmp, kind, c, xs, x, m, k2)
        for i in range(N): tmp[i] = th[i]+0.5*dt*k2[i]
        _drift(tmp, kind, c, xs, x, m, k3)
        for i in range(N): tmp[i] = th[i]+dt*k3[i]
        _drift(tmp, kind, c, xs, x, m, k4)
        for i in range(N): th[i] += dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0
    return -1

@njit(cache=True)
def _drift_decay(th, kind, c, xs, lam, x, m, out):
    N = th.shape[0]
    for i in range(N): x[i] = 1.0/(1.0+np.exp(-th[i]))
    neigh_mean(x, kind, m)
    for i in range(N): out[i] = c*(m[i]-xs) - lam*th[i]

@njit(cache=True)
def run_free(th0, kind, c, xs, lam, dt, tmax, level):
    """Unsteered run with logit decay lam (lam = 0: no decay). Classical Runge-Kutta, step dt.
    Returns the first time each agent's x exceeds `level` (-1 if never). The crossing time is interpolated
    linearly inside the step."""
    N = th0.shape[0]; th = th0.copy(); x = np.empty(N); m = np.empty(N); new = np.empty(N)
    k1 = np.empty(N); k2 = np.empty(N); k3 = np.empty(N); k4 = np.empty(N); tmp = np.empty(N)
    ll = np.log(level/(1.0-level)); first = -np.ones(N); left = N
    for i in range(N):
        if th[i] > ll: first[i] = 0.0; left -= 1
    for t in range(tmax):
        if left == 0: break
        _drift_decay(th, kind, c, xs, lam, x, m, k1)
        for i in range(N): tmp[i] = th[i]+0.5*dt*k1[i]
        _drift_decay(tmp, kind, c, xs, lam, x, m, k2)
        for i in range(N): tmp[i] = th[i]+0.5*dt*k2[i]
        _drift_decay(tmp, kind, c, xs, lam, x, m, k3)
        for i in range(N): tmp[i] = th[i]+dt*k3[i]
        _drift_decay(tmp, kind, c, xs, lam, x, m, k4)
        for i in range(N):
            new[i] = th[i] + dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0
            if first[i] < 0 and new[i] > ll:
                first[i] = (t + (ll-th[i])/(new[i]-th[i]))*dt; left -= 1
            th[i] = new[i]
    return first

@njit(cache=True)
def steer_sim(th0, target, kind, c, xs, alpha, U, T):
    """Discrete-time steering with the update and cost of Huang et al. (2025): pay U on H to agent i while x_i < target_i.
    Returns final logits, cost sum x*v, regret term sum x*(-g) (called work), step term sum KL (called D), steps used."""
    N = th0.shape[0]; th = th0.copy(); x = np.empty(N); m = np.empty(N)
    cost = 0.0; work = 0.0; D = 0.0; steps = 0
    for t in range(T):
        for i in range(N): x[i] = 1.0/(1.0+np.exp(-th[i]))
        neigh_mean(x, kind, m)
        active = False
        for i in range(N):
            if x[i] < target[i]: active = True
        if not active: break
        steps += 1
        for i in range(N):
            gi = c*(m[i]-xs); v = U if x[i] < target[i] else 0.0
            s = alpha*(gi+v)
            cost += x[i]*v; work += -x[i]*gi
            D += np.log(1.0 - x[i] + x[i]*np.exp(s)) - x[i]*s
            th[i] += s
    return th, cost, work, D, steps

def scale_min(d, th0, kind, c, xs, iters=30):
    """Smallest s such that th0 + s*d lies in the basin of all-H (d >= 0)."""
    lo, hi = 0.0, 1.0
    while basin(th0 + hi*d, kind, c, xs) != 1:
        hi *= 2
        if hi > 64: return None
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if basin(th0 + mid*d, kind, c, xs) == 1: hi = mid
        else: lo = mid
    return hi

def potential(th, th0): return float(np.sum(L(th) - L(th0)))     # energy difference E(th) - E(th0) of the note

def search(N, kind, c, xs, x0, iters=800, seed=0, init=None):
    """Random local search for the cheapest state in the basin of all-H. It rescales and reshapes the
    starting pattern; it does not add agents to it."""
    rng = np.random.default_rng(seed); t0 = logit(x0); th0 = np.full(N, t0)
    d = np.ones(N) if init is None else init.astype(float).copy()
    s = scale_min(d, th0, kind, c, xs)
    if s is None: return np.inf, None
    bd = s*d; best = potential(th0+bd, th0); step = 0.5
    for it in range(iters):
        nd = np.maximum(bd*np.exp(step*rng.normal(0, 1, N)*(rng.random(N) < 0.5)), 0)
        if rng.random() < 0.1: nd[rng.integers(N)] = 0
        if nd.max() <= 0: continue
        s = scale_min(nd, th0, kind, c, xs)
        if s is None: continue
        cc = potential(th0 + s*nd, th0)
        if cc < best - 1e-10: best = cc; bd = s*nd
        if it % 200 == 199: step *= 0.6
    return best, sig(th0 + bd)

def closed_form_complete(U, x0, xs, c, N):
    """alpha-weighted cost of symmetric steering at constant payment U from x0 to xs (continuous time)."""
    a = U - c*xs
    return N*U/(a+c)*np.log((a+c*xs)*(1-x0)/((a+c*x0)*(1-xs)))
