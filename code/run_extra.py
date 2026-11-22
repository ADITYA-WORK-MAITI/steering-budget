"""Extra checks. Usage: python run_extra.py [holding|steps|feedback|harmonic|euclid|frontK ...]
holding : ring, x* < 1/2, pay-once seed against seed plus small holding payments
steps   : cheapest symmetric cost to cross x* in discrete time with a payment cap (large-step effect)
feedback: one agent, three feedback models
harmonic: Matching Pennies, payments and the weighted Fenchel coupling
euclid  : Euclidean regulariser, symmetric fast payment
frontK  : constant in the front recursion t_{j+1} = rho t_j + K"""
import numpy as np, json, sys, os
from numba import njit
from steer import sig, L, logit
c = 3.0; x0 = 0.1
out = json.load(open('res_extra.json')) if os.path.exists('res_extra.json') else {}
parts = sys.argv[1:] or ['holding', 'steps', 'feedback', 'harmonic', 'euclid', 'frontK']
def save(): json.dump(out, open('res_extra.json', 'w'), indent=1)

@njit(cache=True)
def _hold_drift(th, xs, lh, x, out):
    """Drift g + v with the holding payment v = -g for agents at or below the holding level with g < 0.
    Returns the cost rate sum_i x_i v_i."""
    N = th.shape[0]; rate = 0.0
    for i in range(N): x[i] = 1.0/(1.0+np.exp(-th[i]))
    for i in range(N):
        g = 3.0*(0.5*(x[(i+1) % N]+x[(i-1) % N]) - xs); v = 0.0
        if th[i] <= lh and g < 0: v = -g
        rate += x[i]*v; out[i] = g+v
    return rate

@njit(cache=True)
def ring_hold(N, xs, y, xh, dt, tmax):
    """Pair (0,1) raised to y at time 0 (its cost is counted outside). Agents not yet reached are held at
    x >= xh by paying v = -g while theta <= logit(xh) and g < 0. Classical Runge-Kutta, step dt.
    Returns the time until all x > 1/2 and the holding cost."""
    th = np.full(N, np.log(0.1/0.9)); ly = np.log(y/(1-y)); th[0] = ly; th[1] = ly
    lh = np.log(xh/(1-xh)) if xh > 0 else -1e300
    x = np.empty(N); cost = 0.0
    k1 = np.empty(N); k2 = np.empty(N); k3 = np.empty(N); k4 = np.empty(N); tmp = np.empty(N)
    for t in range(tmax):
        done = True
        for i in range(N):
            if th[i] <= 0.0: done = False
        if done: return t*dt, cost
        r1 = _hold_drift(th, xs, lh, x, k1)
        for i in range(N): tmp[i] = th[i]+0.5*dt*k1[i]
        r2 = _hold_drift(tmp, xs, lh, x, k2)
        for i in range(N): tmp[i] = th[i]+0.5*dt*k2[i]
        r3 = _hold_drift(tmp, xs, lh, x, k3)
        for i in range(N): tmp[i] = th[i]+dt*k3[i]
        r4 = _hold_drift(tmp, xs, lh, x, k4)
        cost += dt*(r1+2*r2+2*r3+r4)/6.0
        for i in range(N): th[i] += dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0
    return -1.0, cost

if 'holding' in parts:
    H = []
    for xs in [0.2, 0.3]:
        y = 2*xs+0.02; seed = 2*np.log((1-x0)/(1-y))
        for N, xh in [(12, 0.0), (16, 0.0), (20, 0.0), (20, 1e-6), (20, 1e-8), (40, 1e-8), (80, 1e-8), (160, 1e-8)]:
            t, hc = ring_hold(N, xs, y, xh, 0.02, 2000000000)
            H.append(dict(xs=xs, N=N, hold_at=xh, time=t, seed_cost=seed, holding_cost=hc)); print(H[-1], flush=True)
            out['holding'] = H; save()

def sym_min(U, al=0.01, xs=2/3, post=False, nv=401, ng=3001):
    lt = logit(xs); lo, hi = logit(x0)-0.2, lt+al*(U+c)+0.2
    grid = np.linspace(lo, hi, ng); V = np.where(grid >= lt, 0.0, 1e9); vs = np.linspace(0, U, nv)
    x = sig(grid)[:, None]; nxt = grid[:, None]+al*(c*(x-xs)+vs[None, :])
    stage = al*(sig(nxt) if post else x)*vs[None, :]
    for it in range(3000):
        Vn = np.where(grid >= lt, 0.0, (stage+np.interp(nxt, grid, V, left=1e9)).min(1))
        k = np.searchsorted(grid, logit(x0))
        if abs(Vn[k]-V[k]) < 1e-13 and Vn[k] < 1e8: V = Vn; break
        V = Vn
    return float(np.interp(logit(x0), grid, V))
if 'steps' in parts:
    LS = []
    for U in [10, 20, 30, 100, 300, 1000]:
        LS.append(dict(U=U, pre=sym_min(U), post=sym_min(U, post=True))); print(LS[-1], flush=True)
        out['large_steps'] = dict(rows=LS, one_step=float(x0*(logit(2/3)-logit(x0))), continuous=float(np.log((1-x0)/(1-2/3)))); save()

def feedback(kind, v=10.0, al=0.01, runs=4000, seed=0):
    rng = np.random.default_rng(seed); tot = 0.0; lt = logit(2/3)
    for r in range(runs):
        th = logit(x0); cost = 0.0
        while th < lt:
            x = sig(th)
            if kind == 'exact': cost += al*x*v; th += al*v
            elif rng.random() < x:
                cost += al*v; th += al*v/x if kind == 'importance' else al*v
        if kind == 'exact': return cost
        tot += cost
    return tot/runs
if 'feedback' in parts:
    out['feedback'] = {k: feedback(k) for k in ['exact', 'importance', 'received']}
    out['feedback']['logit_distance'] = float(logit(2/3)-logit(x0)); print(out['feedback']); save()

def mp(dt=1e-4, T=6.0):
    th = np.array([0.7, -0.4]); paid = 0.0
    F = lambda th: float(np.sum(np.logaddexp(0, th) - 0.5*th))    # sum_i KL(q_i || pi_i) up to a constant, q uniform
    F0 = F(th)
    for k in range(int(T/dt)):
        x = sig(th); t = k*dt
        g = np.array([2*(2*x[1]-1), -2*(2*x[0]-1)])                # Matching Pennies, payoff of H minus payoff of T
        uH = np.array([1+np.sin(t), 0.5]); uT = np.array([0.3, 1+np.cos(2*t)])
        paid += dt*np.sum((x-0.5)*(uH-uT))                        # sum_i <u_i, x_i - q_i>
        th = th + dt*(g+uH-uT)
    return dict(change_of_F=F(th)-F0, integral=float(paid))
if 'harmonic' in parts:
    out['harmonic'] = mp(); print(out['harmonic']); save()

def eucl(U, xs=2/3, dt=1e-6):
    x = x0; cost = 0.0
    while x < xs:
        cost += dt*x*U; x += dt*0.5*(c*(x-xs)+U)
    return cost
if 'euclid' in parts:
    out['euclid'] = dict(sim=eucl(1e4), formula=(2/3)**2-x0**2); print(out['euclid']); save()

if 'frontK' in parts:
    K = []
    for r in json.load(open('res_fronts.json'))['fronts']:
        if r['lam'] == 0.0:
            t = [a for a in r['times'] if a > 0]; rho = r['rho_pred']
            K.append(dict(xs=r['xs'], K=[t[j+1]-rho*t[j] for j in range(len(t)-1)]))     # the first values are transient; the last is a finite-ring effect
    out['front_K'] = K; print(K); save()
