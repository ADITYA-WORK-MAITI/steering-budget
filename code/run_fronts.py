import numpy as np, json
from steer import *
c = 3.0; x0 = 0.1; out = {}
def pair_seed(N, xs, margin=1.05):
    """Two adjacent agents raised by the same logit shift, `margin` times the smallest shift that reaches the basin."""
    th0 = np.full(N, logit(x0)); d = np.zeros(N); d[:2] = 1
    s = scale_min(d, th0, 1, c, xs, iters=34)
    return th0 + margin*s*d, potential(th0 + margin*s*d, th0)      # the seed that is simulated, and its cost
# D. activation times along the ring, exact NPG (lam=0) and with logit decay (lam>0)
F = []
for xs in [0.2, 0.3]:
    for lam in [0.0, 0.05, 0.2]:
        N = 26 if xs == 0.3 else 40
        th, cost = pair_seed(N, xs)
        first = run_free(th, 1, c, xs, lam, 0.05, 40000000, 0.5)
        hops = [float(first[1+j]) for j in range(1, N//2)]     # agents 2,3,... on one side of the pair (0,1)
        F.append(dict(xs=xs, lam=lam, N=N, times=hops, seed_cost=cost, rho_pred=1/(1-2*xs)))
        r = [hops[j+1]/hops[j] for j in range(len(hops)-2)] if hops[-2] > 0 else None
        print(xs, lam, 'cost', round(cost, 4), 'times', np.round(hops[:12], 1), 'ratios', None if r is None else np.round(r[-5:], 3), flush=True)
out['fronts'] = F
# E. time for a ring of size d to convert from one pair seed (unit cell of a periodic seeding), lam=0
H = []
for xs in [0.2, 0.3]:
    for d in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        th, cost = pair_seed(d, xs)
        first = run_free(th, 1, c, xs, 0.0, 0.05, 40000000, 0.5)
        H.append(dict(xs=xs, d=d, time=float(first.max()), per_agent=cost/d, sym=np.log((1-x0)/(1-xs))))
        print(H[-1], flush=True)
out['unit_cell'] = H
# F. discrete-time check of the seed with the alpha and U of Huang et al.: N = 200 ring, pay two agents only
G = []
for xs in [0.2, 0.3]:
    N = 200; th0 = np.full(N, logit(x0)); thS, pot = pair_seed(N, xs, 1.0)
    target = np.full(N, 0.0); target[:2] = sig(thS[0]) + 0.01
    th, cost, work, D, steps = steer_sim(th0, target, 1, c, xs, 0.01, 10.0, 100000)
    G.append(dict(xs=xs, N=N, impulse=pot, paid=0.01*cost, work=0.01*work, D=D, steps=steps, in_basin=int(basin(th, 1, c, xs)),
                  energy_paid=float(np.sum(L(th[:2])-L(th0[:2]))), pair_x=float(sig(th[0])),
                  complete_lower=N*np.log((1-x0)/(1-xs))))
    print(G[-1])
out['seed_discrete'] = G
json.dump(out, open('res_fronts.json', 'w'), indent=1)
