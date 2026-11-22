"""Threshold rule for the two-player Stag Hunt of Huang et al. (alpha = 0.01, T = 500, U_max = 10):
pay U_max on H to an agent while its logit is below K, nothing otherwise. Exact rollouts, no grid.
Costs are reported with the payment charged at pi_t (the paper's definition) and at pi_{t+1} (the released code)."""
import numpy as np, json
from numba import njit
AL = 0.01; T = 500; U = 10.0
@njit(cache=True)
def roll(x1, x2, K):
    t1 = np.log(x1/(1-x1)); t2 = np.log(x2/(1-x2)); pre = 0.0; post = 0.0
    for t in range(T):
        a = 1.0/(1.0+np.exp(-t1)); b = 1.0/(1.0+np.exp(-t2))
        v1 = U if t1 < K else 0.0; v2 = U if t2 < K else 0.0
        n1 = t1+AL*(3*b-2+v1); n2 = t2+AL*(3*a-2+v2)
        pre += AL*(a*v1+b*v2)
        post += AL*(v1/(1.0+np.exp(-n1))+v2/(1.0+np.exp(-n2)))
        t1 = n1; t2 = n2
    a = 1.0/(1.0+np.exp(-t1)); b = 1.0/(1.0+np.exp(-t2))
    return 6.0*(1.0-a*b), pre, post
def curve(Ks):
    seq = [0.1, 0.3, 0.5, 0.7, 0.9]; rows = []
    for K in Ks:
        R = np.array([roll(a, b, K) for a in seq for b in seq]); rows.append([K]+R.mean(0).tolist())
    return np.array(rows)
if __name__ == '__main__':
    C = curve(np.linspace(1.0, 9.0, 1601))
    rep = [(10, 3.8e-3, 5.02, 4.73, 5.31), (25, 2.5e-3, 6.10, 5.79, 6.41), (100, 0.6e-3, 9.03, 8.60, 9.46)]
    out = dict(curve=C[::8].tolist(), at_reported=[])
    lg = -np.log(C[:, 1])
    for beta, gap, cost, lo, hi in rep:
        pre = float(np.interp(-np.log(gap), lg, C[:, 2])); post = float(np.interp(-np.log(gap), lg, C[:, 3]))
        out['at_reported'].append(dict(beta=beta, gap=gap, reported=cost, lo=lo, hi=hi, rule_pre=pre, rule_post=post))
        print(out['at_reported'][-1])
    # value w*gap + cost of the rule against the dynamic programme
    D = json.load(open('frontier_dp.json')); cmp = []
    for w in sorted(D, key=float):
        val = float(w)*C[:, 1]+C[:, 2]; k = val.argmin()
        cmp.append(dict(w=float(w), dp=D[w]['value'], rule=float(val[k]), K=float(C[k, 0]))); print(cmp[-1])
    out['against_dp'] = cmp
    # slope of the curve between the reported gaps, per unit of log(1/gap)
    A = out['at_reported']
    out['slope'] = [(A[k+1]['rule_pre']-A[k]['rule_pre'])/np.log(A[k]['gap']/A[k+1]['gap']) for k in range(2)]; print(out['slope'])
    # start (0.1, 0.1): cost of the rule at three gaps against the budget of Proposition 1 for entering the basin
    R = np.array([roll(0.1, 0.1, K) for K in np.linspace(1, 9, 1601)]); l2 = -np.log(R[:, 0])
    out['start_01'] = [dict(gap=g, cost=float(np.interp(-np.log(g), l2, R[:, 1])), basin=float(2*np.log(0.9/(1/3)))) for g in (3.8e-3, 2.5e-3, 6e-4)]
    print(out['start_01'])
    json.dump(out, open('res_threshold.json', 'w'), indent=1)
