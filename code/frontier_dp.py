"""Grid dynamic programme for the trade-off between terminal steering gap and alpha-weighted steering cost in the
two-player Stag Hunt of Huang et al. (2025) (alpha = 0.01, T = 500, U_max = 10). For each weight w, minimise
w*gap(pi_{T+1}) + cost on a 201 x 201 grid of logits with 21 payment levels. The grid error is up to about 1 percent,
so the value is not a lower bound: the threshold rule of threshold_rule.py is slightly cheaper."""
import numpy as np, sys, json
from numba import njit, prange
from bench_dp import AL, T, U, LO, HI, M, grid, h, sg, gap, interp
@njit(cache=True)
def best(V,t1,t2,levels):
    x1=sg(t1); x2=sg(t2); g1=3*x2-2; g2=3*x1-2
    bv=1e18; b1=0.0; b2=0.0
    for a in levels:
        n1=t1+AL*(g1+a)
        for b in levels:
            n2=t2+AL*(g2+b)
            val=AL*(x1*a+x2*b)+interp(V,n1,n2)
            if val<bv: bv=val; b1=a; b2=b
    return bv,b1,b2
@njit(parallel=True,cache=True)
def backup(V,levels):
    W=np.empty_like(V)
    for i in prange(M):
        for j in range(M): W[i,j]=best(V,grid[i],grid[j],levels)[0]
    return W
def solve(w,nlev=int(__import__('os').environ.get('DP_NLEV',21))):
    levels=np.linspace(0,U,nlev); V=np.empty((M,M))
    for i in range(M):
        for j in range(M): V[i,j]=w*gap(grid[i],grid[j])
    Vs=[V]
    for t in range(T): Vs.append(backup(Vs[-1],levels))
    return Vs[::-1],levels
def rollout(Vs,levels,x10,x20):
    t1=np.log(x10/(1-x10)); t2=np.log(x20/(1-x20)); cost=0.0
    for t in range(T):
        _,a,b=best(Vs[t+1],t1,t2,levels)
        x1=sg(t1); x2=sg(t2); cost+=AL*(x1*a+x2*b)
        t1,t2=t1+AL*(3*x2-2+a), t2+AL*(3*x1-2+b)
    return gap(t1,t2),cost
if __name__=='__main__':
    out={}
    for w in [float(a) for a in sys.argv[1:]]:
        Vs,levels=solve(w); seq=[0.1,0.3,0.5,0.7,0.9]
        R=np.array([rollout(Vs,levels,a,b) for a in seq for b in seq])
        out[w]=dict(mean_gap=R[:,0].mean(),mean_cost=R[:,1].mean(),value=w*R[:,0].mean()+R[:,1].mean(),runs=R.tolist())
        print('w',w,'mean gap %.3e mean cost %.3f'%(R[:,0].mean(),R[:,1].mean()),flush=True)
        json.dump(out,open(__import__('os').environ.get('DP_OUT','frontier_dp.json'),'w'))
