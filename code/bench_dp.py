"""Grid dynamic programme for the two-player Stag Hunt experiment of Huang et al. (2025).
Objective (as in the released code, obj=MinGap): sum_t alpha*payment_t + alpha*beta*gap(pi_{t+1})."""
import numpy as np, sys, json, os
from numba import njit, prange
AL=0.01; T=500; U=10.0
LO,HI,M=-6.0,14.0,int(os.environ.get('DP_M',201))   # DP_M changes the grid size (use a fresh NUMBA_CACHE_DIR)
grid=np.linspace(LO,HI,M); h=grid[1]-grid[0]
@njit(cache=True)
def sg(t): return 1.0/(1.0+np.exp(-t))
@njit(cache=True)
def gap(t1,t2):
    x1=sg(t1); x2=sg(t2)
    u1=5*x1*x2+4*(1-x1)*x2+2*(1-x1)*(1-x2)
    u2=5*x1*x2+4*(1-x2)*x1+2*(1-x1)*(1-x2)
    return 10.0-u1-u2
@njit(cache=True)
def interp(V,a,b):
    fa=(a-LO)/h; fb=(b-LO)/h
    if fa<0: fa=0.0
    if fb<0: fb=0.0
    if fa>M-1.000001: fa=M-1.000001
    if fb>M-1.000001: fb=M-1.000001
    i=int(fa); j=int(fb); p=fa-i; q=fb-j
    return (1-p)*(1-q)*V[i,j]+p*(1-q)*V[i+1,j]+(1-p)*q*V[i,j+1]+p*q*V[i+1,j+1]
@njit(cache=True)
def best(V,t1,t2,beta,levels):
    x1=sg(t1); x2=sg(t2); g1=3*x2-2; g2=3*x1-2
    bv=1e18; b1=0.0; b2=0.0
    for a in levels:
        n1=t1+AL*(g1+a)
        for b in levels:
            n2=t2+AL*(g2+b)
            val=AL*(x1*a+x2*b)+AL*beta*gap(n1,n2)+interp(V,n1,n2)
            if val<bv: bv=val; b1=a; b2=b
    return bv,b1,b2
@njit(parallel=True,cache=True)
def backup(V,beta,levels):
    W=np.empty_like(V)
    for i in prange(M):
        for j in range(M):
            W[i,j]=best(V,grid[i],grid[j],beta,levels)[0]
    return W
def solve(beta,nlev=21):
    levels=np.linspace(0,U,nlev); Vs=[np.zeros((M,M))]
    for t in range(T): Vs.append(backup(Vs[-1],beta,levels))
    return Vs[::-1],levels      # Vs[t] = cost-to-go with T-t steps left
def rollout(Vs,levels,beta,x10,x20):
    t1=np.log(x10/(1-x10)); t2=np.log(x20/(1-x20)); cost=0.0
    for t in range(T):
        _,a,b=best(Vs[t+1],t1,t2,beta,levels)
        x1=sg(t1); x2=sg(t2); cost+=AL*(x1*a+x2*b)
        t1,t2=t1+AL*(3*x2-2+a), t2+AL*(3*x1-2+b)
    return gap(t1,t2),cost,sg(t1),sg(t2)
if __name__=='__main__':
    out={}
    betas=[float(b) for b in sys.argv[2:]] or [10.0,25.0,100.0]
    for beta in betas:
        Vs,levels=solve(beta); seq=[0.1,0.3,0.5,0.7,0.9]; R=[]
        for a in seq:
            for b in seq: R.append(rollout(Vs,levels,beta,a,b))
        R=np.array(R); out[beta]=R.tolist()
        print('beta',beta,'mean gap %.2e  mean cost %.3f   (G-basin inits: cost %.2f)'%(R[:,0].mean(),R[:,1].mean(),R[0,1]),flush=True)
    json.dump(out,open(sys.argv[1] if len(sys.argv)>1 else 'bench_dp.json','w'))
