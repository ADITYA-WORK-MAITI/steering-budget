import numpy as np, json, time
from steer import *
c = 3.0; x0 = 0.1; out = {}
# A. identity check in discrete time, random payments
rng = np.random.default_rng(1); chk = []
for kind, name in [(0, 'complete'), (1, 'ring')]:
    N = 12; al = 0.01; xs = 2/3; U = 10; th = rng.normal(-1, 1, N); th0 = th.copy()
    cost = work = D = 0.0
    for t in range(500):
        x = sig(th); m = np.empty(N); neigh_mean(x, kind, m); g = c*(m-xs); v = rng.uniform(0, U, N)
        s = al*(g+v); cost += al*np.sum(x*v); work += al*np.sum(-x*g); D += np.sum(np.log(1-x+x*np.exp(s))-x*s); th = th+s
    chk.append(dict(graph=name, cost=cost, rhs=potential(th, th0)+work-D, potential=potential(th, th0), work=work, D=D))
out['identity'] = chk; print(chk)
# B. complete graph, finite U: simulation at alpha=0.01 against closed form
B = []
for U in [2.5, 3, 5, 10, 20]:
    for N in [2, 10, 50, 200]:
        xs = 2/3; th0 = np.full(N, logit(x0))
        th, cost, work, D, steps = steer_sim(th0, np.full(N, xs), 0, c, xs, 0.01, U, 100000)
        B.append(dict(U=U, N=N, sim=0.01*cost, closed=closed_form_complete(U, x0, xs, c, N), steps=steps,
                      impulse=N*np.log((1-x0)/(1-xs)), reached=int(basin(th, 0, c, xs))))
out['complete_finiteU'] = B
# finer step size: the alpha-weighted cost converges to the closed form
B2=[]
for al in [0.01,0.001,0.0001]:
    N=10; xs=2/3; U=10.0; th0=np.full(N,logit(x0))
    th,cost,work,D,steps=steer_sim(th0,np.full(N,xs),0,c,xs,al,U,10000000)
    B2.append(dict(alpha=al,sim=al*cost,closed=closed_form_complete(U,x0,xs,c,N),time=al*steps,D=D)); print(B2[-1])
out['alpha_convergence']=B2
for b in B: print(b)
json.dump(out, open('res_checks.json', 'w'), indent=1, default=float)
