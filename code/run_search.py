import numpy as np, json, sys
from steer import *
c = 3.0; x0 = 0.1; import os
C = json.load(open('res_search.json')) if os.path.exists('res_search.json') else []
done = {(r['N'], round(r['xs'], 4), r['graph']) for r in C}
jobs = [(8, xs) for xs in [0.2, 0.3, 0.55, 2/3, 0.8, 0.4, 0.45]] + [(12, xs) for xs in [0.2, 0.3, 0.55, 2/3, 0.8]]
for N, xs in jobs:
    for kind, name in [(0, 'complete'), (1, 'ring')]:
        if (N, round(xs, 4), name) in done: continue
        res = []
        for sd in range(4):
            init = None
            if sd == 1: init = np.r_[np.ones(3), np.zeros(N-3)]
            if sd == 2: init = np.random.default_rng(5).random(N)
            if sd == 3: init = np.r_[np.ones(2), np.zeros(N-2)]
            res.append(search(N, kind, c, xs, x0, iters=500, seed=sd, init=init))
        b = min(res, key=lambda r: r[0])
        C.append(dict(N=N, xs=xs, graph=name, best=float(b[0]), sym=float(N*np.log((1-x0)/(1-xs))), x=np.round(b[1], 3).tolist()))
        print(C[-1], flush=True)
        json.dump(C, open('res_search.json', 'w'), indent=1)
