"""Cheapest pay-once seeds on the ring for x* < 1/2.
Part 1 starts from the ring seeds of run_search.py and optimises them further.
Part 2 optimises blocks of 2, 3, 4 and 5 adjacent agents separately."""
import numpy as np, json
from steer import *
c = 3.0; x0 = 0.1; t0 = logit(x0)
S = json.load(open('res_search.json')); out = []
for r in S:
    if r['graph'] != 'ring' or r['xs'] > 0.5: continue
    N, xs = r['N'], r['xs']; th0 = np.full(N, t0)
    d = np.maximum(logit(np.clip(np.array(r['x']), 1e-9, 1-1e-9)) - t0, 0); d[d < 1e-3] = 0
    s = scale_min(d, th0, 1, c, xs, iters=34); bd = s*d; best = potential(th0+bd, th0)
    rng = np.random.default_rng(0); step = 0.05; idx = np.nonzero(d)[0]
    for it in range(120):
        nd = bd.copy(); nd[idx] *= np.exp(step*rng.normal(0, 1, len(idx)))
        s = scale_min(nd, th0, 1, c, xs, iters=34)
        if s is None: continue
        cc = potential(th0+s*nd, th0)
        if cc < best-1e-9: best = cc; bd = s*nd
        if it % 40 == 39: step *= 0.5
    out.append(dict(N=N, xs=xs, old=r['best'], best=best, x=np.round(sig(th0+bd), 4).tolist())); print(out[-1], flush=True)
    json.dump(dict(seeds=out), open('res_refine.json', 'w'), indent=1)
json.dump(dict(seeds=out), open('res_refine.json', 'w'), indent=1)
# Blocks of k adjacent agents, k = 2..5: the search of run_search.py cannot add agents to a pattern, so each block
# size is optimised separately, starting from equal logit shifts.
blocks = []
for xs, N in [(0.2, 8), (0.2, 12), (0.3, 8), (0.3, 12), (0.4, 8), (0.45, 8), (0.48, 12)]:
    th0 = np.full(N, t0)
    for k in [2, 3, 4, 5]:
        d = np.zeros(N); d[:k] = 1.0; idx = np.arange(k)
        s = scale_min(d, th0, 1, c, xs, iters=34); bd = s*d; best = potential(th0+bd, th0)
        rng = np.random.default_rng(k); step = 0.2
        for it in range(240):
            nd = bd.copy(); nd[idx] *= np.exp(step*rng.normal(0, 1, k))
            s = scale_min(nd, th0, 1, c, xs, iters=34)
            if s is None: continue
            cc = potential(th0+s*nd, th0)
            if cc < best-1e-9: best = cc; bd = s*nd
            if it % 40 == 39: step *= 0.6
        blocks.append(dict(N=N, xs=xs, k=k, best=best, x=np.round(sig(th0+bd)[:k], 4).tolist())); print(blocks[-1], flush=True)
        json.dump(dict(seeds=out, blocks=blocks), open('res_refine.json', 'w'), indent=1)
