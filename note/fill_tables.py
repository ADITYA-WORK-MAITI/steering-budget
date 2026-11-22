"""Fills the eight tables and two quoted constants of the note from the result files in ../code.
Reads note_template.tex, writes note.tex. Run from inside note/:  python fill_tables.py"""
import json, re, sys, numpy as np
C = '../code/'
src = sys.argv[1] if len(sys.argv) > 1 else 'note_template.tex'
dst = sys.argv[2] if len(sys.argv) > 2 else 'note.tex'
t = open(src).read()
ch = json.load(open(C+'res_checks.json')); S = json.load(open(C+'res_search.json')); F = json.load(open(C+'res_fronts.json'))
R = json.load(open(C+'res_refine.json')); X = json.load(open(C+'res_extra.json')); T = json.load(open(C+'res_threshold.json'))
# Table 1
t = t.replace('ALPHATAB', '\n'.join('%g & %.3f & %.3f & %.3f\\\\' % (r['alpha'], r['sim'], r['closed'], r['time']) for r in ch['alpha_convergence']))
# Table 2
t = t.replace('STEPSTAB', '\n'.join('%d & %g & %.3f & %.3f\\\\' % (r['U'], 0.01*r['U'], r['pre'], r['post']) for r in X['large_steps']['rows'] if r['U'] <= 100))
# Table 3: x* < 1/2 from the block search (cheapest block size; the smallest block within 1e-5 of the minimum), x* > 1/2 from run_search
rows = []
for xs in [0.2, 0.3, 0.4, 0.45, 0.48, 0.55, 2/3, 0.8]:
    for N in [8, 12]:
        sym = N*np.log(0.9/(1-xs)); xl = '2/3' if abs(xs-2/3) < 1e-6 else '%g' % xs
        if xs < 0.5:
            bb = [b for b in R['blocks'] if b['N'] == N and abs(b['xs']-xs) < 1e-6]
            if not bb: continue
            lo = min(b['best'] for b in bb); b = min([b for b in bb if b['best'] < lo*(1+1e-5)], key=lambda b: b['k'])
            ss = [r for r in R['seeds'] if r['N'] == N and abs(r['xs']-xs) < 1e-6]
            assert not ss or ss[0]['best'] > b['best']*(1-1e-5), (ss[0]['best'], b['best'])
            rows.append('$%s$ & %d & %.3f & %.3f & %d adjacent: %s\\\\' % (xl, N, sym, b['best'], b['k'], ', '.join('%.3f' % x for x in b['x'])))
        else:
            rr = [r for r in S if r['N'] == N and abs(r['xs']-xs) < 1e-6 and r['graph'] == 'ring']
            if not rr: continue
            assert abs(rr[0]['best']-sym) < 2e-3*sym, (rr[0]['best'], sym)
            rows.append('$%s$ & %d & %.3f & %.3f & all %d agents at $%.3f$\\\\' % (xl, N, sym, rr[0]['best'], N, rr[0]['x'][0]))
t = t.replace('SEARCHTAB', '\n'.join(rows))
# Table 4
t = t.replace('SEEDTAB', '\n'.join('%g & %d & %.3f & %.3f & %.3f & %.1f\\\\' % (r['xs'], r['steps'], r['paid'], r['energy_paid'], r['impulse'], r['complete_lower']) for r in F['seed_discrete']))
# Table 5
hr = []
for r in X['holding']:
    if abs(r['xs']-0.3) > 1e-6: continue
    if r['hold_at'] == 0.0: hr.append('%d & pay once & %.0f & %.3f & 0\\\\' % (r['N'], r['time'], r['seed_cost']))
    elif r['hold_at'] == 1e-8: hr.append('%d & seed and hold & %.0f & %.3f & %.5f\\\\' % (r['N'], r['time'], r['seed_cost'], r['holding_cost']))
t = t.replace('HOLDTAB', '\n'.join(hr))
# Table 6
t = t.replace('CELLTAB', '\n'.join('%d & %.3f & %.1f\\\\' % (r['d'], r['per_agent'], r['time']) for r in F['unit_cell'] if abs(r['xs']-0.3) < 1e-6))
# Table 7
fb = X['feedback']
t = t.replace('FEEDTAB', 'Exact expected advantage (this note) & %.3f\\\\\nSampled actions, importance-weighted update & %.3f\\\\\nSampled actions, update only on payments received & %.3f\\\\' % (fb['exact'], fb['importance'], fb['received']))
# Table 8
t = t.replace('BENCHTAB', '\n'.join('%d & $%.1f\\times10^{-3}$ & %.2f & %.2f & %.2f\\\\' % (r['beta'], r['gap']*1e3, r['reported'], r['rule_pre'], r['rule_post']) for r in T['at_reported']))
# the two front constants quoted in Section 7.3: value of K_j before the agents opposite the seed
K = {round(r['xs'], 1): r['K'] for r in X['front_K']}
for key, xs in (('KTWO', 0.2), ('KTHREE', 0.3)):
    k = K[xs]; assert max(abs(a-k[-2]) for a in k[5:-1]) < 2e-3, k
    t = t.replace(key, '$%.3f$' % k[-2])
left = re.findall(r'[A-Z]{4,}TAB|KTWO|KTHREE', t); assert not left, left
open(dst, 'w').write(t)
