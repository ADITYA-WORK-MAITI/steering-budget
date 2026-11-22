import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.7})
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#555555'
x0 = 0.1
os.makedirs('../figures', exist_ok=True)
def save(fig, name, final):
    """PDF for the note, and PDF and PNG (300 dpi) in ../figures."""
    fig.tight_layout(); fig.savefig('../note/%s.pdf' % name); fig.savefig('../figures/%s.pdf' % final); fig.savefig('../figures/%s.png' % final, dpi=300)
    plt.close(fig)
S = json.load(open('res_search.json')); F = json.load(open('res_fronts.json')); R = json.load(open('res_refine.json'))
def srch(N, xs, graph): return [r for r in S if r['N'] == N and abs(r['xs']-xs) < 1e-6 and r['graph'] == graph]
# ---------- Figure 1: budget against N
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.9))
N = np.arange(3, 201)
for a, xs, ttl in [(ax[0], 0.3, r'$x^*=0.3$ (target is risk dominant)'), (ax[1], 2/3, r'$x^*=2/3$ (Stag Hunt of Huang et al.)')]:
    sym = np.log((1-x0)/(1-xs))
    a.plot(N, N*sym, color=BLUE, lw=1.6, label='complete graph (Prop. 1)')
    if xs < 0.5:
        sd = [(n, min(b['best'] for b in R['blocks'] if b['N'] == n and abs(b['xs']-xs) < 1e-6)) for n in (8, 12)]
        a.plot([p[0] for p in sd], [p[1] for p in sd], 'o', color=ORANGE, ms=4, label='ring: local search')
        a.plot(N, np.full(len(N), 2*np.log((1-x0)/(1-2*xs))), color=ORANGE, lw=0.9, ls='--', label='ring: upper bound of Prop. 3')
        a.plot(N, np.full(len(N), sym), color=ORANGE, lw=0.9, ls=':', label='ring: lower bound of Prop. 3')
        d = [r for r in F['seed_discrete'] if abs(r['xs']-xs) < 1e-6][0]
        a.plot([200], [d['paid']], 'o', color=ORANGE, ms=5, mfc='white', label=r'ring: discrete run, $N=200$')
        a.set_yscale('log'); a.set_ylim(0.2, 100)
    else:
        lo = np.ceil(N/2)*np.log((1-x0)/(2*(1-xs)))
        a.fill_between(N, lo, N*sym, color=ORANGE, alpha=0.18, lw=0, label='ring: range allowed by Prop. 4')
        pts = [(n, srch(n, xs, 'ring')[0]['best']) for n in (8, 12) if srch(n, xs, 'ring')]
        a.plot([p[0] for p in pts], [p[1] for p in pts], 'o', color=ORANGE, ms=4, label='ring: local search')
    a.set_xlabel('number of agents $N$'); a.set_ylabel('budget'); a.set_title(ttl, fontsize=9)
    a.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1)
save(fig, 'fig_budget', 'Figure1_budget')
# ---------- Figure 2: front times and cost against time for pay-once seeds
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.7))
for xs, col in [(0.2, BLUE), (0.3, ORANGE)]:
    for r in F['fronts']:
        if abs(r['xs']-xs) > 1e-6: continue
        t = np.array(r['times']); t = t[t > 0][:11]; j = np.arange(1, 1+len(t))
        if r['lam'] == 0.0:
            ax[0].semilogy(j, t, '-o', color=col, ms=3, lw=1.3, label=r'$x^*=%.1f$, no decay' % xs)
            rho = 1/(1-2*xs); ax[0].semilogy(j, 1.6*t[-1]*rho**(j-j[-1]), '--', color=col, lw=0.8)
        elif r['lam'] == 0.05:
            ax[0].semilogy(j, t, ':', color=col, lw=1.4, label=r'$x^*=%.1f$, decay $\lambda=0.05$' % xs)
ax[0].set_xlabel('distance $j$ from the seed'); ax[0].set_ylabel(r'time $t_j$ at which agent $j$ first has $x>1/2$')
ax[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2); ax[0].set_title('Pay-once front on the ring', fontsize=9)
for xs, col in [(0.2, BLUE), (0.3, ORANGE)]:
    H = [r for r in F['unit_cell'] if abs(r['xs']-xs) < 1e-6]
    ax[1].semilogx([r['time'] for r in H], [r['per_agent'] for r in H], '-o', color=col, ms=3, lw=1.3, label=r'$x^*=%.1f$, seeds every $d$ agents' % xs)
    ax[1].axhline(H[0]['sym'], color=col, lw=0.8, ls='--')
ax[1].axvline(5, color=GREY, lw=0.8, ls=':'); ax[1].text(5.6, 0.008, r'$\alpha T=5$' + '\n(Huang et al.)', fontsize=7, color=GREY)
ax[1].set_xlabel(r'time $\tau$ needed'); ax[1].set_ylabel('cost per agent'); ax[1].set_ylim(0, 0.34)
ax[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1); ax[1].set_title('Pay-once seeds: cost against time', fontsize=9)
save(fig, 'fig_fronts', 'Figure2_fronts')
# ---------- Figure 3: consistency check
T = json.load(open('res_threshold.json')); C = np.array(T['curve']); D = json.load(open('frontier_dp.json'))
keep = (C[:, 1] < 8e-3) & (C[:, 1] > 2.5e-4)
fig, a = plt.subplots(figsize=(4.8, 3.8))
a.semilogx(C[keep, 1], C[keep, 2], '-', color=BLUE, lw=1.4, label=r'threshold rule, charged at $\pi_t$')
a.semilogx(C[keep, 1], C[keep, 3], '--', color=BLUE, lw=1.2, label=r'threshold rule, charged at $\pi_{t+1}$')
ws = sorted(D, key=float)
a.semilogx([D[w]['mean_gap'] for w in ws], [D[w]['mean_cost'] for w in ws], 'o', color=GREY, ms=3.5, label=r'grid dynamic programme, charged at $\pi_t$')
rep = np.array([[3.8e-3, 5.02, 0.2e-3, 0.29], [2.5e-3, 6.10, 0.4e-3, 0.31], [0.6e-3, 9.03, 0.1e-3, 0.43]])
a.errorbar(rep[:, 0], rep[:, 1], xerr=rep[:, 2], yerr=rep[:, 3], fmt='s', color=ORANGE, ms=5, mfc='white', lw=0.9, capsize=2,
           label='Figure 5 of Huang et al., with its error bars')
a.set_xlabel('mean terminal gap'); a.set_ylabel('mean cost'); a.invert_xaxis()
a.set_xticks([6e-3, 3e-3, 1e-3, 3e-4]); a.set_xticklabels(['0.006', '0.003', '0.001', '0.0003']); a.minorticks_off()
a.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
save(fig, 'fig_frontier', 'Figure3_figure5_check')
