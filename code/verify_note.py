"""Independent numerical checks of the statements and numbers in the note. Prints one line per check.
Run: python verify_note.py"""
import numpy as np, json, itertools
from scipy.integrate import solve_ivp
rng = np.random.default_rng(7); ok_all = True
def sig(t): return 1/(1+np.exp(-np.clip(t, -700, 700)))
def logit(x): return np.log(x/(1-x))
def E(t): return np.logaddexp(0, t)
def check(name, cond, info=''):
    global ok_all; ok_all &= bool(cond); print(('PASS ' if cond else 'FAIL ')+name+('  '+info if info else ''))
def kl2(x, y): return x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))

# 1. eq (1): reward of H minus reward of G equals c (m - x*), random payoffs and graph
err = 0
for _ in range(200):
    P, S_ = rng.random(2); R = rng.random()+1.5; T = rng.random()+0.2
    if not (R > T and P > S_): continue
    N = 7; A = (rng.random((N, N)) < 0.5).astype(float); A = np.triu(A, 1); A = A+A.T
    for i in range(N):
        if A[i].sum() == 0: A[i, (i+1) % N] = A[(i+1) % N, i] = 1
    x = rng.random(N); m = A@x/A.sum(1); c = (R-T)+(P-S_); xs = (P-S_)/c
    rH = (A@(x*R+(1-x)*S_))/A.sum(1); rG = (A@(x*T+(1-x)*P))/A.sum(1)
    err = max(err, np.abs(rH-rG-c*(m-xs)).max())
check('eq (1): advantage = c (m - x*)', err < 1e-12, '%.1e' % err)
check('Stag Hunt (5,0,4,2): c = 3, x* = 2/3', abs((5-4)+(2-0)-3) < 1e-12 and abs((2-0)/3-2/3) < 1e-12)

# 2. update (2) equals pi' ~ pi exp(alpha A) with centred advantages
err = 0
for _ in range(1000):
    x = rng.random(); g = rng.normal(); v = rng.random()*10; al = 0.01+rng.random()
    qH, qG = g+v, 0.0; mean = x*qH+(1-x)*qG
    w = np.array([x*np.exp(al*(qH-mean)), (1-x)*np.exp(al*(qG-mean))]); xn = w[0]/w.sum()
    err = max(err, abs(logit(xn)-(logit(x)+al*(g+v))))
check('update (2) is the exponential-weights step', err < 1e-10, '%.1e' % err)

# 3. Lemma 2: identity, step term = KL(pi_t || pi_{t+1}), bound 0 <= KL <= s^2/8, either sign; post-update form
e1 = e2 = e3 = 0; lo = 1; hi = 0
for _ in range(200000):
    x = rng.random()*0.998+0.001; s = rng.uniform(-6, 6); xn = sig(logit(x)+s)
    D = np.log(1-x+x*np.exp(s))-x*s
    e1 = max(e1, abs(E(logit(xn))-E(logit(x))-x*s-D)); e2 = max(e2, abs(D-kl2(x, xn)))
    e3 = max(e3, abs(xn*s-(E(logit(xn))-E(logit(x)))-kl2(xn, x)))
    lo = min(lo, D); hi = max(hi, D/(s*s/8) if abs(s) > 1e-6 else 0)
check('Lemma 2: x s = Delta E - KL(pi_t||pi_t+1)', e1 < 1e-9 and e2 < 1e-9, '%.1e %.1e' % (e1, e2))
check('Lemma 2: 0 <= step term <= s^2/8', lo > -1e-12 and hi <= 1+1e-9, 'max ratio %.6f' % hi)
check('Remark 4: x_{t+1} s = Delta E + KL(pi_t+1||pi_t)', e3 < 1e-9, '%.1e' % e3)

# 4. Remark 5: K actions, unpaid action b
err = 0
for _ in range(2000):
    K = 5; z = rng.normal(size=K); pi = np.exp(z)/np.exp(z).sum(); q = rng.normal(size=K); u = rng.random(K)*3; b = rng.integers(K); u[b] = 0
    al = 0.3; zn = z+al*(q+u); pn = np.exp(zn)/np.exp(zn).sum()
    lhs = al*pi@u; rhs = (-np.log(pn[b])+np.log(pi[b]))+al*(q[b]-pi@q)-np.sum(pi*np.log(pi/pn))
    err = max(err, abs(lhs-rhs))
check('Remark 5: K-action identity', err < 1e-11, '%.1e' % err)

# 5. eq (4) in continuous time and the decay term of Section 6, one agent with a time-varying game term
def rhs(t, y, lam):
    th = y[0]; x = sig(th); g = np.sin(t)-0.5; v = 1+np.cos(3*t)
    return [g+v-lam*th, x*v, x*(-g), x*th]
for lam in (0.0, 0.07):
    s = solve_ivp(rhs, (0, 6), [logit(0.1), 0, 0, 0], args=(lam,), rtol=1e-11, atol=1e-13).y[:, -1]
    check('eq (4) with decay lambda=%.2f' % lam, abs(s[1]-(E(s[0])-E(logit(0.1))+s[2]+lam*s[3])) < 1e-8)

# 6. Proposition 1: value, and sum x_i m_i <= S^2/N on the graphs of Remark 6
check('Prop 1: log(0.9/(1/3)) = 0.993', abs(np.log(0.9/(1/3))-0.9933) < 1e-4)
def W(n, edges):
    A = np.zeros((n, n))
    for a, b in edges: A[a, b] = A[b, a] = 1
    return A/A.sum(1, keepdims=True)
graphs = {'K5': W(5, itertools.combinations(range(5), 2)), 'ring4': W(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
          'K33': W(6, [(a, b) for a in range(3) for b in range(3, 6)]), 'ring5': W(5, [(i, (i+1) % 5) for i in range(5)])}
for k, M in graphs.items():
    worst = max((x@M@x-x.sum()**2/len(x)) for x in rng.random((200000, M.shape[0])))
    if k == 'ring5': print('INFO ring of five: max of sum x m - S^2/N = %.3f (positive, so Remark 6 does not cover it)' % worst)
    else: check('Remark 6: sum x m <= S^2/N on '+k, worst <= 1e-12, '%.2e' % worst)

# 7. Proposition 2: closed forms (5), (6) against an ODE solve; values 0.993N, 1.072N, 0.321
def Csym(U, x0, xs, c): return U/(U+c*(1-xs))*np.log(U*(1-x0)/((U-c*(xs-x0))*(1-xs)))
def tsym(U, x0, xs, c):
    a = U-c*xs; return np.log(xs/x0)/a+np.log((1-x0)/(1-xs))/(a+c)-c/(a*(a+c))*np.log(U/(U-c*(xs-x0)))
worst = 0
for U, x0, xs, c in [(10, .1, 2/3, 3), (2.5, .1, 2/3, 3), (1.0, .05, .3, 3), (40, .2, .8, 5), (2.5, .1, 2/3, 3)]:
    ev = lambda t, y: y[0]-logit(xs); ev.terminal = True
    s = solve_ivp(lambda t, y: [U+c*(sig(y[0])-xs), sig(y[0])*U], (0, 1e4), [logit(x0), 0], events=ev, rtol=1e-12, atol=1e-13)
    worst = max(worst, abs(s.y[1, -1]-Csym(U, x0, xs, c)), abs(s.t[-1]-tsym(U, x0, xs, c)))
check('Prop 2: (5) and (6) against ODE', worst < 1e-7, '%.1e' % worst)
check('Prop 2: 1.072 and 0.321 at U=10', abs(Csym(10, .1, 2/3, 3)-1.0723) < 1e-4 and abs(tsym(10, .1, 2/3, 3)-0.3211) < 1e-4)
Us = np.linspace(1.71, 500, 20000); check('Prop 2: decreasing in U', np.all(np.diff(Csym(Us, .1, 2/3, 3)) < 0))

# 8. Ring propositions: numbers, and the invariance of the two-agent boxes
check('Prop 3 at x*=0.3: bounds 0.251 and 1.622', abs(np.log(.9/.7)-0.2513) < 1e-4 and abs(2*np.log(.9/.4)-1.6219) < 1e-4)
def ring_run(th, xs, c=3.0, T=4000, dt=0.01):
    for _ in range(int(T/dt)):
        x = sig(th); th = th+dt*c*(0.5*(np.roll(x, 1)+np.roll(x, -1))-xs)
    return sig(th)
okb = True
for xs in (0.1, 0.3, 0.45):
    for N in (3, 4, 6):
        th = np.full(N, logit(1e-6)); th[:2] = logit(2*xs+0.01); okb &= ring_run(th, xs).min() > 0.5
check('Prop 3: a pair above 2x* spreads (N = 3, 4, 6; others at 1e-6)', okb)
okc = True
for xs in (0.6, 2/3, 0.8):
    th = np.full(8, logit(1-1e-9)); th[:2] = logit(2*xs-1-0.005); okc &= ring_run(th, xs, T=300)[:2].max() < 2*xs-1
check('Prop 4: a pair below 2x*-1 never rises (others at 1)', okc)
check('Prop 4: bound empty at x*=0.55, x0=0.1', abs((1+0.1)/2-0.55) < 1e-12)

# 9. Remark 1: pair potential and its maximiser on a ring (brute force, N = 6)
def pot_ok(R, S_, T, P):
    N = 6; best = None; good = True
    for a in itertools.product([0, 1], repeat=N):          # 1 = H
        phi = sum((R-T) if a[i] == a[(i+1) % N] == 1 else (P-S_) if a[i] == a[(i+1) % N] == 0 else 0 for i in range(N))/2
        for i in range(N):
            b = list(a); b[i] = 1-b[i]
            pay = lambda p, i: sum((R if p[i] and p[j] else S_ if p[i] else T if p[j] else P) for j in ((i-1) % N, (i+1) % N))/2
            phib = sum((R-T) if b[k] == b[(k+1) % N] == 1 else (P-S_) if b[k] == b[(k+1) % N] == 0 else 0 for k in range(N))/2
            good &= abs((pay(a, i)-pay(b, i))-(phi-phib)) < 1e-12
        if best is None or phi > best[0]: best = (phi, a)
    return good, best[1]
g1, a1 = pot_ok(5, 1.5, 4, 2); g2, a2 = pot_ok(5, 0, 4, 2)
check('Remark 1: potential; all-H maximiser iff x*<1/2', g1 and g2 and a1 == (1,)*6 and a2 == (0,)*6)

# 10. Divide and conquer subsidies (Section 5)
def dc_ring(N, xs, c=3): return c*xs+(N-2)*c*max(xs-0.5, 0)
def dc_comp(N, xs, c=3): return sum(c*max(xs-(k-1)/(N-1), 0) for k in range(1, N+1))
check('Section 5: subsidy on complete graph ~ c N x*^2/2', abs(dc_comp(2000, 2/3)/(3*2000*(2/3)**2/2)-1) < 0.01)
def dc_ring_direct(N, xs, c=3):
    """Agents 0, 1, ..., N-1 switch in this order; each is paid the shortfall c (x* - m)^+ at its switch."""
    a = np.zeros(N); tot = 0.0
    for i in range(N): tot += c*max(xs-0.5*(a[(i-1) % N]+a[(i+1) % N]), 0); a[i] = 1
    return tot
check('Section 5: subsidy on the ring = c x* + (N-2) c (x*-1/2)^+', all(abs(dc_ring(N, xs)-dc_ring_direct(N, xs)) < 1e-12 for N in (5, 8, 13) for xs in (0.3, 0.5, 2/3, 0.8)))

# 11. Section 6: rho, rest point with decay; Section 8: utility 4 + 6 x1 x2; Section 10: worst-case step bound
check('rho = 1/(1-2x*): 1.667 and 2.5', abs(1/(1-0.4)-1.6667) < 1e-4 and abs(1/(1-0.6)-2.5) < 1e-12)
from scipy.optimize import brentq
for xs, lam in ((0.2, 0.05), (0.3, 0.05)):
    r = brentq(lambda th: 3*(sig(th)-xs)-lam*th, -40, -5); check('decay 0.05: inferior rest point exists (x*=%.1f)' % xs, r < -5, 'theta=%.1f' % r)
x1, x2 = rng.random(2); u1 = 5*x1*x2+4*(1-x1)*x2+2*(1-x1)*(1-x2); u2 = 5*x1*x2+4*(1-x2)*x1+2*(1-x1)*(1-x2)
check('Section 8: total utility = 4 + 6 x1 x2', abs(u1+u2-(4+6*x1*x2)) < 1e-12)
# Limitation 2: with g <= 0 and 0 <= v <= U, the energy rise of a step is at most (cost of the step) * (e^{aU}-1)/(aU)
worst = 0.0
for _ in range(20000):
    al = 10**rng.uniform(-3, -0.5); U = 10**rng.uniform(-1, 2); x = rng.uniform(0.001, 0.999); g = -rng.uniform(0, 5); vv = rng.uniform(0, U)
    dE = E(logit(x)+al*(g+vv))-E(logit(x)); worst = max(worst, dE - al*x*vv*(np.exp(al*U)-1)/(al*U))
check('Limitation 2: discrete lower-bound factor aU/(e^{aU}-1); 0.95 at aU = 0.1', worst < 1e-12 and abs(0.1/(np.exp(0.1)-1)-0.951) < 1e-3)
check('Remark 4: one step costs 0.29', abs(0.1*(logit(2/3)-logit(0.1))-0.289) < 1e-3)

# 12. Open question 2: scaled potential gap on a ring is 2/N (brute force, x* < 1/2, N = 8)
N = 8; vals = sorted({sum(1.0 if a[i] == a[(i+1) % N] == 1 else 1.5 if a[i] == a[(i+1) % N] == 0 else 0 for i in range(N))
                      for a in itertools.product([0, 1], repeat=N)})      # R-T = 1.0 < P-S = 1.5: all-G is the maximiser here
check('Open question 2: gap of scaled potential = 2/N', abs((vals[-1]-vals[-2])/(vals[-1]-vals[0])-2/N) < 1e-12)

# 13. Open question 4: Matching Pennies is harmonic with uniform centre
x = rng.random(2); v = [2*(2*x[1]-1), -2*(2*x[0]-1)]
check('Open question 4: sum <v_i, x_i - q_i> = 0 in Matching Pennies', abs(v[0]*(x[0]-.5)+v[1]*(x[1]-.5)) < 1e-12)

# 14. numbers quoted from the result files
X = json.load(open('res_extra.json')); T = json.load(open('res_threshold.json')); R = json.load(open('res_refine.json')); F = json.load(open('res_fronts.json'))
check('result files: harmonic equality', abs(X['harmonic']['change_of_F']-X['harmonic']['integral']) < 1e-3)
check('result files: Euclidean 0.436 against 0.434', abs(X['euclid']['sim']-X['euclid']['formula']) < 3e-3)
st = T['start_01']; check('Sections 1 and 8: 1.99 against 7.1, 7.9 and 10.6', abs(st[0]['basin']-1.99) < 5e-3 and [round(r['cost'], 1) for r in st] == [7.1, 7.9, 10.6])
for r in X['front_K']:
    q = 1.625 if r['xs'] < 0.25 else 2.948; mid = r['K'][5:-1]      # the agents opposite the seed are left out
    check('front constant K settles at %.3f (x*=%.1f)' % (q, r['xs']), max(abs(k-q) for k in mid) < 2e-3)
for r in F['fronts']:
    if r['lam'] == 0:
        t = [a for a in r['times'] if a > 0]; check('front ratio at x*=%.1f' % r['xs'], abs(t[-2]/t[-3]-r['rho_pred']) < 0.01, '%.4f' % (t[-2]/t[-3]))
bl = {(b['xs'], b['N'], b['k']): b['best'] for b in R['blocks']}
check('Table 3: block sizes 2, 3, 3, 3, 4 at x* = 0.2, 0.3, 0.4, 0.45, 0.48',
      [min(range(2, 6), key=lambda k: bl[(xs, N, k)]) for xs, N in ((0.2, 12), (0.3, 12), (0.4, 8), (0.45, 8), (0.48, 12))] == [2, 3, 3, 3, 4])
check('Table 3: cost at N = 8 and N = 12 agrees to six digits (x* = 0.2, 0.3)', all(abs(min(bl[(xs, 8, k)] for k in range(2, 6))/min(bl[(xs, 12, k)] for k in range(2, 6))-1) < 1e-6 for xs in (0.2, 0.3)))
uc = {r['d']: r['time'] for r in F['unit_cell'] if abs(r['xs']-0.3) < 1e-6}
check('Section 1 and Table 6: spacings cheaper than symmetric take 22.8 to 16240', abs(uc[6]-22.8) < 0.05 and abs(uc[20]-16240) < 1)
sd = F['seed_discrete']; check('Table 4: cost paid is below the energy rise of the paid pair', all(r['paid'] < r['energy_paid'] for r in sd))
print('\nALL PASS' if ok_all else '\nSOME CHECKS FAILED')
