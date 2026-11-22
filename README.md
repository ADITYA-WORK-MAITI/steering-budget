# The smallest steering budget for exponential-weights learners in a coordination game

Code, figures and LaTeX source for the note of the same title (Aditya Maiti, 22 November 2026).
The note is `note/Maiti_Steering_Budget_Note.pdf`.

Setting: agents follow natural policy gradient with exact advantages (exponential weights) in a two-action
coordination game on a graph. A mediator pays action-contingent rewards, with the cost of Huang, Thoma, Shen,
Nax and He, "Learning to Steer Markovian Agents under Model Uncertainty" (ICLR 2025, arXiv:2407.10207).

## Layout

| Folder | Content |
|---|---|
| `code/` | Scripts and the result files (`.json`) used in the note |
| `note/` | The note as PDF, its LaTeX source `note.tex`, the three figure files it includes, and the script that fills its tables |
| `figures/` | The three figures as PDF and as PNG (300 dpi), written by `code/make_figs.py` |

## Requirements

Python with `numpy`, `numba` and `matplotlib`. `verify_note.py` also needs `scipy`.
The included results were produced with Python 3.13, numpy 2.5, scipy 1.18, numba 0.68 and matplotlib 3.11.

```
pip install -r requirements.txt
```

## Files in `code/`

| File | What it does | Output |
|---|---|---|
| `steer.py` | Dynamics, basin test, steering simulation, local search, closed form | library |
| `run_checks.py` | Numerical check of the energy identity; symmetric steering against the closed form (Table 1) | `res_checks.json` |
| `run_search.py` | Local search for the cheapest state in the basin of all-H, complete graph and ring (Table 3, rows with x* > 1/2) | `res_search.json` |
| `run_refine.py` | Cheapest pay-once seeds on the ring for x* < 1/2: blocks of 2 to 5 adjacent agents (Table 3, rows with x* < 1/2) | `res_refine.json` |
| `run_fronts.py` | Front times on the ring, pay-once seeds every d agents, discrete run at N = 200 (Tables 4 and 6, Figure 2) | `res_fronts.json` |
| `run_extra.py` | Holding payments, large steps, sampled payoffs, harmonic and Euclidean checks, front constants (Tables 2, 5 and 7) | `res_extra.json` |
| `threshold_rule.py` | Threshold rule for the two-player Stag Hunt, both cost conventions (Table 8, Figure 3) | `res_threshold.json` |
| `frontier_dp.py`, `bench_dp.py` | Grid dynamic programme for the same problem (Figure 3) | `frontier_dp.json`, `bench_dp.json` |
| `make_figs.py` | Draws the three figures from the result files | `note/fig_*.pdf`, `figures/*` |
| `verify_note.py` | Independent numerical checks of the lemmas, the closed forms and the numbers quoted in the note | prints PASS or FAIL per check; a saved run is `verify_out.txt` |

## Run

Run the scripts from inside `code/`, in this order. Later scripts read the result files of earlier ones.

```
cd code
python run_checks.py
python run_search.py
python run_refine.py
python run_fronts.py
python run_extra.py
python frontier_dp.py 300 500 800 1500 3000 6000
python threshold_rule.py
python make_figs.py
python verify_note.py
```

The whole list takes about 15 minutes on two cores. All result files are included, so each script can also be run alone.
`run_search.py` skips the cases that are already in `res_search.json`. Delete that file first to repeat the search.
`python bench_dp.py` writes `bench_dp.json` (about 5 minutes). The note does not use that file.

To build the PDF: `cd note`, then `pdflatex note.tex` twice.

The numbers in the eight tables are not typed by hand. `note/fill_tables.py` reads the result files and writes them
into `note_template.tex`, which gives `note.tex`. Run `python fill_tables.py` inside `note/` after changing a result file.

## Notes

- `x` is the probability of the target action H. `theta` is its logit.
- Advantage of H over G for agent i: `c * (m_i - xs)`, where `m_i` is the mean of `x` over the neighbours.
  The Stag Hunt of Huang et al. has `c = 3`, `xs = 2/3`.
- Cost of a step: `alpha * sum_i x_i * v_i` for a payment `v_i >= 0` on H (the paper's cost times the step size).
- All continuous-time runs use a classical Runge-Kutta step: 0.05 for the basin test and the front times,
  0.02 for the runs with holding payments. Crossing times are interpolated linearly inside a step.
- The basin test stops early on the ring by the invariant two-agent boxes of Propositions 3 and 4,
  so it cannot be used to test those propositions.
- Names in the result files: `potential` and `best` are energy differences in the sense of the note, `work` is the
  regret term, `D` is the step term, `impulse` is the energy of the smallest two-agent seed, and `energy_paid` is the
  energy rise of the two paid agents.
- The three points of Figure 5 of Huang et al. and their error bars are taken from the vector graphics of the
  published figure. They are typed into `threshold_rule.py` and `make_figs.py`.
- `bench_dp.json` holds the optimum of the per-step training objective (Eq. (12) of Huang et al.) on the grid.
  The note does not use it. `frontier_dp.json` is a grid computation with an error of up to about 1 percent;
  the threshold rule is exact for the rule it describes.

## Licence

The code is under the MIT licence (see `LICENSE`). The note and the figures are by the author; please cite the note if you use them.
