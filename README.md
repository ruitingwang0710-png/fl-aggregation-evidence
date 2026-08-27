# fl-aggregation-evidence

What can a reviewer who was not present establish, from the record that a
single round of federated aggregation leaves behind?

The repository contains **two reproduction tracks**, with different costs.

**Track 1 — the controlled counterexample and the blind verifier.** No data set,
no training, no randomness. Runs on one CPU in seconds; every number is
reproducible by inspection.

**Track 2 — the empirical reachability study.** Federated training on real data:
31 converging runs, 11,000 recorded rounds, 16 clients, two model families,
i.i.d. and Dirichlet partitions. All seeds fixed. Scripts and per-run results are
under `m1_*` and `m1_results/`.

## Reproduce — Track 1 (seconds, one CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install "flwr[simulation]==1.33.0" numpy

python evidence.py         # builds every bundle, verifies it, writes results/summary.csv
python check_ab.py         # C2 differences, and the A/B byte-level comparison
python bit_exact_check.py  # asserts the two aggregates are identical bit for bit
python separability.py     # substitution gap vs. re-implementation noise floor
```

## Reproduce — Track 2 (about ten minutes, two cores)

```bash
pip install scikit-learn

python m1_run.py           # 31 federated training runs on the handwritten-digits set
python m1_analyse.py       # writes m1_results/*.csv
python mkfig_m1.py         # regenerates figures/reachability.pdf
python m1_krumfloor.py     # why the re-implementation noise floor stops being constant
```

Track 2 needs no network access: the data set ships with scikit-learn.

Optional, in the order they were originally run:

```bash
python walkthrough_1.py   # one aggregation each under FedAvg and Krum
python walkthrough_2.py   # the coincident fixture: the two rules agree bit for bit
python align_check.py     # independent NumPy implementation vs. Flower, both rules
python inspect_state.py   # what the SuperLink state retains after a completed run
cd coincident-app && flwr run . local-simulation --stream   # a full simulated round
```

## What is here

| file | what it does |
|---|---|
| `reference_krum.py` | Krum re-implemented in plain NumPy. **Imports nothing from Flower** — re-running the implementation under test would establish only that it agrees with itself. |
| `evidence.py` | Builds four worlds x four evidence levels plus a tampered bundle, verifies each with a blind verifier, writes `results/summary.csv`. |
| `check_ab.py` | Reports the C2 differences and whether the A and B bundles are byte-identical at each level. |
| `bit_exact_check.py` | Asserts identity of *representation*, not just of value: compares the `uint64` views and the raw bytes of the two aggregates. Equality of value is weaker — `-0.0` and `0.0` compare equal while differing in their leading bit — so this is what earns the word "bit-exact". Fails the run if the assertion does not hold. |
| `align_check.py` | Checks the independent implementation against real Flower on both rules and both fixtures. |
| `inspect_state.py` | Row counts of every table in the SuperLink SQLite state after a run. |
| `coincident-app/` | A minimal, framework-free Flower app: five simulated SuperNodes, one round, unmodified `Krum`. |
| `evidence/` | The generated bundles. `truth.json` sits **outside** each bundle directory and is never read by the verifier. |
| `results/summary.csv` | 51 rows: 17 cases x 3 claims. Machine-generated; not edited by hand. |

## The three claims and four verdicts

The verifier is handed a bundle directory and nothing else.

| claim | question |
|---|---|
| C1 | What does the record state was configured? |
| C2 | Does the recorded aggregate equal an independent re-performance of the declared rule on the recorded submissions? |
| C3 | Did the declared rule's code path execute? |

Verdicts: `consistent`, `contradicted`, `unable_to_determine`, `invalid_bundle`.
`unable_to_determine` is never folded into `consistent`.

## Headline numbers

| quantity | value |
|---|---|
| Krum vs. FedAvg output gap, `coincident` | `0.0` |
| Krum vs. FedAvg output gap, `divergent` | `1.62` |
| Independent NumPy vs. Flower, all four comparisons | `0.0` |
| A and B bundles byte-identical | at E0, E1 and E2; they differ only at E3 |

## Environment

Python 3.13.6, flwr 1.33.0, numpy 2.5.2, macOS. The controlled study also
reproduced under numpy 2.4.x, and the training study under Python 3.11 with
numpy 2.4.4 and scikit-learn 1.8.0 on Linux.

Runtimes: Track 1 completes in seconds on a single core. Track 2 takes about
ten minutes on two cores.

## Limitations

- **The recorder is trusted, and that is the largest assumption.** The runtime
  record behind C3 is written by the same process that performs the aggregation.
  It establishes execution only if that recorder is sound and was not bypassed.
  Locating that boundary is the contribution; crossing it is not.
- **Reachability is measured, and the answer is negative — for what was run.**
  None of the thirty-one converging runs entered the coincident region. That is
  not a proof that no training procedure reaches it. One data set, two model
  families; a deep network, or a far more homogeneous client population, is not
  covered.
- **Scale.** The training study is small by design: 1,797 examples, 16 clients,
  650 and 2,410 parameters. The controlled study uses five clients and four
  dimensions. No attack is executed anywhere — every client in every run is
  honest, so this work measures what the check can see, not what a defence can
  stop.
- **The evidence ladder is ours.** "C3 is unanswerable below E3" is a statement
  about the schema defined here, not a theorem about records in general.
- **Integrity is not authenticity.** `invalid_bundle` means a file changed after
  the manifest was written. It never means the file was false when written.
- **Search scope.** The account of Flower's retention behaviour rests on keyword
  search and call-site analysis of the local-simulation path, and is not an
  exhaustive proof that no alternative retention configuration exists in that
  version.

## How this was built

The question, the separation into three claims, the verdict design, the choice of
what to measure and the stated limitations are mine. Parts of the implementation
were written with an AI coding assistant. Every number reported here and in the
accompanying manuscript is regenerated by the scripts in this repository; nothing
is quoted from memory.
