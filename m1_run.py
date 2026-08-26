"""Drives the M1 arms and writes CSVs.  No analysis here; see m1_analyse.py."""
import csv, os, sys, time
import numpy as np
import m1_fed as M

OUT = "m1_results"
os.makedirs(OUT, exist_ok=True)
SEEDS = (0, 1, 2)
ROUNDS = 300


def write(name, rows):
    keys = list(rows[0])
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12e}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  -> {OUT}/{name}  ({len(rows)} rows)")


def arm_primary():
    """A. Real converging runs. logreg, FedAvg-driven, 3 partitions x 2
    optimisers x 3 seeds x 300 rounds."""
    rows = []
    for part in ("iid", "dir0.5", "dir0.1"):
        for opt in ("sgd", "fullbatch"):
            for s in SEEDS:
                t0 = time.time()
                rows += M.run("logreg", part, opt, s, rounds=ROUNDS)
                print(f"    A {part:7s} {opt:9s} seed{s}  {time.time()-t0:5.1f}s",
                      flush=True)
    write("m1_primary.csv", rows)


def arm_model():
    """B. Does the answer depend on model size?  MLP, 2410 params."""
    rows = []
    for part in ("iid", "dir0.5"):
        for s in SEEDS:
            t0 = time.time()
            rows += M.run("mlp", part, "sgd", s, rounds=ROUNDS, lr=0.5)
            print(f"    B {part:7s} mlp seed{s}  {time.time()-t0:5.1f}s", flush=True)
    write("m1_model.csv", rows)


def arm_driver():
    """C. Does the answer depend on which rule shaped the trajectory?
    Same measurement along a Krum-driven run."""
    rows = []
    for part in ("iid", "dir0.5"):
        for s in SEEDS:
            t0 = time.time()
            rows += M.run("logreg", part, "sgd", s, rounds=ROUNDS, driver="krum")
            print(f"    C {part:7s} krum-driver seed{s}  {time.time()-t0:5.1f}s",
                  flush=True)
    write("m1_driver.csv", rows)


def arm_longrun():
    """D. Asymptote probe: 2000 rounds, IID, full batch -- the configuration
    most favourable to the claim that convergence homogenises clients."""
    rows = M.run("logreg", "iid", "fullbatch", 0, rounds=2000)
    write("m1_longrun.csv", rows)


def arm_shared():
    """E. How much data overlap would it take?  Fraction of each client's
    dataset drawn from a common pool, 0 -> disjoint, 1 -> identical."""
    rows = []
    for rho in (0.0, 0.5, 0.8, 0.9, 0.955, 0.9775, 0.98876, 1.0):
        for s in SEEDS:
            rows += M.run("logreg", "iid", "fullbatch", s, rounds=50, shared=rho)
        print(f"    E shared={rho}", flush=True)
    write("m1_shared.csv", rows)


def arm_stepsize():
    """F. Step-size probe.  From one trained global model, take a single round
    at a range of local learning rates and measure the gap.  Isolates the
    dependence of the *absolute* gap on the size of the step, holding client
    heterogeneity fixed."""
    Xtr, ytr, Xte, yte = M.load_data(0)
    parts = M.partition(ytr, 16, "iid", 0)
    nex = [len(p) for p in parts]
    cls = M.LogReg

    warm = M.run("logreg", "iid", "fullbatch", 0, rounds=200)
    # reconstruct the round-200 global model by replaying the same run
    model = cls(seed=0)
    gp = [a.copy() for a in model.p]
    rng = np.random.default_rng(4000 + 0)
    for _ in range(200):
        ups = [M.local_train(cls, gp, Xtr[p], ytr[p], 3, 0.5, None, rng)
               for p in parts]
        gp = M.flower_aggregate("fedavg", ups, nex)
    assert abs(warm[-1]["test_acc"] - cls.loss_acc(gp, Xte, yte)[1]) < 1e-12

    rows = []
    for lr in (5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8,
               5e-9, 5e-10, 5e-11, 5e-12, 5e-13, 5e-14):
        ups = [M.local_train(cls, gp, Xtr[p], ytr[p], 3, lr, None, rng)
               for p in parts]
        row = {"lr": lr, "warm_round": 200}
        row.update(M.dispersion(ups, gp))
        agg = {r: M.flower_aggregate(r, ups, nex)
               for r in ("fedavg",) + M.RULES}
        for r in M.RULES:
            row[f"gap_{r}"] = M.linf(agg[r], agg["fedavg"])
        for r in ("fedavg",) + M.RULES:
            row[f"noise_{r}"] = M.linf(M.reference_aggregate(r, ups, nex), agg[r])
        row["step_rms"] = float(np.sqrt(np.mean(
            np.concatenate([(a - b).ravel()
                            for a, b in zip(agg["fedavg"], gp)]) ** 2)))
        rows.append(row)
        print(f"    F lr={lr:.0e} gap_krum={row['gap_krum']:.3e} "
              f"noise={max(row[f'noise_{r}'] for r in ('fedavg',)+M.RULES):.3e}",
              flush=True)
    write("m1_stepsize.csv", rows)


ARMS = {"primary": arm_primary, "model": arm_model, "driver": arm_driver,
        "longrun": arm_longrun, "shared": arm_shared, "stepsize": arm_stepsize}

if __name__ == "__main__":
    want = sys.argv[1:] or list(ARMS)
    for a in want:
        print(f"[{a}]", flush=True)
        t0 = time.time()
        ARMS[a]()
        print(f"[{a}] done in {time.time()-t0:.1f}s\n", flush=True)
