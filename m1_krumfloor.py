"""Where, and why, does the re-implementation noise floor stop being constant?

Flower computes Krum's squared distances as ||x||^2 + ||y||^2 - 2 x.y.  The
reference computes ||x - y||^2 directly.  These agree in exact arithmetic.  In
floating point the expansion loses accuracy in proportion to ||x||^2 / ||x-y||^2,
so the two implementations can disagree once clients get close relative to the
size of the model vector.  This probe holds the trained model and the client
partition fixed and shrinks only the local step, so the only thing that changes
is how close the client updates are to one another.
"""
import csv, os
import numpy as np
import m1_fed as M

OUT = "m1_results"
os.makedirs(OUT, exist_ok=True)

Xtr, ytr, Xte, yte = M.load_data(0)
parts = M.partition(ytr, 16, "iid", 0)
nex = [len(p) for p in parts]
cls = M.LogReg

model = cls(seed=0)
gp = [a.copy() for a in model.p]
rng = np.random.default_rng(4000)
for _ in range(200):
    ups = [M.local_train(cls, gp, Xtr[p], ytr[p], 3, 0.5, None, rng) for p in parts]
    gp = M.flower_aggregate("fedavg", ups, nex)
print("warm model: test acc %.4f" % cls.loss_acc(gp, Xte, yte)[1])

rows = []
lrs = [5e-1 * 10 ** (-k / 3.0) for k in range(0, 34)]
for lr in lrs:
    ups = [M.local_train(cls, gp, Xtr[p], ytr[p], 3, lr, None, rng) for p in parts]
    flat = np.stack([np.concatenate([a.ravel() for a in u]) for u in ups])
    unorm = float(np.median(np.linalg.norm(flat, axis=1)))
    pair = [float(np.linalg.norm(flat[i] - flat[j]))
            for i in range(16) for j in range(i + 1, 16)]
    row = {"lr": lr, "u_norm": unorm,
           "min_pair_l2": min(pair), "med_pair_l2": float(np.median(pair)),
           "sep_ratio": min(pair) / unorm}
    row.update(M.krum_selection(ups, nex))
    agg = {r: M.flower_aggregate(r, ups, nex) for r in ("fedavg",) + M.RULES}
    for r in M.RULES:
        row[f"gap_{r}"] = M.linf(agg[r], agg["fedavg"])
    for r in ("fedavg",) + M.RULES:
        row[f"noise_{r}"] = M.linf(M.reference_aggregate(r, ups, nex), agg[r])
    rows.append(row)
    print("lr=%.2e sep=%.2e  agree=%d  gap_krum=%.3e n_krum=%.3e "
          "n_mk3=%.3e n_med=%.3e n_trim=%.3e n_avg=%.3e"
          % (lr, row["sep_ratio"], row["krum_sel_agree"], row["gap_krum"],
             row["noise_krum"], row["noise_multikrum3"], row["noise_median"],
             row["noise_trimmed"], row["noise_fedavg"]), flush=True)

with open(os.path.join(OUT, "m1_krumfloor.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.12e}" if isinstance(v, float) else v)
                    for k, v in r.items()})
print(f"wrote {OUT}/m1_krumfloor.csv ({len(rows)} rows)")
