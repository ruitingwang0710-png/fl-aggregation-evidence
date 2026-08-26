"""Is output-based re-performance able to separate rule substitution from honest
re-implementation?

Three measurements, all on real Flower 1.33.0 strategies:

  E1  substitution gap   |declared-rule output - FedAvg output| on random
                         honest-client configurations, over four declared rules
                         and five dispersions.
  E2  noise floor        |independent NumPy implementation - Flower| for the
                         SAME rule on the SAME input.  A tolerance can separate
                         substitution from honest re-implementation only if
                         these two distributions do not overlap.
  E3  constructed input  the same gap on a symmetric configuration (one centroid
                         plus symmetric +/- pairs), for the same four rules.

Writes results/separability_*.csv and figures/separability.pdf
"""
import csv, logging, os
for n in list(logging.root.manager.loggerDict) + ["flwr"]:
    logging.getLogger(n).disabled = True

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import (FedAvg, Krum, MultiKrum, FedMedian,
                                     FedTrimmedAvg)
import reference_krum as rk

D, NEX, F = 16, 10, 1
RULES = ("krum", "multikrum3", "median", "trimmed")
LABEL = {"krum": "Krum", "multikrum3": "Multi-Krum ($m{=}3$)",
         "median": "Coord. median", "trimmed": "Trimmed mean"}


def _msgs(u):
    return [Message(content=RecordDict({
        "arrays": ArrayRecord([np.asarray(x, dtype=np.float64)]),
        "metrics": MetricRecord({"num-examples": NEX})}),
        dst_node_id=i + 1, message_type="train") for i, x in enumerate(u)]


def flower(rule, u, n):
    S = {"fedavg":     lambda: FedAvg(min_train_nodes=n, min_available_nodes=n),
         "krum":       lambda: Krum(min_train_nodes=n, min_available_nodes=n,
                                    num_malicious_nodes=F),
         "multikrum3": lambda: MultiKrum(min_train_nodes=n, min_available_nodes=n,
                                         num_malicious_nodes=F, num_nodes_to_select=3),
         "median":     lambda: FedMedian(min_train_nodes=n, min_available_nodes=n),
         "trimmed":    lambda: FedTrimmedAvg(min_train_nodes=n, min_available_nodes=n,
                                             beta=0.2)}
    a, _ = S[rule]().aggregate_train(1, _msgs(u))
    return a.to_numpy_ndarrays()[0]


gap = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
SIGMAS = (1.0, 0.3, 0.1, 0.03, 0.01)
N, TRIALS = 7, 400
rng = np.random.default_rng(0)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---- E1 -------------------------------------------------------------------
e1 = []
curves = {r: [] for r in RULES}
for sigma in SIGMAS:
    draws = [rng.normal(1.0, sigma, size=(N, D)) for _ in range(TRIALS)]
    for rule in RULES:
        g = np.array([gap(flower(rule, u, N), flower("fedavg", u, N)) for u in draws])
        curves[rule].append(np.median(g))
        e1.append(dict(sigma=sigma, rule=rule, trials=TRIALS,
                       median=f"{np.median(g):.6e}", p05=f"{np.percentile(g,5):.6e}",
                       minimum=f"{g.min():.6e}",
                       frac_below_1e_3=f"{np.mean(g < 1e-3):.4f}"))
with open("results/separability_random.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(e1[0])); w.writeheader(); w.writerows(e1)

# ---- E2 -------------------------------------------------------------------
e2 = []
for sigma in SIGMAS:
    kr, av = [], []
    for _ in range(200):
        u = rng.normal(1.0, sigma, size=(N, D))
        U, NE = [u[i] for i in range(N)], [NEX] * N
        kr.append(gap(rk.krum(U, NE, f=F)[3], flower("krum", u, N)))
        av.append(gap(rk.fedavg(U, NE), flower("fedavg", u, N)))
    e2.append(dict(sigma=sigma, draws=200,
                   krum_max=f"{max(kr):.6e}", fedavg_max=f"{max(av):.6e}"))
with open("results/separability_noisefloor.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(e2[0])); w.writeheader(); w.writerows(e2)
NOISE = max(float(r["krum_max"]) for r in e2) + 0.0
NOISE = max(NOISE, max(float(r["fedavg_max"]) for r in e2))

# ---- E3 -------------------------------------------------------------------
def symmetric(n_pairs, d, offset=0.1):
    c = np.ones(d); rows = [c.copy()]
    for k in range(n_pairs):
        e = np.zeros(d); e[k % d] = offset * (1 + 0.3 * k)
        rows += [c + e, c - e]
    return np.array(rows)

e3 = []
for n_pairs in (2, 3, 5):
    n = 2 * n_pairs + 1
    u = symmetric(n_pairs, D)
    avg = flower("fedavg", u, n)
    for rule in RULES:
        out = flower(rule, u, n)
        e3.append(dict(n_clients=n, rule=rule, gap=f"{gap(out, avg):.6e}",
                       bit_identical=bool(np.array_equal(out.view(np.uint64),
                                                         avg.view(np.uint64)))))
with open("results/separability_constructed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(e3[0])); w.writeheader(); w.writerows(e3)

# ---- figure ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"krum": "#2a78d6", "multikrum3": "#eb6834",
       "median": "#1baf7a", "trimmed": "#4a3aa7"}
MK = {"krum": "o", "multikrum3": "s", "median": "^", "trimmed": "D"}

fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.set_xscale("log"); ax.set_yscale("log")

ax.axhspan(1e-18, NOISE, color="#c9c8c1", alpha=0.55, lw=0, zorder=0)
ax.text(1.15, NOISE * 0.45, "re-implementation noise floor",
        fontsize=7.5, color="#52514e", ha="right", va="center")

for rule in RULES:
    ax.plot(SIGMAS, curves[rule], marker=MK[rule], color=COL[rule],
            lw=2, ms=5, mew=0, zorder=3, clip_on=False)
    ax.annotate(LABEL[rule], xy=(SIGMAS[-1], curves[rule][-1]),
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.5, color=COL[rule])

ax.annotate("constructed symmetric input:\ngap $=0$ for all four rules",
            xy=(0.02, NOISE * 0.55), xytext=(0.05, 3e-13),
            fontsize=7.5, color="#0b0b0b", ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#52514e"))

ax.set_xlabel(r"client dispersion $\sigma$", fontsize=9)
ax.set_ylabel(r"$\|$declared rule $-$ FedAvg$\|_\infty$", fontsize=9)
ax.set_ylim(1e-18, 1e1); ax.set_xlim(0.008, 1.4)
ax.invert_xaxis()
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c1")
ax.grid(axis="y", color="#e8e7e1", lw=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig("figures/separability.pdf", bbox_inches="tight")
fig.savefig("figures/separability.png", dpi=200, bbox_inches="tight")

print("noise floor (max over all draws and both rules): %.3e" % NOISE)
print("\nE1 random-input substitution gap:")
for r in e1:
    print("  ", r)
print("\nE3 constructed symmetric input:")
for r in e3:
    print("  ", r)
print("\nwrote results/separability_{random,noisefloor,constructed}.csv")
print("wrote figures/separability.{pdf,png}")
