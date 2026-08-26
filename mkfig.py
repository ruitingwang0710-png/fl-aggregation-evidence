import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RULES = ("krum", "multikrum3", "median", "trimmed")
LABEL = {"krum": "Krum", "multikrum3": "Multi-Krum ($m{=}3$)",
         "median": "Coord. median", "trimmed": "Trimmed mean"}
COL = {"krum": "#2a78d6", "multikrum3": "#eb6834",
       "median": "#1baf7a", "trimmed": "#4a3aa7"}
MK = {"krum": "o", "multikrum3": "s", "median": "^", "trimmed": "D"}
GREY, INK, MUT = "#c9c8c1", "#0b0b0b", "#52514e"

rows = list(csv.DictReader(open("results/separability_random.csv")))
sig = sorted({float(r["sigma"]) for r in rows}, reverse=True)
curves = {r: [float(x["median"]) for s in sig for x in rows
              if x["rule"] == r and float(x["sigma"]) == s] for r in RULES}
nf = list(csv.DictReader(open("results/separability_noisefloor.csv")))
NOISE = max(max(float(r["krum_max"]), float(r["fedavg_max"])) for r in nf)
CONS = 2.220446e-16

fig, (hi, lo) = plt.subplots(2, 1, figsize=(5.2, 3.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.1, 1.0], hspace=0.10))

# ---- upper panel: random honest configurations ----------------------------
hi.set_yscale("log")
for r in RULES:
    hi.plot(sig, curves[r], marker=MK[r], color=COL[r], lw=1.8, ms=4.5, mew=0,
            zorder=3, clip_on=False)
    hi.annotate(LABEL[r], xy=(sig[0], curves[r][0]), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7.2, color=COL[r], zorder=4)
hi.set_ylim(6e-4, 4e0)
hi.set_ylabel(r"$\|$declared rule $-$ FedAvg$\|_\infty$", fontsize=8.5)
hi.yaxis.set_label_coords(-0.105, 0.30)
hi.text(0.985, 0.90, "random honest configurations", transform=hi.transAxes,
        ha="right", va="top", fontsize=7.5, color=MUT)

# ---- lower panel: noise floor and constructed input ------------------------
lo.set_yscale("log")
lo.axhspan(1e-17, NOISE, color=GREY, alpha=0.5, lw=0, zorder=0)
lo.plot(sig, [CONS] * len(sig), marker="x", ls="none", ms=5, mew=1.4,
        color=INK, zorder=3, clip_on=False)
lo.set_ylim(3e-17, 6e-15)
lo.set_yticks([1e-16, 1e-15])
lo.text(0.985, 0.14, f"re-implementation noise floor  $\\leq$ {NOISE:.1e}",
        transform=lo.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUT)
lo.annotate("constructed symmetric input, all four rules:\n"
            r"$2.2\times10^{-16}$ at $n{=}7,11$;  exactly $0$ at $n{=}5$",
            xy=(sig[2], CONS), xytext=(0, 13), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.2, color=INK, zorder=4)

for ax in (hi, lo):
    ax.set_xscale("log"); ax.set_xlim(1.55, 0.0072); 
    ax.tick_params(labelsize=7.5, colors=MUT)
    ax.grid(axis="y", color="#eceae4", lw=0.55, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GREY)
hi.spines["bottom"].set_visible(False)
hi.tick_params(axis="x", which="both", length=0, labelbottom=False)

# axis-break marks
kw = dict(marker=[(-1, -0.55), (1, 0.55)], ms=6, ls="none", mec=MUT, mew=0.9,
          clip_on=False)
hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

lo.set_xlabel(r"client dispersion $\sigma$   (more homogeneous $\rightarrow$)",
              fontsize=8.5, color=INK)
fig.savefig("figures/separability.pdf", bbox_inches="tight")
fig.savefig("figures/separability.png", dpi=210, bbox_inches="tight")
print("ok; noise floor =", NOISE)
