"""Manuscript Figure 2: the substitution gap through a converging run.

R2: broken y-axis.  The full range spans seventeen decades but nothing is
observed between 1e-14 and 1e-4 (verified: 0 of 99,000 recorded values), so that
interval is omitted and the break is marked on both panels.  The separation it
represents is stated in words at the break rather than left to the eye.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import m1_analyse as A

GREY, INK, MUT = "#c9c8c1", "#0b0b0b", "#52514e"
BLUE, GREEN = "#2a78d6", "#12855c"

prim, mdl = A.read("m1_primary.csv"), A.read("m1_model.csv")
drv, lng = A.read("m1_driver.csv"), A.read("m1_longrun.csv")
nat = prim + mdl + drv + lng

floor = max(A.noise_of(r) for r in nat)
gmin = min(min(r[f"gap_{x}"] for x in A.RULES) for r in nat)

env = {}
for r in prim + mdl:
    env.setdefault(int(r["round"]), []).extend(r[f"gap_{x}"] for x in A.RULES)
xs = sorted(env)
lo_e = np.array([min(env[t]) for t in xs])
hi_e = np.array([max(env[t]) for t in xs])
md = np.array([np.median(env[t]) for t in xs])
xl = [int(r["round"]) for r in lng]
yl = [min(r[f"gap_{x}"] for x in A.RULES) for r in lng]

# assert the omitted interval really is empty before drawing a break across it
_all = np.array([r[f"gap_{x}"] for r in nat for x in A.RULES]
                + [r[c] for r in nat for c in A.NOISE_COLS])
assert ((_all > 1e-14) & (_all < 1e-4)).sum() == 0, "break interval not empty"

fig, (up, dn) = plt.subplots(2, 1, figsize=(5.2, 3.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.0, 1.0],
                                              hspace=0.14))

# ---- upper panel: the gap through training --------------------------------
up.set_yscale("log")
up.fill_between(xs, lo_e, hi_e, color=BLUE, alpha=0.18, lw=0, zorder=2)
up.plot(xs, md, color=BLUE, lw=1.9, zorder=4)
up.plot(xl, yl, color=GREEN, lw=1.3, ls="--", zorder=4)
up.axhline(gmin, color=INK, lw=0.8, ls=":", zorder=3)

up.set_ylim(6e-4, 5e-1)
up.set_ylabel(r"$\|$declared rule $-$ FedAvg$\|_\infty$", fontsize=8.5)
up.yaxis.set_label_coords(-0.105, 0.30)
up.text(430, 2.0e-1, "median and full range over four rules,\n"
        "three partitions, two optimisers, three seeds",
        fontsize=7.4, color=BLUE, ha="left", va="center")
up.text(2300, 5.2e-3, "2,000-round probe\n(i.i.d., full batch)",
        fontsize=7.4, color=GREEN, ha="right", va="bottom")
up.text(1.25, 1.30e-3, "smallest gap in any round of any run:  "
        r"$1.4\times10^{-3}$",
        fontsize=7.4, color=INK, ha="left", va="top")

# ---- lower panel: the re-implementation noise floor ------------------------
dn.set_yscale("log")
dn.axhspan(1e-17, floor, color=GREY, alpha=0.5, lw=0, zorder=0)
dn.set_ylim(3e-17, 6e-15)
dn.set_yticks([1e-16, 1e-15])
dn.text(0.985, 0.13, "re-implementation noise floor, re-measured each round  "
        f"$\\leq$ {floor:.1e}",
        transform=dn.transAxes, ha="right", va="bottom", fontsize=7.2, color=MUT)

for ax in (up, dn):
    ax.set_xscale("log")
    ax.set_xlim(1, 2400)
    ax.tick_params(labelsize=7.5, colors=MUT)
    ax.grid(axis="y", color="#eceae4", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
up.spines["bottom"].set_visible(False)
up.tick_params(axis="x", which="both", length=0, labelbottom=False)

# ---- axis break: marked on both panels, and stated in words ----------------
kw = dict(marker=[(-1, -0.55), (1, 0.55)], ms=6, ls="none", mec=MUT, mew=0.9,
          clip_on=False)
up.plot([0, 1], [0, 0], transform=up.transAxes, **kw)
dn.plot([0, 1], [1, 1], transform=dn.transAxes, **kw)
fig.text(0.545, 0.302, r"$10^{-4}$ to $10^{-14}$ omitted:  "
         "11.9 orders of magnitude, no observations",
         ha="center", va="center", fontsize=7.3, color=INK,
         bbox=dict(fc="white", ec="none", pad=1.6))

dn.set_xlabel("training round", fontsize=8.5, color=INK)
fig.savefig("figures/reachability.pdf", bbox_inches="tight")
fig.savefig("figures/reachability.png", dpi=210, bbox_inches="tight")
print("floor %.3e  gmin %.3e  orders %.2f" % (floor, gmin, math.log10(gmin / floor)))
print("wrote figures/reachability.{pdf,png}")
