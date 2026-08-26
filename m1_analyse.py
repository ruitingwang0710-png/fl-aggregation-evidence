"""Analysis of the M1 arms.  Reads m1_results/*.csv, writes m1_results/
summary_*.csv and prints the numbers the report quotes.  No plotting."""
import csv, math, os
import numpy as np

OUT = "m1_results"
RULES = ("krum", "multikrum3", "median", "trimmed")
NOISE_COLS = [f"noise_{r}" for r in ("fedavg",) + RULES]


def read(name):
    with open(os.path.join(OUT, name)) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("model", "partition", "optimiser", "driver"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def ols_slope(lx, ly):
    lx, ly = np.asarray(lx, float), np.asarray(ly, float)
    ok = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[ok], ly[ok]
    if len(lx) < 5:
        return float("nan")
    A = np.vstack([lx, np.ones_like(lx)]).T
    return float(np.linalg.lstsq(A, ly, rcond=None)[0][0])


def cellkey(r):
    return (r["model"], r["partition"], r["optimiser"], r["driver"])


def group(rows):
    g = {}
    for r in rows:
        g.setdefault(cellkey(r), []).append(r)
    return g


def noise_of(r):
    return max(r[c] for c in NOISE_COLS)


# ---------------------------------------------------------------------------
def trajectory_table(rows, tag):
    """Per cell and rule: gap early/late, decay factor, minimum, dispersion
    correlation, and headroom above the measured noise floor."""
    out = []
    for key, rs in sorted(group(rows).items()):
        rs.sort(key=lambda r: (r["seed"], r["round"]))
        R = int(max(r["round"] for r in rs))
        nfloor = max(noise_of(r) for r in rs)
        by_round = {}
        for r in rs:
            by_round.setdefault(int(r["round"]), []).append(r)

        def med(rnd, col):
            return float(np.median([x[col] for x in by_round[rnd]]))

        for rule in RULES:
            g = f"gap_{rule}"
            allg = np.array([r[g] for r in rs])
            imin = int(np.argmin(allg))
            late = min(R, 300)
            out.append(dict(
                arm=tag, model=key[0], partition=key[1], optimiser=key[2],
                driver=key[3], rule=rule, rounds=R, seeds=len(set(r["seed"] for r in rs)),
                gap_r1=f"{med(1, g):.6e}",
                gap_r10=f"{med(10, g):.6e}",
                gap_r50=f"{med(50, g):.6e}",
                gap_rlate=f"{med(late, g):.6e}",
                decay_r1_to_late=f"{med(1, g)/max(med(late, g),1e-300):.3f}",
                gap_min=f"{allg.min():.6e}",
                gap_min_round=int(rs[imin]["round"]),
                gap_min_seed=int(rs[imin]["seed"]),
                noise_floor=f"{nfloor:.6e}",
                orders_above_floor=f"{math.log10(allg.min()/nfloor):.2f}",
                disp_r1=f"{med(1,'disp_rms_coord'):.6e}",
                disp_rlate=f"{med(late,'disp_rms_coord'):.6e}",
                disp_decay=f"{med(1,'disp_rms_coord')/max(med(late,'disp_rms_coord'),1e-300):.3f}",
                spearman_gap_disp=f"{spearman([r['disp_rms_coord'] for r in rs], [r[g] for r in rs]):.3f}",
                loglog_slope=f"{ols_slope(np.log10([r['disp_rms_coord'] for r in rs]), np.log10([r[g] for r in rs])):.3f}",
                spearman_gap_round=f"{spearman([r['round'] for r in rs], [r[g] for r in rs]):.3f}",
                relgap_r1=f"{med(1,'relgap_'+rule):.6e}",
                relgap_rlate=f"{med(late,'relgap_'+rule):.6e}",
                test_acc_r1=f"{med(1,'test_acc'):.4f}",
                test_acc_rlate=f"{med(late,'test_acc'):.4f}",
                krum_sel_agree_frac=f"{np.mean([r['krum_sel_agree'] for r in rs]):.4f}",
                krum_margin_min=f"{min(r['krum_score_margin_rel'] for r in rs):.3e}",
            ))
    return out


def main():
    allrows, tables = [], []
    for tag, fn in (("primary", "m1_primary.csv"), ("model", "m1_model.csv"),
                    ("driver", "m1_driver.csv"), ("longrun", "m1_longrun.csv")):
        if not os.path.exists(os.path.join(OUT, fn)):
            continue
        rows = read(fn)
        allrows += rows
        tables += trajectory_table(rows, tag)

    with open(os.path.join(OUT, "summary_trajectory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tables[0])); w.writeheader()
        w.writerows(tables)

    GLOBAL_FLOOR = max(noise_of(r) for r in allrows)
    GLOBAL_MIN = min(min(r[f"gap_{x}"] for x in RULES) for r in allrows)
    print("=" * 78)
    print("GLOBAL, over every recorded round of every converging run")
    print(f"  rounds recorded            : {len(allrows)}")
    print(f"  max re-implementation noise: {GLOBAL_FLOOR:.6e}")
    print(f"  min substitution gap       : {GLOBAL_MIN:.6e}")
    print(f"  orders of magnitude between: {math.log10(GLOBAL_MIN/GLOBAL_FLOOR):.2f}")
    agree = np.mean([r["krum_sel_agree"] for r in allrows])
    print(f"  Krum selection agreement   : {agree:.4f} "
          f"({int(round((1-agree)*len(allrows)))} disagreements)")
    print(f"  min relative score margin  : "
          f"{min(r['krum_score_margin_rel'] for r in allrows):.3e}")
    print("=" * 78)

    hdr = ("arm      model  partition opt       driver rule        "
           "gap_r1    gap_late     decay      min_gap  ord>floor  rho(g,d) slope")
    print(hdr); print("-" * len(hdr))
    for t in tables:
        print(f"{t['arm']:<8} {t['model']:<6} {t['partition']:<9} "
              f"{t['optimiser']:<9} {t['driver']:<6} {t['rule']:<11} "
              f"{float(t['gap_r1']):.2e} {float(t['gap_rlate']):.2e} "
              f"{float(t['decay_r1_to_late']):8.2f} {float(t['gap_min']):.2e} "
              f"{float(t['orders_above_floor']):9.2f} "
              f"{float(t['spearman_gap_disp']):8.3f} {float(t['loglog_slope']):6.2f}")

    # ---- shared-data arm --------------------------------------------------
    p = os.path.join(OUT, "m1_shared.csv")
    if os.path.exists(p):
        rows = read("m1_shared.csv")
        print("\n" + "=" * 78)
        print("HOW MUCH CLIENT OVERLAP WOULD IT TAKE?  (IID, full batch, 50 rounds)")
        print("shared   disp_rms    gap_krum    gap_mk3     gap_median  gap_trim"
              "    noise      ord>floor")
        out = []
        for rho in sorted(set(r["shared"] for r in rows)):
            rs = [r for r in rows if r["shared"] == rho]
            nf = max(noise_of(r) for r in rs)
            g = {x: float(np.median([r[f"gap_{x}"] for r in rs])) for x in RULES}
            rec = dict(shared=rho,
                       disp=f"{np.median([r['disp_rms_coord'] for r in rs]):.6e}",
                       noise=f"{nf:.6e}",
                       **{f"gap_{x}": f"{g[x]:.6e}" for x in RULES},
                       orders_above_floor=f"{math.log10(max(min(g.values()),1e-300)/nf):.2f}")
            out.append(rec)
            print(f"{rho:<8.5f} {np.median([r['disp_rms_coord'] for r in rs]):.3e}  "
                  f"{g['krum']:.3e}   {g['multikrum3']:.3e}   {g['median']:.3e}   "
                  f"{g['trimmed']:.3e}   {nf:.2e}  {rec['orders_above_floor']:>8}")
        with open(os.path.join(OUT, "summary_shared.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # ---- step-size arm ----------------------------------------------------
    p = os.path.join(OUT, "m1_stepsize.csv")
    if os.path.exists(p):
        rows = read("m1_stepsize.csv")
        print("\n" + "=" * 78)
        print("STEP-SIZE PROBE  (same trained model, same clients, one round)")
        print("lr         step_rms    disp_rms    gap_krum    gap_median  "
              "gap_trim    noise      ord>floor")
        for r in rows:
            nf = noise_of(r)
            mn = min(r[f"gap_{x}"] for x in RULES)
            print(f"{r['lr']:.0e}  {r['step_rms']:.3e}  {r['disp_rms_coord']:.3e}  "
                  f"{r['gap_krum']:.3e}  {r['gap_median']:.3e}  {r['gap_trimmed']:.3e}  "
                  f"{nf:.2e}  {math.log10(max(mn,1e-300)/max(nf,1e-300)):8.2f}")

    print("\nwrote summary_trajectory.csv, summary_shared.csv")


if __name__ == "__main__":
    main()
