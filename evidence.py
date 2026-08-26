"""Four worlds x four evidence levels x three claims x four verdicts.

Every world *declares* Krum (f=1). They differ in what actually ran, and in
whether that difference reaches the output.

    world  declared  executed  fixture      meaning
    A      krum      krum      coincident   the declared rule ran
    B      krum      fedavg    coincident   it did not run, and the output is the same
    C      krum      fedavg    divergent    it did not run, and the output differs
    D      krum      krum      divergent    the declared rule ran
"""
import csv, hashlib, json, sys
from pathlib import Path

import numpy as np
import reference_krum as rk
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, Krum
import flwr

FIXTURES = {
    "coincident": [[1., 1, 1, 1], [1.1, 1, 1, 1], [0.9, 1, 1, 1], [1, 1.1, 1, 1], [1, 0.9, 1, 1]],
    "divergent":  [[1., 1, 1, 1], [1.1, 1, 1, 1], [0.9, 1, 1, 1], [1, 1.1, 1, 1], [9, 9, 9, 9]],
}
CLIENTS = [f"c{i}" for i in range(5)]
NUM_EX = [10] * 5
F = 1
TOL = 1e-12

WORLDS = [
    ("A", "krum", "krum",   "coincident", "declared rule ran; the two rules agree on this input"),
    ("B", "krum", "fedavg", "coincident", "declared rule did not run; the two rules agree on this input"),
    ("C", "krum", "fedavg", "divergent",  "declared rule did not run; the two rules disagree on this input"),
    ("D", "krum", "krum",   "divergent",  "declared rule ran; the two rules disagree on this input"),
]
LEVELS = {
    "E0": ["declaration.json"],
    "E1": ["declaration.json", "output.json"],
    "E2": ["declaration.json", "output.json", "inputs.json"],
    "E3": ["declaration.json", "output.json", "inputs.json", "execution.json"],
}

TRACE_SCOPE = "flwr/serverapp/strategy"
CONSISTENT = "consistent"
CONTRADICTED = "contradicted"
UNABLE = "unable_to_determine"
INVALID = "invalid_bundle"


def run_real(rule, updates):
    """Run a real Flower strategy, recording which functions were entered."""
    msgs = []
    for i, u in enumerate(updates):
        content = RecordDict({
            "arrays": ArrayRecord([np.array(u, dtype=np.float64)]),
            "metrics": MetricRecord({"num-examples": 10}),
        })
        msgs.append(Message(content=content, dst_node_id=i + 1, message_type="train"))
    if rule == "krum":
        strat = Krum(min_train_nodes=5, min_available_nodes=5, num_malicious_nodes=F)
    else:
        strat = FedAvg(min_train_nodes=5, min_available_nodes=5)
    seen = []

    def tracer(frame, event, arg):
        if event == "call":
            fn = frame.f_code.co_filename.replace("\\", "/")
            if TRACE_SCOPE in fn:
                mod = fn.split("flwr/")[-1].removesuffix(".py").replace("/", ".")
                seen.append("flwr." + mod + ":" + frame.f_code.co_name)
        return None

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        arrays, _ = strat.aggregate_train(1, msgs)
    finally:
        sys.settrace(prev)
    return arrays.to_numpy_ndarrays()[0], sorted(set(seen))


def dumps(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":")) + "\n"


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def write_bundle(root, level, decl, out, inp, exe):
    avail = {"declaration.json": decl, "output.json": out,
             "inputs.json": inp, "execution.json": exe}
    bd = root / "bundle"
    bd.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in LEVELS[level]:
        t = dumps(avail[name])
        (bd / name).write_text(t)
        files[name] = sha(t)
    (bd / "manifest.json").write_text(dumps({"evidence_level": level, "files": files}))
    return bd


def verify(bd):
    """The reviewer. Sees the bundle directory only; truth.json is out of reach."""
    man_p = bd / "manifest.json"
    if not man_p.is_file():
        return {c: (INVALID, "manifest missing") for c in ("C1", "C2", "C3")}, None
    man = json.loads(man_p.read_text())
    parsed = {}
    for name, digest in man["files"].items():
        p = bd / name
        if not p.is_file():
            return {c: (INVALID, name + " missing") for c in ("C1", "C2", "C3")}, None
        text = p.read_text()
        if sha(text) != digest:
            return {c: (INVALID, name + " digest does not match the manifest")
                    for c in ("C1", "C2", "C3")}, None
        parsed[name] = json.loads(text)

    decl = parsed["declaration.json"]
    res = {"C1": (CONSISTENT, "declares %s with f=%d over %d clients"
                  % (decl["rule"], decl["f"], len(decl["clients"])))}

    diff = None
    if "output.json" not in parsed:
        res["C2"] = (UNABLE, "no recorded output to re-perform against")
    elif "inputs.json" not in parsed:
        res["C2"] = (UNABLE, "no client submissions; the decision cannot be re-performed")
    else:
        inp = parsed["inputs.json"]
        U = [np.array(inp["updates"][c], dtype=np.float64) for c in decl["clients"]]
        N = [inp["num_examples"][c] for c in decl["clients"]]
        if decl["rule"] == "krum":
            replay = rk.krum(U, N, f=decl["f"])[3]
        else:
            replay = rk.fedavg(U, N)
        rec = np.array(parsed["output.json"]["aggregate"], dtype=np.float64)
        diff = float(np.max(np.abs(replay - rec)))
        if diff <= TOL:
            res["C2"] = (CONSISTENT,
                         "re-performance reproduces the recorded aggregate; max abs difference %.3e" % diff)
        else:
            res["C2"] = (CONTRADICTED,
                         "re-performance differs from the recorded aggregate by %.3e" % diff)

    if "execution.json" not in parsed:
        res["C3"] = (UNABLE, "no runtime record; execution is neither supported nor contradicted")
    else:
        fns = parsed["execution.json"]["functions"]
        marker = "select_multikrum" if decl["rule"] == "krum" else "aggregate_arrayrecords"
        hit = any(f.endswith(":" + marker) for f in fns)
        anti = decl["rule"] == "fedavg" and any(f.endswith(":select_multikrum") for f in fns)
        if hit and not anti:
            res["C3"] = (CONSISTENT,
                         "runtime record contains " + marker + "; holds only if the recorder is sound")
        else:
            res["C3"] = (CONTRADICTED, "runtime record does not contain " + marker)
    return res, diff


def main():
    root = Path("evidence")
    results = Path("results")
    root.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    rows = []
    cases = []

    for name, declared, executed, fx, note in WORLDS:
        updates = [np.array(v, dtype=np.float64) for v in FIXTURES[fx]]
        agg, trace = run_real(executed, FIXTURES[fx])
        decl = {"round": 1, "rule": declared, "f": F, "m": 1,
                "clients": CLIENTS, "framework": "flwr " + flwr.__version__}
        out = {"aggregate": [float(x) for x in agg]}
        inp = {"updates": {c: [float(x) for x in u] for c, u in zip(CLIENTS, updates)},
               "num_examples": dict(zip(CLIENTS, NUM_EX))}
        exe = {"flwr": flwr.__version__, "functions": trace}
        wdir = root / name
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "truth.json").write_text(dumps(
            {"world": name, "declared": declared, "executed": executed, "fixture": fx,
             "declared_rule_ran": declared == executed, "note": note}))
        for lv in LEVELS:
            bd = write_bundle(wdir / lv, lv, decl, out, inp, exe)
            cases.append((name + "/" + lv, bd, declared, executed, fx, declared == executed))

    src = root / "D" / "E3" / "bundle"
    tdir = root / "tampered" / "E3" / "bundle"
    tdir.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        (tdir / p.name).write_text(p.read_text())
    o = json.loads((tdir / "output.json").read_text())
    o["aggregate"] = [x + 0.5 for x in o["aggregate"]]
    (tdir / "output.json").write_text(dumps(o))
    (root / "tampered" / "truth.json").write_text(dumps(
        {"world": "tampered", "declared": "krum", "executed": "krum", "fixture": "divergent",
         "declared_rule_ran": True,
         "note": "world D at E3 with output.json edited after the manifest was written"}))
    cases.append(("tampered/E3", tdir, "krum", "krum", "divergent", True))

    for case, bd, declared, executed, fx, ran in cases:
        res, diff = verify(bd)
        for c in ("C1", "C2", "C3"):
            v, why = res[c]
            rows.append({"case": case, "declared": declared, "executed": executed,
                         "fixture": fx, "declared_rule_ran": ran, "claim": c,
                         "verdict": v, "max_abs_diff": "" if diff is None else diff,
                         "reason": why})

    with (results / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    idx = {(r["case"], r["claim"]): r["verdict"] for r in rows}
    print("")
    print("%-14s%-22s%-22s%-22s" % ("case", "C1 declaration", "C2 re-performance", "C3 execution"))
    print("-" * 80)
    for case, bd, declared, executed, fx, ran in cases:
        line = "%-14s" % case
        for c in ("C1", "C2", "C3"):
            line += "%-22s" % idx[(case, c)]
        print(line)
    print("")
    print("Results written to results/summary.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
