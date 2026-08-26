"""Alignment check: does the independent implementation agree with real Flower?

"Independent re-performance" only means something if the independent
implementation reproduces the rule under test. If the two disagree, the replay
is not replaying the same rule.
"""
import numpy as np
import reference_krum as rk
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, Krum

COINCIDENT = [[1., 1, 1, 1], [1.1, 1, 1, 1], [0.9, 1, 1, 1], [1, 1.1, 1, 1], [1, 0.9, 1, 1]]
DIVERGENT  = [[1., 1, 1, 1], [1.1, 1, 1, 1], [0.9, 1, 1, 1], [1, 1.1, 1, 1], [9, 9, 9, 9]]
N = [10] * 5


def flower(rule, updates):
    msgs = [Message(content=RecordDict({
                "arrays": ArrayRecord([np.array(u, dtype=np.float64)]),
                "metrics": MetricRecord({"num-examples": 10})}),
            dst_node_id=i + 1, message_type="train")
            for i, u in enumerate(updates)]
    s = (Krum(min_train_nodes=5, min_available_nodes=5, num_malicious_nodes=1)
         if rule == "krum" else
         FedAvg(min_train_nodes=5, min_available_nodes=5))
    a, _ = s.aggregate_train(1, msgs)
    return a.to_numpy_ndarrays()[0]


for name, fx in [("coincident", COINCIDENT), ("divergent", DIVERGENT)]:
    U = [np.array(v, dtype=np.float64) for v in fx]
    chosen, scores, k, ref_krum = rk.krum(U, N, f=1)
    ref_avg = rk.fedavg(U, N)
    flw_krum = flower("krum", fx)
    flw_avg  = flower("fedavg", fx)
    gap = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    print(f"\n===== {name} =====")
    print(f"  neighbourhood k = {k}   scores = {np.round(scores, 4)}   selected = {chosen}")
    print(f"  independent Krum   = {ref_krum}")
    print(f"  Flower      Krum   = {flw_krum}    difference = {gap(ref_krum, flw_krum)}")
    print(f"  independent FedAvg = {ref_avg}")
    print(f"  Flower      FedAvg = {flw_avg}    difference = {gap(ref_avg, flw_avg)}")
    print(f"  >>> Krum vs FedAvg output gap = {gap(flw_krum, flw_avg)}")
