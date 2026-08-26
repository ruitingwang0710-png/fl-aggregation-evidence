"""The coincident fixture: FedAvg and Krum produce the same output.

One centroid plus two symmetric +/-0.1 pairs. The centroid is therefore both the
arithmetic mean of all five updates and the update Krum selects, so the two rules
agree -- here, to the last bit.
"""
import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, Krum

updates = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([1.1, 1.0, 1.0, 1.0]),
    np.array([0.9, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.1, 1.0, 1.0]),
    np.array([1.0, 0.9, 1.0, 1.0]),
]


def to_messages(u):
    return [Message(content=RecordDict({"arrays": ArrayRecord([x]),
            "metrics": MetricRecord({"num-examples": 10})}),
            dst_node_id=i + 1, message_type="train") for i, x in enumerate(u)]


out = {}
for name, s in [("FedAvg", FedAvg(min_train_nodes=5, min_available_nodes=5)),
                ("Krum",   Krum(min_train_nodes=5, min_available_nodes=5, num_malicious_nodes=1))]:
    a, _ = s.aggregate_train(1, to_messages(updates))
    out[name] = a.to_numpy_ndarrays()[0]
    print(f"{name:7} -> {out[name]}")

print()
print("identical in every bit:", np.array_equal(out["FedAvg"], out["Krum"]))
print("max absolute difference:", float(np.max(np.abs(out["FedAvg"] - out["Krum"]))))
print("FedAvg exact bits:", [x.hex() for x in out["FedAvg"]])
print("Krum   exact bits:", [x.hex() for x in out["Krum"]])
