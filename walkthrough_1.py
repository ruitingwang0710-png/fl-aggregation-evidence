"""One aggregation each under FedAvg and Krum, on five fixed client updates.

Clients 0-3 sit close together; client 4 is far away. FedAvg averages everyone,
so the outlier moves the result; Krum selects the single most central update and
returns it unchanged.
"""
import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, Krum

updates = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([1.1, 1.0, 1.0, 1.0]),
    np.array([0.9, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.1, 1.0, 1.0]),
    np.array([9.0, 9.0, 9.0, 9.0]),
]


def to_messages(updates):
    """Wrap bare NumPy vectors as the reply Messages a Flower strategy consumes."""
    msgs = []
    for i, u in enumerate(updates):
        content = RecordDict({
            "arrays":  ArrayRecord([u]),
            "metrics": MetricRecord({"num-examples": 10}),
        })
        msgs.append(Message(content=content, dst_node_id=i + 1, message_type="train"))
    return msgs


for name, strategy in [
    ("FedAvg", FedAvg(min_train_nodes=5, min_available_nodes=5)),
    ("Krum  ", Krum(min_train_nodes=5, min_available_nodes=5, num_malicious_nodes=1)),
]:
    arrays, _ = strategy.aggregate_train(1, to_messages(updates))
    print(name, "->", arrays.to_numpy_ndarrays()[0])
