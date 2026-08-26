"""Assert bit-exact equality of Flower Krum and FedAvg on the coincident fixture."""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg, Krum


UPDATES = [
    np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    np.array([1.1, 1.0, 1.0, 1.0], dtype=np.float64),
    np.array([0.9, 1.0, 1.0, 1.0], dtype=np.float64),
    np.array([1.0, 1.1, 1.0, 1.0], dtype=np.float64),
    np.array([1.0, 0.9, 1.0, 1.0], dtype=np.float64),
]


def to_messages(updates: list[np.ndarray]) -> list[Message]:
    return [
        Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord([update]),
                    "metrics": MetricRecord({"num-examples": 10}),
                }
            ),
            dst_node_id=index + 1,
            message_type="train",
        )
        for index, update in enumerate(updates)
    ]


def aggregate(strategy: FedAvg | Krum) -> np.ndarray:
    arrays, _ = strategy.aggregate_train(1, to_messages(UPDATES))
    return arrays.to_numpy_ndarrays()[0]


def main() -> None:
    fedavg = aggregate(FedAvg(min_train_nodes=5, min_available_nodes=5))
    krum = aggregate(
        Krum(
            min_train_nodes=5,
            min_available_nodes=5,
            num_malicious_nodes=1,
        )
    )

    same_values = np.array_equal(fedavg, krum)
    same_uint64 = np.array_equal(fedavg.view(np.uint64), krum.view(np.uint64))
    same_bytes = fedavg.tobytes(order="C") == krum.tobytes(order="C")

    print("FedAvg:", fedavg)
    print("Krum:  ", krum)
    print("FedAvg uint64:", [f"0x{value:016x}" for value in fedavg.view(np.uint64)])
    print("Krum   uint64:", [f"0x{value:016x}" for value in krum.view(np.uint64)])
    print("same_values:", same_values)
    print("same_uint64:", same_uint64)
    print("same_bytes:", same_bytes)
    print("max_abs_difference:", float(np.max(np.abs(fedavg - krum))))

    assert same_values
    assert same_uint64
    assert same_bytes


if __name__ == "__main__":
    main()
