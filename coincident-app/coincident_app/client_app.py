import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()

# The coincident fixture: one centroid plus two symmetric +/-0.1 pairs.
# Each client returns the row matching its partition id.
FIXTURE = [
    [1.0, 1.0, 1.0, 1.0],
    [1.1, 1.0, 1.0, 1.0],
    [0.9, 1.0, 1.0, 1.0],
    [1.0, 1.1, 1.0, 1.0],
    [1.0, 0.9, 1.0, 1.0],
]


@app.train()
def train(msg: Message, context: Context) -> Message:
    pid = int(context.node_config["partition-id"])
    vec = np.array(FIXTURE[pid], dtype=np.float64)
    content = RecordDict({
        "arrays": ArrayRecord([vec]),
        "metrics": MetricRecord({"num-examples": 10}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    return Message(content=RecordDict({"metrics": MetricRecord({"num-examples": 10, "loss": 0.0})}), reply_to=msg)
