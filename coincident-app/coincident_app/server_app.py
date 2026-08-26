import numpy as np
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Krum

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    rounds = int(context.run_config["num-server-rounds"])
    strategy = Krum(fraction_train=1.0, min_train_nodes=5,
                    min_evaluate_nodes=5, min_available_nodes=5,
                    num_malicious_nodes=1)
    result = strategy.start(grid=grid,
                            initial_arrays=ArrayRecord([np.zeros(4, dtype=np.float64)]),
                            num_rounds=rounds)
    print("AGGREGATED =", result.arrays.to_numpy_ndarrays()[0])
