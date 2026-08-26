"""M1: does the substitution-separability gap shrink in a real converging
federated run?

Machinery only.  `m1_run.py` drives the arms.

The federated loop is real: real data (UCI optical-recognition-of-handwritten-
digits, shipped with scikit-learn), a real model trained by real local SGD on
real client partitions, and aggregation performed by the *unmodified* Flower
1.33.0 strategy objects -- the same objects a deployment would call.

At every round we record, on the round's actual client updates u_t:
  * dispersion of u_t, several ways
  * ||A_declared(u_t) - FedAvg(u_t)||_inf for each of four declared rules
  * ||ref_A(u_t) - A(u_t)||_inf, the re-implementation noise floor, measured on
    the same vectors so it carries the same dimension and the same magnitude
"""
import logging
for _n in list(logging.root.manager.loggerDict) + ["flwr"]:
    logging.getLogger(_n).disabled = True

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import (FedAvg, Krum, MultiKrum, FedMedian,
                                     FedTrimmedAvg)
import reference_rules as rr

F_MALICIOUS = 1
BETA = 0.2
M_MULTIKRUM = 3
RULES = ("krum", "multikrum3", "median", "trimmed")


# ------------------------------------------------------------------- data ----
def load_data(seed):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = np.asarray(d.data, dtype=np.float64) / 16.0
    y = np.asarray(d.target, dtype=np.int64)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    ntest = int(0.2 * len(y))
    return X[ntest:], y[ntest:], X[:ntest], y[:ntest]


def partition(y, n_clients, mode, seed, min_per_client=12, shared=0.0):
    """Split the training set across clients.

    `shared` is the fraction of each client's local dataset drawn from a single
    common pool.  shared=0 gives disjoint clients; shared=1 gives clients with
    identical data.  It exists to answer *how much* homogenisation would be
    needed to reach the ambiguous regime, rather than assuming an answer.
    """
    rng = np.random.default_rng(2000 + seed)
    n = len(y)
    if shared > 0.0:
        m = n // n_clients
        k = int(round(shared * m))
        idx = rng.permutation(n)
        pool, rest = idx[:k], idx[k:]
        priv = np.array_split(rest, n_clients)
        return [np.sort(np.concatenate([pool, priv[i][:m - k]]))
                for i in range(n_clients)]
    if mode == "iid":
        idx = rng.permutation(n)
        return [np.sort(a) for a in np.array_split(idx, n_clients)]
    alpha = float(mode.split("dir")[1])
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            ic = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(ic, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_per_client:
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"could not partition with alpha={alpha}")


# ------------------------------------------------------------------ models ---
class LogReg:
    """Multinomial logistic regression.  64 -> 10, 650 parameters."""
    name, n_layers = "logreg", 2

    def __init__(self, d_in=64, k=10, seed=0):
        rng = np.random.default_rng(3000 + seed)
        self.p = [rng.normal(0, 0.01, (d_in, k)), np.zeros(k)]

    @staticmethod
    def _fwd(p, X):
        z = X @ p[0] + p[1]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def grads(p, X, y, k=10):
        P = LogReg._fwd(p, X)
        G = P.copy()
        G[np.arange(len(y)), y] -= 1.0
        G /= len(y)
        return [X.T @ G, G.sum(axis=0)]

    @staticmethod
    def loss_acc(p, X, y):
        P = LogReg._fwd(p, X)
        ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean()
        return float(ll), float((P.argmax(1) == y).mean())


class MLP:
    """64 -> 32 (tanh) -> 10.  2410 parameters."""
    name, n_layers = "mlp", 4

    def __init__(self, d_in=64, h=32, k=10, seed=0):
        rng = np.random.default_rng(3000 + seed)
        self.p = [rng.normal(0, np.sqrt(2.0 / (d_in + h)), (d_in, h)), np.zeros(h),
                  rng.normal(0, np.sqrt(2.0 / (h + k)), (h, k)), np.zeros(k)]

    @staticmethod
    def _fwd(p, X):
        H = np.tanh(X @ p[0] + p[1])
        z = H @ p[2] + p[3]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return H, e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def grads(p, X, y, k=10):
        H, P = MLP._fwd(p, X)
        G = P.copy()
        G[np.arange(len(y)), y] -= 1.0
        G /= len(y)
        dW2, db2 = H.T @ G, G.sum(axis=0)
        dH = (G @ p[2].T) * (1.0 - H * H)
        return [X.T @ dH, dH.sum(axis=0), dW2, db2]

    @staticmethod
    def loss_acc(p, X, y):
        _, P = MLP._fwd(p, X)
        ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean()
        return float(ll), float((P.argmax(1) == y).mean())


MODELS = {"logreg": LogReg, "mlp": MLP}


def local_train(cls, p0, X, y, epochs, lr, batch, rng):
    p = [a.copy() for a in p0]
    n = len(y)
    for _ in range(epochs):
        if batch is None:                      # full-batch gradient descent
            g = cls.grads(p, X, y)
            for i in range(len(p)):
                p[i] -= lr * g[i]
        else:
            order = rng.permutation(n)
            for s in range(0, n, batch):
                b = order[s:s + batch]
                if len(b) == 0:
                    continue
                g = cls.grads(p, X[b], y[b])
                for i in range(len(p)):
                    p[i] -= lr * g[i]
    return p


# ------------------------------------------------------------- aggregation ---
def _msgs(updates, nex):
    return [Message(content=RecordDict({
        "arrays": ArrayRecord([np.ascontiguousarray(a, dtype=np.float64)
                               for a in u]),
        "metrics": MetricRecord({"num-examples": int(m)})}),
        dst_node_id=i + 1, message_type="train")
        for i, (u, m) in enumerate(zip(updates, nex))]


def _strategy(rule, n):
    if rule == "fedavg":
        return FedAvg(min_train_nodes=n, min_available_nodes=n)
    if rule == "krum":
        return Krum(min_train_nodes=n, min_available_nodes=n,
                    num_malicious_nodes=F_MALICIOUS)
    if rule == "multikrum3":
        return MultiKrum(min_train_nodes=n, min_available_nodes=n,
                         num_malicious_nodes=F_MALICIOUS,
                         num_nodes_to_select=M_MULTIKRUM)
    if rule == "median":
        return FedMedian(min_train_nodes=n, min_available_nodes=n)
    if rule == "trimmed":
        return FedTrimmedAvg(min_train_nodes=n, min_available_nodes=n, beta=BETA)
    raise KeyError(rule)


def flower_aggregate(rule, updates, nex):
    """Run the unmodified Flower strategy.  Returns a list of layer arrays."""
    a, _ = _strategy(rule, len(updates)).aggregate_train(1, _msgs(updates, nex))
    return a.to_numpy_ndarrays()


def reference_aggregate(rule, updates, nex):
    """Independent NumPy implementation, per layer, no flwr import."""
    L = len(updates[0])
    out = []
    for j in range(L):
        layers = [u[j] for u in updates]
        if rule == "fedavg":
            out.append(rr.fedavg(layers, nex))
        elif rule == "median":
            out.append(rr.median(layers))
        elif rule == "trimmed":
            out.append(rr.trimmed_mean(layers, BETA))
        else:
            out.append(None)                   # krum handled below
    if rule in ("krum", "multikrum3"):
        m = 1 if rule == "krum" else M_MULTIKRUM
        flat = [np.concatenate([a.ravel() for a in u]) for u in updates]
        chosen, _, _, _ = rr.krum(flat, nex, f=F_MALICIOUS, m=m)
        w = np.array([float(nex[i]) for i in chosen])
        out = []
        for j in range(L):
            stack = np.stack([updates[i][j] for i in chosen])
            out.append(np.tensordot(w, stack, axes=(0, 0)) / float(w.sum()))
    return out


def linf(a, b):
    return float(max(np.max(np.abs(x - y)) for x, y in zip(a, b)))


def krum_selection(updates, nex):
    """Which client does each implementation select, and by what margin?

    Flower computes squared distances as ||x||^2 + ||y||^2 - 2 x.y; the
    reference computes ||x - y||^2 directly.  The two agree in exact
    arithmetic and can disagree in floating point once clients are close,
    because the expansion cancels catastrophically.  Recording both lets us
    see whether that ever changes which update Krum returns.
    """
    from flwr.serverapp.strategy.multikrum import select_multikrum
    msgs = _msgs(updates, nex)
    contents = [m.content for m in msgs]
    sel_f = select_multikrum(contents, F_MALICIOUS, 1)
    idx_f = next(i for i, c in enumerate(contents) if c is sel_f[0])

    flat = [np.concatenate([a.ravel() for a in u]) for u in updates]
    chosen, scores, _, _ = rr.krum(flat, nex, f=F_MALICIOUS, m=1)
    s = np.sort(scores)
    return {
        "krum_idx_flower": idx_f,
        "krum_idx_ref": int(chosen[0]),
        "krum_sel_agree": int(idx_f == int(chosen[0])),
        "krum_score_best": float(s[0]),
        "krum_score_margin_rel": float((s[1] - s[0]) / max(s[1], 1e-300)),
    }


# -------------------------------------------------------------- dispersion ---
def dispersion(updates, global_p):
    flat = np.stack([np.concatenate([a.ravel() for a in u]) for u in updates])
    g = np.concatenate([a.ravel() for a in global_p])
    mean = flat.mean(axis=0)
    delta = flat - g                              # per-client local update
    dmean = delta.mean(axis=0)
    return {
        "disp_rms_coord": float(np.sqrt(np.mean(flat.var(axis=0)))),
        "disp_mean_l2": float(np.mean(np.linalg.norm(flat - mean, axis=1))),
        "disp_max_l2": float(np.max(np.linalg.norm(flat - mean, axis=1))),
        "disp_rel_to_weights": float(np.sqrt(np.mean(flat.var(axis=0)))
                                     / max(float(np.sqrt(np.mean(mean ** 2))), 1e-300)),
        "delta_rms": float(np.sqrt(np.mean(dmean ** 2))),
        "delta_disp_rel": float(np.sqrt(np.mean(delta.var(axis=0)))
                                / max(float(np.sqrt(np.mean(dmean ** 2))), 1e-300)),
        "weights_rms": float(np.sqrt(np.mean(mean ** 2))),
        "min_pair_l2": float(np.min([np.linalg.norm(flat[i] - flat[j])
                                     for i in range(len(flat))
                                     for j in range(i + 1, len(flat))])),
    }


# ------------------------------------------------------------------- a run ---
def run(model_name, partition_mode, optimiser, seed, rounds=100, n_clients=16,
        epochs=3, lr=0.5, driver="fedavg", shared=0.0):
    Xtr, ytr, Xte, yte = load_data(seed)
    parts = partition(ytr, n_clients, partition_mode, seed, shared=shared)
    nex = [len(p) for p in parts]
    cls = MODELS[model_name]
    model = cls(seed=seed)
    gp = [a.copy() for a in model.p]
    batch = None if optimiser == "fullbatch" else 32
    rng = np.random.default_rng(4000 + seed)

    rows = []
    for t in range(1, rounds + 1):
        updates = [local_train(cls, gp, Xtr[p], ytr[p], epochs, lr, batch, rng)
                   for p in parts]

        row = {"model": model_name, "partition": partition_mode,
               "optimiser": optimiser, "driver": driver, "seed": seed,
               "shared": shared, "round": t, "n_clients": n_clients}
        row.update(dispersion(updates, gp))
        row.update(krum_selection(updates, nex))

        agg = {r: flower_aggregate(r, updates, nex)
               for r in ("fedavg",) + RULES}
        for r in RULES:
            row[f"gap_{r}"] = linf(agg[r], agg["fedavg"])
            row[f"relgap_{r}"] = row[f"gap_{r}"] / max(
                float(max(np.max(np.abs(a)) for a in agg["fedavg"])), 1e-300)
        for r in ("fedavg",) + RULES:
            row[f"noise_{r}"] = linf(reference_aggregate(r, updates, nex), agg[r])

        gp = [a.copy() for a in agg[driver]]
        tl, ta = cls.loss_acc(gp, Xtr, ytr)
        vl, va = cls.loss_acc(gp, Xte, yte)
        row.update(train_loss=tl, train_acc=ta, test_loss=vl, test_acc=va)
        rows.append(row)
    return rows
