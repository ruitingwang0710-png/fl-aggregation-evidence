import numpy as np


def pairwise_sq_dist(updates):
    flat = [np.asarray(u, dtype=np.float64).ravel() for u in updates]
    n = len(flat)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = flat[i] - flat[j]
            d[i, j] = d[j, i] = float(np.dot(diff, diff))
    return d


def krum_scores(d, f):
    n = d.shape[0]
    k = max(1, n - f - 2)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(d[i], i)
        scores[i] = float(np.sum(np.sort(others)[:k]))
    return scores, k


def krum(updates, num_examples, f, m=1):
    d = pairwise_sq_dist(updates)
    scores, k = krum_scores(d, f)
    chosen = tuple(int(i) for i in np.argsort(scores, kind="stable")[:m])
    w = np.array([float(num_examples[i]) for i in chosen])
    stack = np.stack([np.asarray(updates[i], dtype=np.float64) for i in chosen])
    agg = np.tensordot(w, stack, axes=(0, 0)) / float(w.sum())
    return chosen, scores, k, agg


def fedavg(updates, num_examples):
    w = np.array([float(x) for x in num_examples], dtype=np.float64)
    stack = np.stack([np.asarray(u, dtype=np.float64) for u in updates])
    return np.tensordot(w, stack, axes=(0, 0)) / float(w.sum())
