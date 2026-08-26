"""Independent re-implementations of the four declared aggregation rules.

Deliberately imports nothing from flwr.  Written from the published definitions,
not from Flower's source, and using different primitives where a choice exists
(sort rather than partition; explicit self-removal rather than dropping the
first sorted element).  The point is to measure what an honest, competent
re-implementation differs from Flower by -- the noise floor that any tolerance
must clear before it can be said to detect anything.
"""
import numpy as np


# ---------------------------------------------------------------- distances --
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
        others = np.delete(d[i], i)          # explicit self-removal
        scores[i] = float(np.sum(np.sort(others)[:k]))
    return scores, k


# -------------------------------------------------------------------- rules --
def fedavg(updates, num_examples):
    w = np.asarray(num_examples, dtype=np.float64)
    stack = np.stack([np.asarray(u, dtype=np.float64) for u in updates])
    return np.tensordot(w, stack, axes=(0, 0)) / float(w.sum())


def krum(updates, num_examples, f=1, m=1):
    d = pairwise_sq_dist(updates)
    scores, k = krum_scores(d, f)
    chosen = tuple(int(i) for i in np.argsort(scores, kind="stable")[:m])
    w = np.array([float(num_examples[i]) for i in chosen])
    stack = np.stack([np.asarray(updates[i], dtype=np.float64) for i in chosen])
    agg = np.tensordot(w, stack, axes=(0, 0)) / float(w.sum())
    return chosen, scores, k, agg


def median(updates, num_examples=None):
    """Coordinate-wise median.  Sort-based rather than partition-based."""
    stack = np.stack([np.asarray(u, dtype=np.float64) for u in updates])
    s = np.sort(stack, axis=0)
    n = s.shape[0]
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def trimmed_mean(updates, beta=0.2):
    """Coordinate-wise trimmed mean, cutting floor(beta*n) from each tail.

    Sort-based; the surviving block is summed then divided, rather than
    np.mean'd over a partitioned view.
    """
    stack = np.stack([np.asarray(u, dtype=np.float64) for u in updates])
    n = stack.shape[0]
    lo = int(beta * n)
    hi = n - lo
    if lo > hi:
        raise ValueError("beta too large")
    s = np.sort(stack, axis=0)[lo:hi]
    return np.sum(s, axis=0) / float(s.shape[0])
