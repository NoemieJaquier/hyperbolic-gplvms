import numpy as np


# Functions from: https://github.com/pymanopt/pymanopt/pull/123
def mobius_addition(x, y):
    scalar_product = np.sum(x * y, axis=0)
    norm2x = np.sum(x * x, axis=0)
    norm2y = np.sum(y * y, axis=0)

    return (x * (1 + 2 * scalar_product + norm2y) + y * (1 - norm2x)) / (1 + 2 * scalar_product + norm2x * norm2y)


def poincare_expmap(x, u, t=1.0):
    norm_u = np.linalg.norm(u, axis=0)
    # Handle the case where U is null.
    w = u * np.divide(np.tanh(t * norm_u / (1 - np.sum(x * x, axis=0))), norm_u, out=np.zeros_like(u), where=norm_u != 0,)
    return mobius_addition(x, w)


def poincare_logmap(x, y):
    w = mobius_addition(-x, y)
    norm_w = np.linalg.norm(w, axis=0)
    return (1 - np.sum(x * x, axis=0)) * np.arctanh(norm_w) * w / norm_w