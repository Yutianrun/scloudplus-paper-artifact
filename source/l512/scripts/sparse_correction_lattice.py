"""Primal embedding retaining the sparse correction delta in the short vector.

Unlike eliminating delta and reducing only equation noise, this embedding seeks
[-delta, e, 1] with small norm: sparsity of timing errors is used directly.
Reduction is heuristic; acceptance checks ALL public equations and ternary S.
"""
import numpy as np


def build_sparse(A, b, guess, support, equation_count=100, q=1024):
    A = np.asarray(A, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    guess = np.asarray(guess, dtype=np.int64)
    support = np.sort(np.asarray(support, dtype=np.int64))
    if (A.ndim != 2 or b.shape != (A.shape[0],) or guess.shape != (A.shape[1],)
            or not np.isin(guess, [-1, 0, 1]).all()
            or len(np.unique(support)) != len(support)
            or np.any((support < 0) | (support >= len(guess)))
            or equation_count < 1):
        raise ValueError('invalid sparse-correction input')
    rows = np.arange(min(equation_count, len(A)))
    k, m = len(support), len(rows)
    residual = (b-A@guess) % q
    basis = np.zeros((k+m+1, k+m+1), dtype=np.int64)
    basis[:k, :k] = np.eye(k, dtype=np.int64)
    basis[:k, k:k+m] = A[rows][:, support].T % q
    basis[k:k+m, k:k+m] = q*np.eye(m, dtype=np.int64)
    basis[-1, k:k+m] = residual[rows]
    basis[-1, -1] = 1
    return basis, support


def extract_sparse(reduced_columns, A, b, guess, support, q=1024):
    k = len(support)
    for vector in reduced_columns.T:
        sign = int(vector[-1])
        if abs(sign) != 1:
            continue
        oriented = vector*sign
        if np.any(np.abs(oriented[k:-1]) > 1):
            continue
        delta = -oriented[:k]
        if np.any(np.abs(delta) > 2):
            continue
        s = guess.copy()
        s[support] += delta
        if not np.isin(s, [-1, 0, 1]).all():
            continue
        residual = (b-A@s+q//2) % q-q//2
        if np.all(np.abs(residual) <= 1):
            return s
    return None
