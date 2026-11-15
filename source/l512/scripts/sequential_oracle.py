#!/usr/bin/env python3
"""Anytime-valid two-sided sign evidence, assuming conditional sign nulls.

This tests a timing median against a FIXED boundary, not bucket correctness.
Under a low null, P(X_t > boundary | past) <= 1/2 at every step;
under a high null, P(X_t < boundary | past) <= 1/2. Ties abstain.
Mixtures of fixed Bernoulli likelihood ratios are e-processes under these
nulls. Split alpha between directions (Ville + union bound). No Gaussian
model, variance estimate, normal approximation or fixed-look p-values.
"""
import numpy as np

THETA = np.array([.51, .525, .55, .6, .7, .8, .9])


def evidence(samples, boundary):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or not np.isfinite(boundary):
        raise ValueError('finite one-dimensional samples and boundary required')
    plus = np.cumsum(x > boundary)[:, None]
    minus = np.cumsum(x < boundary)[:, None]
    hit = np.log(2 * THETA)
    miss = np.log(2 * (1 - THETA))
    def mix(logs):
        m = logs.max(axis=1)
        return m + np.log(np.exp(logs - m[:, None]).mean(axis=1))
    low = mix(minus * hit + plus * miss)
    high = mix(plus * hit + minus * miss)
    return np.column_stack([low, high])


def decide(samples, boundary, alpha=.001, min_samples=32):
    if not 0 < alpha < 1 or min_samples < 1:
        raise ValueError('require 0 < alpha < 1 and min_samples >= 1')
    loge = evidence(samples, boundary)
    if not len(loge):
        return {'side': None, 'samples': 0, 'p_any': 1.0}
    best = loge.max(axis=1)
    crossings = np.flatnonzero((best >= np.log(2 / alpha)) &
                              (np.arange(len(best)) + 1 >= min_samples))
    n = int(crossings[0] + 1) if len(crossings) else len(best)
    side = int(loge[n-1].argmax()) if len(crossings) else None
    p_any = float(np.exp(min(0., np.log(2.) - best[:n].max())))
    return {'side': side, 'samples': n, 'p_any': p_any}
