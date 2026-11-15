"""Fixed, label-free ranking of voted guesses using every timing capture.

Fit bias and MAD per capture and column from valid raw guesses. Sum the three
standardized squared-distance evidences, then compare the EXISTING voted class
to its strongest rival. No class prior, truth-based tuning, or vote replacement.
This is a ranking heuristic, not a calibrated probability: captures may share
systematic noise. Matching public tables does not establish key/cache identity;
capture provenance must pin those separately.
"""
import numpy as np
from evaluate_confidence import read_log, recovery_prediction
from vote3_merge import PROBE_FIELDS


def load_runs(paths, merged):
    """Read only observable features, align by coordinates, check compatibility."""
    if len(paths) < 2:
        raise ValueError('joint confidence requires at least two captures')
    if 'vote_runs' not in merged or np.any(merged['vote_runs'] != len(paths)):
        raise ValueError('capture count differs from marked vote_runs')
    keys = list(zip(merged['coord'], merged['position']))
    runs = []
    for path in paths:
        raw, _ = read_log(path, include_truth=False)
        lookup = {key: i for i, key in enumerate(zip(raw['coord'], raw['position']))}
        if set(lookup) != set(keys):
            raise ValueError(f'{path}: coordinate set differs from merged log')
        order = np.array([lookup[key] for key in keys])
        raw = {key: value[order] for key, value in raw.items()}
        if 'vote_runs' in raw:
            raise ValueError('joint confidence requires raw captures, not merged logs')
        for field in PROBE_FIELDS:
            if not np.array_equal(raw[field], merged[field]):
                raise ValueError(f'{path}: probe table differs ({field})')
        runs.append(raw)
    return runs


def joint_score(merged, runs, fit_mask=None):
    """Larger means stronger support for the voted class; negative is suspect.

    Default fits all observable positions (unsupervised). fit_mask permits
    sensitivity/held-out checks without changing any scoring hyperparameters.
    MAD floor 0.05 and minimum group size 20 match evaluate_confidence.py.
    Invalid raw guesses are excluded from fitting, but their timing still counts.
    """
    pred = recovery_prediction(merged, 'logged')
    n = len(pred)
    if not runs:
        raise ValueError('no raw captures')
    fit_mask = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if fit_mask.shape != (n,):
        raise ValueError('fit mask shape mismatch')
    evidences = []
    for raw in runs:
        for key in ('coord', 'position', *PROBE_FIELDS):
            if not np.array_equal(raw[key], merged[key]):
                raise ValueError(f'unaligned capture or differing probe table: {key}')
        xy = np.column_stack([raw['median_a_ns'], raw['median_b_ns']]) / raw['gap_ns'][:, None]
        centers = np.stack([
            np.column_stack([raw['a_' + s], raw['b_' + s]]) - raw['ref_bucket'][:, None]
            for s in ('neg', 'zero', 'pos')], axis=1)
        valid = np.isin(raw['guess'], [-1, 0, 1]) & fit_mask
        if valid.sum() < 20:
            raise ValueError('fewer than 20 valid observations for nuisance fitting')
        own = np.clip(raw['guess'].astype(int) + 1, 0, 2)
        residual = xy - centers[np.arange(n), own]
        offset, scale = np.zeros_like(xy), np.ones_like(xy)
        for coord in np.unique(raw['coord']):
            mask = raw['coord'] == coord
            fit = mask & valid
            if fit.sum() < 20:
                fit = valid
            bias = np.median(residual[fit], axis=0)
            mad = 1.4826 * np.median(np.abs(residual[fit] - bias), axis=0)
            offset[mask], scale[mask] = bias, np.maximum(mad, 0.05)
        delta = (xy[:, None, :] - offset[:, None, :] - centers) / scale[:, None, :]
        evidences.append(-0.5 * np.square(delta).sum(axis=2))
    # Sorting summands makes aggregation invariant to capture order, even in FP.
    evidence = np.sort(np.stack(evidences), axis=0).sum(axis=0)
    own = evidence[np.arange(n), pred + 1].copy()
    evidence[np.arange(n), pred + 1] = -np.inf
    score = own - evidence.max(axis=1)
    if not np.isfinite(score).all():
        raise ValueError('nonfinite joint confidence')
    return score
