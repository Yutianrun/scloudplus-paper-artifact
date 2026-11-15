#!/usr/bin/env python3
"""Rank existing e2e predictions without using truth in confidence features.

Usage: python3 scripts/evaluate_confidence.py LOG... --out DIRECTORY
Requires numpy. Mixed stdout/stderr logs are accepted. Invalid guesses rank last.
Metric selection uses a fixed random half; the other half is a within-run holdout,
not an independent key/run. Scores never use truth, noiseless buckets or channel_ok.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

FEATURES = ['coord', 'position', 'worker', 'guess', 'median_a_ns', 'median_b_ns',
            'gap_ns', 'ref_bucket', 'a_neg', 'a_zero', 'a_pos',
            'b_neg', 'b_zero', 'b_pos']


def read_log(path, *, include_truth=True):
    rows, header = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith('coord,position,true,guess,'):
            header = line.split(',')
        elif header and line and line[0].isdigit():
            fields = line.split(',')
            if len(fields) != len(header):
                raise ValueError(f'{path}: malformed data row: {line[:100]}')
            rows.append(dict(zip(header, fields)))
    if not rows:
        raise ValueError(f'{path}: no e2e CSV rows')
    data = {k: np.array([float(row[k]) for row in rows]) for k in FEATURES}
    if 'vote_runs' in header:
        data['vote_runs'] = np.array([int(row['vote_runs']) for row in rows])
    if any(not np.isfinite(x).all() for x in data.values()) or np.any(data['gap_ns'] <= 0):
        raise ValueError(f'{path}: nonfinite feature or nonpositive calibration gap')
    if len(set(zip(data['coord'], data['position']))) != len(rows):
        raise ValueError(f'{path}: duplicate coordinate/position')
    truth = np.array([int(row['true']) for row in rows]) if include_truth else None
    return data, truth


def recovery_prediction(f, mode='auto'):
    """Preserve marked vote outputs; nearest-center replay is for raw captures."""
    voted = 'vote_runs' in f
    if mode == 'logged' or (mode == 'auto' and voted):
        if not np.isin(f['guess'], [-1, 0, 1]).all():
            raise ValueError('logged prediction contains invalid/abstained guesses')
        return f['guess'].astype(np.int64).copy()
    if voted:
        raise ValueError('nearest-center replay would overwrite the voted guesses')
    if mode not in ('auto', 'nearest'):
        raise ValueError(f'unknown prediction mode: {mode}')
    xy = np.column_stack([f['median_a_ns'], f['median_b_ns']]) / f['gap_ns'][:, None]
    centers = np.stack([np.column_stack([f['a_'+s], f['b_'+s]]) - f['ref_bucket'][:, None]
                        for s in ['neg', 'zero', 'pos']], axis=1)
    return ((xy[:, None, :] - centers)**2).sum(axis=2).argmin(axis=1) - 1


def confidence_scores(f, transform=None, prior=None):
    """Only observable timing, calibration, probe table and existing guess enter."""
    xy = np.column_stack([f['median_a_ns'], f['median_b_ns']]) / f['gap_ns'][:, None]
    centers = np.stack([np.column_stack([f['a_'+s], f['b_'+s]]) - f['ref_bucket'][:, None]
                        for s in ['neg', 'zero', 'pos']], axis=1)
    valid = np.isin(f['guess'], [-1, 0, 1])
    idx = np.clip(f['guess'].astype(int) + 1, 0, 2)
    rr = np.arange(len(idx))
    if transform is not None:
        offset, scale = transform
        xy = (xy - offset) / scale
        centers = centers / scale[:, None, :]
    delta = xy[:, None, :] - centers
    l1 = np.abs(delta).sum(axis=2)
    l2sq = (delta ** 2).sum(axis=2)
    linf = np.abs(delta).max(axis=2)
    scores = {}
    for name, d in [('l1', l1), ('l2', np.sqrt(l2sq)), ('l2sq', l2sq), ('linf', linf)]:
        own = d[rr, idx]
        rivals = d.copy()
        rivals[rr, idx] = np.inf
        other = rivals.min(axis=1)
        scores[name+'_residual'] = -own
        scores[name+'_margin'] = other - own
        scores[name+'_ratio'] = (other - own) / (other + own + 1e-12)
    own_center = centers[rr, idx]
    sep = np.sqrt(((centers - own_center[:, None, :])**2).sum(axis=2))
    diff = l2sq - l2sq[rr, idx, None]
    boundary = np.full_like(diff, np.inf)
    np.divide(diff, 2 * sep, out=boundary, where=sep > 0)
    scores['boundary_distance'] = boundary.min(axis=1)
    axis = []
    for j in range(2):
        distances = np.abs(delta[:, :, j])
        own = distances[rr, idx]
        alternatives = distances.copy()
        alternatives[centers[:, :, j] == own_center[:, j, None]] = np.inf
        axis.append(alternatives.min(axis=1) - own)
    scores['axis_min_margin'] = np.minimum(*axis)
    scores['axis_min_margin_ns'] = scores['axis_min_margin'] * f['gap_ns']
    for penalty in [0.25, 0.5, 1, 2, 4]:
        scores[f'boundary_residual_penalty_{penalty:g}'] = (
            scores['boundary_distance'] + penalty * scores['l2_residual'])
    # Equal-prior Gaussian likelihood, plus explicit outlier penalty variants.
    # These are ranking scores, not calibrated probabilities of correctness.
    for sigma in [0.1, 0.2, 0.3, 0.5, 1.0]:
        logits = -l2sq / (2 * sigma**2)
        logden = np.logaddexp.reduce(logits, axis=1)
        scores[f'gaussian_logprob_{sigma:g}'] = logits[rr, idx] - logden
    if prior is not None:
        for weight in [0.25, 0.5, 1, 2, 4, 8, 16]:
            logits = -l2sq + weight * np.log(prior)[None, :]
            own = logits[rr, idx].copy()
            logits[rr, idx] = -np.inf
            scores[f'prior_margin_{weight:g}'] = own - logits.max(axis=1)
    for score in scores.values():
        score[~valid] = -np.inf
        if np.isnan(score).any():
            raise ValueError('NaN confidence')
    return scores



def corrected_scores(f, fit_mask):
    """Robust nuisance fits on selection positions, using predictions, never truth.

    Existing guesses stay unchanged. A calibration error that changes a guess
    cannot be silently repaired by this confidence-only evaluation.
    """
    counts = np.array([(f['guess'][fit_mask] == s).sum() for s in [-1, 0, 1]]) + 1
    prior = counts / counts.sum()
    scores = confidence_scores(f, prior=prior)
    xy = np.column_stack([f['median_a_ns'], f['median_b_ns']]) / f['gap_ns'][:, None]
    centers = np.stack([np.column_stack([f['a_'+s], f['b_'+s]]) - f['ref_bucket'][:, None]
                        for s in ['neg', 'zero', 'pos']], axis=1)
    idx = np.clip(f['guess'].astype(int) + 1, 0, 2)
    residual = xy - centers[np.arange(len(idx)), idx]
    valid_fit = fit_mask & np.isin(f['guess'], [-1, 0, 1])
    for grouping in ['global', 'worker', 'coord', 'coord_worker']:
        if grouping == 'global':
            groups = np.zeros(len(idx))
        elif grouping == 'coord_worker':
            groups = f['coord'] * (f['worker'].max() + 1) + f['worker']
        else:
            groups = f[grouping]
        offset = np.zeros_like(xy)
        scale = np.ones_like(xy)
        for group in np.unique(groups):
            mask = groups == group
            fit = mask & valid_fit
            if fit.sum() < 20:
                fit = valid_fit
            bias = np.median(residual[fit], axis=0)
            mad = np.median(np.abs(residual[fit] - bias), axis=0) * 1.4826
            offset[mask] = bias
            scale[mask] = np.maximum(mad, 0.05)
        for variant, transform in [('centered', (offset, np.ones_like(scale))),
                                   ('standardized', (offset, scale))]:
            for name, value in confidence_scores(f, transform, prior).items():
                scores[f'{grouping}_{variant}/{name}'] = value
    return scores


def evaluate(score, correct, coord, pos, mask=None, coverage=0.8):
    ids = np.arange(len(score)) if mask is None else np.flatnonzero(mask)
    # Fixed observable tie break; never sort by correctness.
    order = ids[np.lexsort((pos[ids], coord[ids], -score[ids]))]
    n = len(order)
    if not n:
        raise ValueError('empty evaluation split')
    k = math.ceil(coverage * n)
    errors = np.flatnonzero(~correct[order])
    prefix = int(errors[0]) if len(errors) else n
    selected = order[:k]
    return {'n': n, 'selected': k, 'errors': int((~correct[selected]).sum()),
            'accuracy': float(correct[selected].mean()),
            'zero_error_prefix': prefix, 'zero_error_coverage': prefix/n}, order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--coverage', type=float, default=0.8)
    p.add_argument('--sensitivity-seeds', type=int, default=0,
                   help='Refit exploratory metric using seeds 0..N-1 (no reselection)')
    args = p.parse_args()
    if args.sensitivity_seeds < 0:
        p.error('--sensitivity-seeds must be nonnegative')
    if not 0 < args.coverage <= 1:
        p.error('--coverage must be in (0, 1]')
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, all_results, sensitivity = [], [], {}
    for path in args.logs:
        f, truth = read_log(path)
        selection = np.zeros(len(truth), dtype=bool)
        selection[np.random.default_rng(20261020).permutation(len(truth))[:len(truth)//2]] = True
        scores = corrected_scores(f, selection)
        correct = f['guess'] == truth
        results = {}
        for name, score in scores.items():
            results[name] = {split: evaluate(score, correct, f['coord'], f['position'], mask, args.coverage)[0]
                             for split, mask in [('all', None), ('selection', selection), ('holdout', ~selection)]}
            for split, stats in results[name].items():
                all_results.append({'dataset': path.stem, 'metric': name, 'split': split, **stats})
        best = min(results, key=lambda name: (results[name]['selection']['errors'],
                   -results[name]['selection']['zero_error_coverage'], name))
        observed = min(results, key=lambda name: (results[name]['all']['errors'],
                       -results[name]['all']['zero_error_coverage'], name))
        for role, metric in [('selected', best), ('exploratory', observed)]:
            stats, order = evaluate(scores[metric], correct, f['coord'], f['position'], coverage=args.coverage)
            with (args.out / (path.stem + '-' + role + '-ranked.csv')).open('w') as out:
                writer = csv.writer(out)
                writer.writerow(['rank', 'coord', 'position', 'guess', 'confidence', 'selected', 'true', 'correct'])
                for rank, i in enumerate(order, 1):
                    writer.writerow([rank, int(f['coord'][i]), int(f['position'][i]), int(f['guess'][i]),
                                     scores[metric][i], int(rank <= stats['selected']), int(truth[i]), int(correct[i])])
        summaries.append({'dataset': path.stem, 'source': str(path), 'baseline_accuracy': float(correct.mean()),
                          'metric_count': len(scores), 'selected_metric': best,
                          'exploratory_metric': observed, 'results': results})
        if args.sensitivity_seeds:
            counts = []
            for seed in range(args.sensitivity_seeds):
                fit = np.zeros(len(truth), dtype=bool)
                fit[np.random.default_rng(seed).permutation(len(truth))[:len(truth)//2]] = True
                score = corrected_scores(f, fit)[observed]
                stats, _ = evaluate(score, correct, f['coord'], f['position'], coverage=args.coverage)
                counts.append(stats['errors'])
            sensitivity[path.stem] = counts
        print(path.stem, 'selected:', best, json.dumps(results[best]))
    with (args.out / 'metrics.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(all_results[0]))
        writer.writeheader()
        writer.writerows(all_results)
    if sensitivity:
        (args.out / 'fit-sensitivity.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
