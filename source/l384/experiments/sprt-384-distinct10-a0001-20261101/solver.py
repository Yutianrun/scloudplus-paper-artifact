"""Blind weighted sparse embedding with observable BLASter checkpoints.

Only public A/B and the original guesses/order enter recovery. Integer weights
penalize dense corrections; embedding reduction replaces one-shot Babai. Check
every reduction checkpoint against ALL public equations, including unused rows.
Uses BLASter's internal callback API so a solution is saved before a long tour
finishes. The experiment runner pins that implementation's source hash.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import sys
FIXED = Path(__file__).resolve().parent/'blaster-fixed/src'
sys.path.insert(0,str(FIXED))
import blaster
from blaster import _core
if not Path(_core.__file__).resolve().is_relative_to(FIXED.resolve()):
    raise RuntimeError('wrong BLASter loaded')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from postprocess_recover import center
from sparse_correction_lattice import build_sparse, extract_sparse


def weighted_basis(A, b, guess, support, equations, weight=2, embedding=1):
    if not isinstance(weight, int) or weight < 1 or not isinstance(embedding, int) or embedding < 1:
        raise ValueError('positive integer scales required')
    basis, support = build_sparse(A, b, guess, support, equations)
    basis[:, :len(support)] *= weight
    basis[-1, -1] = embedding
    return basis.T.copy(), support


def extract_weighted(columns, A, b, guess, support, weight=2, embedding=1):
    k = len(support)
    eligible = ((np.abs(columns[-1]) == embedding)
                & np.all(columns[:k] % weight == 0, axis=0))
    unscaled = columns[:, eligible].copy()
    unscaled[:k] //= weight
    unscaled[-1] //= embedding
    return extract_sparse(unscaled, A, b, guess, support)


class StopReduction(Exception):
    pass


def solve(A, b, guess, support, equations, *, weight=2, embedding=1,
          beta=40, tours=2, cores=2, seconds=120, checkpoint=None):
    from blaster.blaster import (TimeProfile, lll_reduce, bkz_reduce,
                                set_num_cores, set_debug_flag)
    start = time.monotonic()
    basis, support = weighted_basis(A, b, guess, support, equations, weight, embedding)
    dimension = len(basis)
    transform = np.eye(dimension, dtype=np.int64)
    scratch = transform.copy()
    profile = TimeProfile()
    set_num_cores(cores)
    set_debug_flag(0)
    result = None
    record = dict(status='reducing', public_valid=False, truth_used=False,
                  previous_solutions_used=False, dimension=dimension,
                  window=len(support), equations=min(equations, len(A)),
                  weight=weight, embedding=embedding, beta=beta, tours=tours,
                  iterations=0, bkz_checkpoints=0, seconds=0.)

    def trace(iteration, prof, note):
        nonlocal result
        record.update(iterations=iteration, phase=note[0], seconds=time.monotonic()-start,
                      shortest_squared_norm=int(np.min(np.sum(basis.astype(float)**2, axis=0))))
        record['bkz_checkpoints'] += int(note[0].startswith('BKZ'))
        result = extract_weighted(basis, A, b, guess, support, weight, embedding)
        if result is not None:
            record.update(status='public_valid', public_valid=True,
                          corrections=int(np.count_nonzero(result != guess)))
        elif record['seconds'] >= seconds:
            record['status'] = 'budget_exhausted'
        if checkpoint:
            checkpoint(record)
        if record['status'] != 'reducing':
            raise StopReduction

    try:
        lll_reduce(basis, transform, scratch, 64, .99, 4, profile,
                   {'checkpoint': trace}, False, False)
        if beta >= 40:
            bkz_reduce(basis, transform, scratch, 64, .99, 4, beta, tours, 64,
                       profile, {'checkpoint': trace}, False, False)
        record['status'] = 'exhausted'
    except StopReduction:
        pass
    record['seconds'] = time.monotonic()-start
    record['profile'] = str(profile)
    return result, record, basis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('public', 'state', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--column', type=int, required=True)
    for name, default in [('window', 236), ('equations', 128), ('weight', 2),
                          ('embedding', 1), ('beta', 40), ('tours', 2), ('cores', 2)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--seconds', type=float, default=120)
    args = p.parse_args()
    with np.load(args.public, allow_pickle=False) as pub:
        A, B = pub['A'].astype(np.int64), pub['B'].astype(np.int64)
    with np.load(args.state, allow_pickle=False) as state:
        guess, order = state['guesses'][args.column], state['orders'][args.column]
        erasures = state['erasures'][args.column]
    if (not 1 <= args.window <= len(guess) or args.seconds <= 0
            or args.equations < 1 or args.cores < 1 or args.tours < 1
            or args.weight < 1 or args.embedding < 1
            or args.beta not in (0, 40, 44, 48, 52, 56, 60)
            or not np.array_equal(np.sort(order), np.arange(len(guess)))
            or not set(np.flatnonzero(erasures)) <= set(order[:args.window])):
        p.error('invalid parameters/order or erasures outside support')
    args.out.mkdir(parents=True, exist_ok=False)
    paths = (args.public, args.state, Path(__file__))
    metadata = dict(column=args.column, inputs_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                                     for f in paths})
    def save(record):
        temporary = args.out/'results.json.tmp'
        temporary.write_text(json.dumps({**metadata, **record}, indent=2)+'\n')
        temporary.replace(args.out/'results.json')
    with (args.out/'reduction.jsonl').open('w') as log:
        def checkpoint(record):
            save(record)
            log.write(json.dumps(record)+'\n')
            log.flush()
        s, record, basis = solve(A, B[:, args.column], guess, order[:args.window], args.equations,
                                weight=args.weight, embedding=args.embedding, beta=args.beta,
                                tours=args.tours, cores=args.cores, seconds=args.seconds,
                                checkpoint=checkpoint)
    if s is not None:
        if np.any(np.abs(center(B[:, args.column]-A@s)) > 1) or not np.isin(s, [-1, 0, 1]).all():
            raise ArithmeticError('public validation failed')
        np.save(args.out/f'column-{args.column}.npy', s)
    np.savez_compressed(args.out/'reduced-state.npz', reduced=basis,
                        support=np.sort(order[:args.window]))
    for path, digest in metadata['inputs_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('input/source changed during solve')
    record['input_hashes_verified'] = True
    save(record)
    print(json.dumps(record), flush=True)
    return 0 if s is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
