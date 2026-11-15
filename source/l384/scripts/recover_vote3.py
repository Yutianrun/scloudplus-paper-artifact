"""Joint timing confidence + bounded, all-column lattice recovery; no labels.

Small windows use deep-LLL only (BLASter beta<40); larger windows use BKZ.
Failure/timeout advances the public-only schedule, never a truth-derived window.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from evaluate_confidence import read_log, recovery_prediction
from joint_confidence import joint_score, load_runs

SCHEDULE = [(20,200,32), (40,200,32), (60,200,32),
            (90,400,40), (119,400,44), (160,500,50), (220,500,56), (300,600,56)]


def prepare(log, runs, public):
    f, _ = read_log(log, include_truth=False)
    with np.load(public, allow_pickle=False) as p:
        A, B = p['A'].astype(np.int64), p['B'].astype(np.int64)
    if A.ndim != 2 or B.ndim != 2 or len(A) != len(B):
        raise ValueError('invalid public matrix shapes')
    expected = {(c, i) for c in range(B.shape[1]) for i in range(A.shape[1])}
    if set(zip(f['coord'], f['position'])) != expected:
        raise ValueError('full matrix capture required: missing or unexpected coordinates')
    score = joint_score(f, load_runs(runs, f))
    pred = recovery_prediction(f, 'logged')
    guesses, orders = [], []
    for j in range(B.shape[1]):
        ids = np.flatnonzero(f['coord'] == j)
        pos = f['position'][ids].astype(int)
        guess = np.zeros(A.shape[1], dtype=np.int64)
        guess[pos] = pred[ids]
        guesses.append(guess)
        orders.append(pos[np.lexsort((pos, score[ids]))])
    return A, B, np.array(guesses), np.array(orders)


def valid(A, b, s):
    return (s.shape == (A.shape[1],) and np.isin(s, [-1,0,1]).all()
            and np.max(np.abs((b-A@s+512)%1024-512)) <= 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--runs', type=Path, nargs=3)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cores', type=int, default=4)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--attempt', type=int, help=argparse.SUPPRESS)
    p.add_argument('--column', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.cores < 1 or a.timeout <= 0:
        p.error('cores and timeout must be positive')
    state = a.out / 'input-state.npz'
    if a.attempt is not None:
        from postprocess_recover import build, try_extract, blaster_reduce
        with np.load(a.public, allow_pickle=False) as public:
            A, B = public['A'].astype(np.int64), public['B'].astype(np.int64)
        with np.load(state, allow_pickle=False) as inputs:
            guess, order = inputs['guesses'][a.column], inputs['orders'][a.column]
        w, extra, beta = SCHEDULE[a.attempt]
        support = order[:w]
        basis, piv, inv, r, m, k = build(A, B[:, a.column], guess, support, extra_rows=extra)
        _, reduced, _ = blaster_reduce(basis.T.copy(), lll_size=64, beta=beta,
                                       bkz_prog=4, bkz_tours=2, cores=a.cores)
        s = try_extract(reduced,m,k,piv,inv,r,guess,support,A,B[:,a.column])
        if s is None:
            return 1
        np.save(a.out / f'column-{a.column}.npy', s)
        return 0
    if not a.log or not a.runs:
        p.error('--log and three --runs are required')
    A, B, guesses, orders = prepare(a.log, a.runs, a.public)
    # Never accept columns left over from a previous input or attempt.
    a.out.mkdir(parents=True, exist_ok=False)
    np.savez(state, guesses=guesses, orders=orders)
    records = []
    for j, guess in enumerate(guesses):
        record = dict(column=j, public_valid=False, attempts=[])
        if valid(A, B[:,j], guess):
            np.save(a.out / f'column-{j}.npy', guess)
            record.update(public_valid=True, window=0)
        else:
            for index, (window, extra, beta) in enumerate(SCHEDULE):
                start = time.monotonic()
                command = [sys.executable, str(Path(__file__).resolve()), '--public', str(a.public),
                           '--out', str(a.out), '--column', str(j), '--attempt', str(index),
                           '--cores', str(a.cores)]
                with (a.out / f'column-{j}-W{window}.log').open('w') as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=a.timeout,
                                                env={**os.environ, 'OPENBLAS_NUM_THREADS':'1',
                                                     'OMP_NUM_THREADS':str(a.cores)})
                        status = result.returncode
                    except subprocess.TimeoutExpired:
                        status = 'timeout'
                seconds = round(time.monotonic()-start,2)
                if status == 0:
                    s = np.load(a.out / f'column-{j}.npy', allow_pickle=False)
                    if not valid(A,B[:,j],s):
                        raise ValueError(f'column {j}: subprocess returned invalid result')
                    record.update(public_valid=True, window=window)
                record['attempts'].append(dict(window=window, extra=extra, beta=beta,
                                               status=status, seconds=seconds))
                print(f'column {j}: W={window} status={status} {seconds:.2f}s', flush=True)
                if record['public_valid']:
                    break
        records.append(record)
        (a.out / 'results.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    passed = all(r['public_valid'] for r in records)
    print('ALL COLUMNS SOLVED' if passed else 'INCOMPLETE: see results.json',flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
