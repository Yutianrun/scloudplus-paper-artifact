"""Experimental fixed agreement/L2 ranking with a per-column 80% frozen mask.

Preserve the original vote3 guesses. No labels or previously recovered columns
enter recovery. Strict mode solves the complete 20% complement; --adaptive
tries smaller prefixes within that complement. Both modes keep the same 80%
mask permanently frozen and accept only full public-equation checks.

The metric was developed retrospectively on 256/384 data. Successful replay is
not evidence of an independent blind evaluation or guaranteed correct freezing.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from evaluate_confidence import read_log, recovery_prediction
from joint_confidence import joint_score, load_runs
from recover_vote3 import SCHEDULE, valid


def prepare(log, runs, public):
    with np.load(public, allow_pickle=False) as p:
        A, B = p['A'].astype(np.int64), p['B'].astype(np.int64)
    f, _ = read_log(log, include_truth=False)
    expected = {(j, i) for j in range(B.shape[1]) for i in range(A.shape[1])}
    if set(zip(f['coord'], f['position'])) != expected:
        raise ValueError('full coordinate set required')
    guess = recovery_prediction(f, 'logged')
    raw = load_runs(runs, f)
    agree = np.stack([r['guess'] == guess for r in raw]).sum(axis=0)
    score = agree + np.arctan(joint_score(f, raw)) / np.pi
    guesses, orders, frozen = [], [], []
    n = A.shape[1]
    # Exact integer ceil(0.8*n), including n not divisible by five.
    w = n - (4*n + 4)//5
    for j in range(B.shape[1]):
        ids = np.flatnonzero(f['coord'] == j)
        pos = f['position'][ids].astype(int)
        g = np.zeros(n, dtype=np.int64)
        g[pos] = guess[ids]
        order = pos[np.lexsort((pos, score[ids]))]
        mask = np.ones(n, dtype=bool)
        mask[order[:w]] = False
        guesses.append(g); orders.append(order); frozen.append(mask)
    return A, B, f, score, np.array(guesses), np.array(orders), np.array(frozen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--runs', type=Path, nargs=3)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--adaptive', action='store_true')
    p.add_argument('--cores', type=int, default=6)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--column', type=int, help=argparse.SUPPRESS)
    p.add_argument('--window', type=int, help=argparse.SUPPRESS)
    p.add_argument('--extra', type=int, default=300, help=argparse.SUPPRESS)
    p.add_argument('--beta', type=int, default=56, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.cores < 1 or a.timeout <= 0:
        p.error('cores and timeout must be positive')
    state_path = a.out/'input-state.npz'
    if a.column is not None:
        from postprocess_recover import build, try_extract, blaster_reduce
        with np.load(a.public, allow_pickle=False) as z:
            A, B = z['A'].astype(np.int64), z['B'].astype(np.int64)
        with np.load(state_path, allow_pickle=False) as z:
            guess, order, frozen = (z[k][a.column] for k in ('guesses','orders','frozen'))
        support = order[:a.window]
        if np.any(frozen[support]):
            raise ValueError('attempt would change frozen entries')
        basis, piv, inv, r, m, k = build(A, B[:,a.column], guess, support,
                                        extra_rows=a.extra)
        print(f'W={k} dim={m+1} beta={a.beta} extra={a.extra}', flush=True)
        _, reduced, _ = blaster_reduce(basis.T.copy(), lll_size=64, beta=a.beta,
                                      bkz_prog=4, bkz_tours=2, cores=a.cores)
        s = try_extract(reduced, m, k, piv, inv, r, guess, support, A, B[:,a.column])
        if s is None:
            return 1
        if not valid(A, B[:,a.column], s) or not np.array_equal(s[frozen], guess[frozen]):
            raise ValueError('public equations or frozen-entry check failed')
        np.save(a.out/f'column-{a.column}.npy', s)
        return 0
    if a.log is None or a.runs is None:
        p.error('a merged log and three raw runs are required')
    A, B, f, score, guesses, orders, frozen = prepare(a.log, a.runs, a.public)
    a.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(state_path, guesses=guesses, orders=orders, frozen=frozen)
    np.savez_compressed(a.out/'observable-scores.npz', coord=f['coord'],
                        position=f['position'], guess=f['guess'], score=score)
    w = int((~frozen[0]).sum())
    schedule = [(w,300,56)]
    if a.adaptive:
        schedule = [(k,extra,beta) for k,extra,beta in SCHEDULE if k < w] + schedule
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (a.log,*a.runs,a.public,Path(__file__))}
    report = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                  metric='coord/agreement_then_l2', coverage=.8, scope='per-column',
                  prediction='unchanged original vote3', truth_used=False,
                  study_scope='retrospective replay; metric developed on 256/384',
                  frozen_per_column=int(frozen[0].sum()), unknown_per_column=w,
                  mode='adaptive' if a.adaptive else 'strict', schedule=schedule,
                  cores=a.cores, timeout_seconds=a.timeout, inputs_sha256=hashes,
                  status='running', columns=[])
    with (a.out/'freeze-mask.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['coord','position','original_vote3_guess','score','frozen80'])
        for i in np.lexsort((f['position'],f['coord'])):
            c,pos = int(f['coord'][i]),int(f['position'][i])
            writer.writerow([c,pos,int(f['guess'][i]),float(score[i]),int(frozen[c,pos])])
    def save():
        (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for j,guess in enumerate(guesses):
        record=dict(column=j, public_valid=bool(valid(A,B[:,j],guess)), attempts=[])
        if record['public_valid']:
            np.save(a.out/f'column-{j}.npy',guess)
            record['window']=0
        else:
            for window,extra,beta in schedule:
                print(f'column {j}: start W={window} extra={extra} beta={beta}',flush=True)
                start=time.monotonic()
                command=[sys.executable,str(Path(__file__).resolve()),'--public',str(a.public),
                         '--out',str(a.out),'--column',str(j),'--window',str(window),
                         '--extra',str(extra),'--beta',str(beta),'--cores',str(a.cores)]
                with (a.out/f'column-{j}-W{window}.log').open('w') as handle:
                    try:
                        proc=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
                            timeout=a.timeout,env={**os.environ,'OPENBLAS_NUM_THREADS':'1',
                                                   'OMP_NUM_THREADS':str(a.cores)})
                        status=proc.returncode
                    except subprocess.TimeoutExpired:
                        status='timeout'
                attempt=dict(window=window,extra=extra,beta=beta,status=status,
                             seconds=round(time.monotonic()-start,2))
                record['attempts'].append(attempt)
                if status==0:
                    s=np.load(a.out/f'column-{j}.npy',allow_pickle=False)
                    if not valid(A,B[:,j],s) or not np.array_equal(s[frozen[j]],guess[frozen[j]]):
                        raise ValueError(f'column {j}: invalid saved solution')
                    record.update(public_valid=True,window=window)
                print(json.dumps({'column':j,**attempt}),flush=True)
                report['current_column']=record
                save()
                if record['public_valid']:
                    break
        report['columns'].append(record)
        report.pop('current_column',None)
        save()
        print(json.dumps(record),flush=True)
    passed=all(c['public_valid'] for c in report['columns'])
    report.update(status='public_equations_passed' if passed else 'incomplete',
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print('ALL COLUMNS SOLVED' if passed else 'INCOMPLETE',flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
