"""Audit an agreement80 recovery against observable inputs and public equations."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from evaluate_confidence import read_log, recovery_prediction
from joint_confidence import joint_score, load_runs
from recover_vote3 import SCHEDULE, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='audit inputs and available columns; do not claim full recovery')
    args = parser.parse_args()
    out = args.directory
    report = json.loads((out / 'results.json').read_text())
    assert report['status'] == 'public_equations_passed' or (
        args.allow_incomplete and report['status'] == 'incomplete')
    assert report['metric'] == 'coord/agreement_then_l2'
    assert report['coverage'] == .8 and report['scope'] == 'per-column'
    assert report['mode'] in ('adaptive', 'adaptive_dual_order')
    dual = report['mode'] == 'adaptive_dual_order'
    if dual:
        assert report['support_orders'] == ['l2', 'agreement']
    assert report['cores'] == 2 and report['timeout_seconds'] == 180
    inputs = report['inputs_sha256']
    for name, expected in inputs.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    logs = [Path(name) for name in inputs if name.endswith('.log')]
    publics = [Path(name) for name in inputs if name.endswith('.npz')]
    assert len(logs) == 4 and len(publics) == 1
    with np.load(publics[0], allow_pickle=False) as z:
        A, B = z['A'], z['B']
    solved = [c['column'] for c in report['columns'] if c['public_valid']]
    S = np.zeros((B.shape[1], A.shape[1]), dtype=np.int64)
    for j in solved:
        s = np.load(out / f'column-{j}.npy', allow_pickle=False)
        assert s.shape == (A.shape[1],) and np.isin(s, [-1, 0, 1]).all()
        assert valid(A.astype(np.int64), B[:, j].astype(np.int64), s)
        S[j] = s
    f, _ = read_log(logs[0], include_truth=False)
    original = recovery_prediction(f, 'logged')
    runs = load_runs(logs[1:], f)
    votes = np.stack([r['guess'] == original for r in runs]).sum(axis=0)
    l2_score = joint_score(f, runs)
    score = votes + np.arctan(l2_score) / np.pi
    with np.load(out / 'observable-scores.npz', allow_pickle=False) as z:
        for field in ('coord', 'position', 'guess'):
            assert np.array_equal(z[field], f[field]), field
        assert np.array_equal(z['score'], score)
    with np.load(out / 'input-state.npz', allow_pickle=False) as z:
        guesses, orders, frozen = (z[k] for k in ('guesses', 'orders', 'frozen'))
        l2_orders = z['l2_orders'] if dual else None
    assert guesses.shape == orders.shape == frozen.shape == S.shape
    count = (4 * A.shape[1] + 4) // 5
    unknown = A.shape[1] - count
    schedule = [list(s) for s in SCHEDULE if s[0] < unknown] + [[unknown, 300, 56]]
    assert report['schedule'] == schedule
    attempt_schedule = ([s + [order] for s in schedule for order in ('l2', 'agreement')]
                        if dual else schedule)
    assert report['frozen_per_column'] == count
    assert report['unknown_per_column'] == unknown
    assert len(report['columns']) == B.shape[1]
    for j, record in enumerate(report['columns']):
        ids = np.flatnonzero(f['coord'] == j)
        pos = f['position'][ids].astype(int)
        assert np.array_equal(np.sort(pos), np.arange(A.shape[1]))
        assert np.array_equal(guesses[j, pos], original[ids])
        order = pos[np.lexsort((pos, score[ids]))]
        assert np.array_equal(orders[j], order)
        mask = np.ones(A.shape[1], dtype=bool)
        mask[order[:unknown]] = False
        assert np.array_equal(frozen[j], mask)
        if dual:
            full = pos[np.lexsort((pos, l2_score[ids]))]
            expected_l2 = np.concatenate((full[~mask[full]], full[mask[full]]))
            assert np.array_equal(l2_orders[j], expected_l2)
        assert record['column'] == j
        attempts = record['attempts']
        actual = [[a['window'], a['extra'], a['beta']] + ([a['order']] if dual else [])
                  for a in attempts]
        assert actual == attempt_schedule[:len(attempts)]
        if not record['public_valid']:
            assert len(attempts) == len(attempt_schedule)
            assert all(a['status'] != 0 for a in attempts)
            assert not (out / f'column-{j}.npy').exists()
            continue
        assert record['window'] == (attempts[-1]['window'] if attempts else 0)
        assert not attempts or attempts[-1]['status'] == 0
        assert all(a['status'] != 0 for a in attempts[:-1])
        if dual and attempts:
            assert record['order'] == attempts[-1]['order']
            order = l2_orders[j] if record['order'] == 'l2' else orders[j]
        assert set(np.flatnonzero(S[j] != guesses[j])) <= set(order[:record['window']])
    assert np.array_equal(S[solved][frozen[solved]], guesses[solved][frozen[solved]])
    with (out / 'freeze-mask.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == S.size
    assert {(int(r['coord']), int(r['position'])) for r in rows} == set(zip(f['coord'], f['position']))
    score_map = {(int(c), int(p)): float(s) for c, p, s in zip(f['coord'], f['position'], score)}
    for row in rows:
        j, p = int(row['coord']), int(row['position'])
        assert int(row['original_vote3_guess']) == guesses[j, p]
        assert int(row['frozen80']) == frozen[j, p]
        assert float(row['score']) == score_map[j, p]
    result = dict(
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='observable-input, public-equation and frozen-mask audit; no labels read',
        inputs_sha256_match=True, original_guesses_preserved=True,
        scores_recomputed_exactly=True, masks_recomputed_exactly=True,
        common_schedule_verified=True, changes_within_successful_support=True,
        mode=report['mode'], support_orders_recomputed_exactly=True,
        columns=B.shape[1], frozen_per_column=count, frozen_total=int(frozen.sum()),
        recovery_complete=len(solved) == B.shape[1], verified_columns=solved,
        frozen_entries_checked=count * len(solved),
        max_unknown_per_column=unknown, frozen_entries_changed=0,
        public_equations_passed=int(A.shape[0] * len(solved)),
        corrections_vs_original_vote3=int(np.count_nonzero(S[solved] != guesses[solved])),
        changes_by_column=[int(np.count_nonzero(S[j] != guesses[j])) if j in solved else None
                           for j in range(B.shape[1])],
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [out / n for n in ('input-state.npz', 'observable-scores.npz',
                                   'freeze-mask.csv', 'results.json')] + [Path(__file__)]})
    (out / 'frozen-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
