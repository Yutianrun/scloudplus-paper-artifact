"""Ten fresh 256 captures, sequential groups, live mixture SPRT then lattice correction."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT/'experiments/sprt-256-20261029'
FIXTURES = ROOT/'experiments/freeze80-256-e2e10-20261027-113105'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def check_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'input changed: {path}')


def capture_stats(capture):
    lines = (capture/'capture.log').read_text().splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith('#')))
    ids = {(int(r['coord']), int(r['position'])) for r in rows}
    if len(rows) != 13024 or ids != {(c, p) for c in range(11) for p in range(1184)}:
        raise ValueError('capture is missing or duplicates coefficients')
    ns = np.array([[int(r['n_a']), int(r['n_b'])] for r in rows])
    if np.any(ns < 0) or np.any(ns > 1000):
        raise ValueError('sample budget violated')
    log = (capture/'capture.err').read_text()
    workers = re.findall(r'SPRT_DONE worker=(\d+) coefficients=(\d+) calls=(\d+) seconds=([\d.]+)', log)
    if len(workers) != 6 or {int(w[0]) for w in workers} != set(range(6)):
        raise ValueError('worker accounting incomplete')
    if sum(int(w[1]) for w in workers) != 13024:
        raise ValueError('worker coefficient totals disagree')
    calls = sum(int(w[2]) for w in workers)
    calibration_attempts = len(re.findall(r'CAL worker=\d+ attempt=', log))
    costs = dict(warmup=2400, calibration=calibration_attempts*6000,
                 target_and_paired_reference=2*int(ns.sum()), local_reference=13024*64)
    if sum(costs.values()) != calls:
        raise ValueError('decapsulation counts do not reconcile')
    wall = re.search(r'SPRT_FINISHED coefficients=13024 undone=0 .* wall=([\d.]+)', log)
    if wall is None:
        raise ValueError('missing successful full capture completion')
    unresolved = np.array([r['tb_base']=='255' or r['tb_shift']=='255' for r in rows])
    ref_bad = np.array([r['ref_ok']=='0' for r in rows])
    return dict(physical_capture_seconds=float(wall[1]), timing_calls=calls,
                timing_costs=costs, mean_samples_per_axis=float(ns.mean()),
                sample_quantiles=np.quantile(ns, [0, .5, .9, .95, .99, 1]).tolist(),
                capped_axes=int((ns==1000).sum()), calibration_attempts=calibration_attempts,
                collector_abstentions=sum(r['guess']=='99' for r in rows),
                unresolved_coefficients=int(unresolved.sum()), reference_rejected_coefficients=int(ref_bad.sum()),
                both_unresolved_and_reference_rejected=int((unresolved & ref_bad).sum()),
                workers=[dict(worker=int(w), coefficients=int(n), timing_calls=int(c), seconds=float(s))
                         for w,n,c,s in workers])


def render(out, run):
    rows = []
    for group in run['groups']:
        time_value = f"{group['physical_capture_seconds']/60:.1f}" if 'physical_capture_seconds' in group else '—'
        recovery = str(group.get('public_accepted_columns', '—'))
        errors = str(group.get('initial_wrong_non_erased', '—'))
        rows.append(f"| {group['group']} | {group['seed']} | {group['status']} | {time_value} | {errors} | {recovery} |")
    complete = [g for g in run['groups'] if g['status']=='complete']
    successful = sum(g.get('kem_verified', False) for g in complete)
    text = f'''# 10 组全新 SPRT 采集与格纠错

状态：**{run['status']}**。完成 **{len(complete)}/10 组**，完整公钥与 KEM 验收成功 **{successful}/10 组**。
未完成组不是失败，也不计作成功。每组采集时均固定协议，不使用其他组的真值调整参数。
当前状态更新于 {run['updated_at_utc']}。

| 组 | fixture seed | 状态 | 物理采集分钟 | 最佳猜测错误数（事后审计） | 公钥通过列数 / 11 |
|---:|---:|---|---:|---:|---:|
{chr(10).join(rows)}

本批次对原有 10 个测试公钥分别做一次**新的完整物理采集**；旧开发试跑不计入。
使用固定 α=0.001 的双侧符号混合 SPRT，N 从 32 起，每批 16，上限 1000。
提前达到阈值即停止；到上限未决或参考拒绝仍给最佳三值猜测并保存证据。
置信度排序优先开放弱证据位置，随后使用稀疏嵌入格、BLASter 约简和全部公钥方程验收。
只有完整 11 列通过后才做随机 KEM、AES-GCM 和负例检查；真值只在公钥验收后审计。

每组六核、组间串行；校准、局部参考、重复测量都纳入实际调用数。
固定格预算为窗口 60/120/180/236 各 25 秒，360/640 各 45 秒，三列并行。
α 和候选分数不是桶正确率后验，也没有全密钥联合错误率保证。
本次沿用既有公钥和公开探针缓存并重新执行公开验证，验证耗时计入采集进程；不含重新搜索探针的成本。

进度 `results.json`；各组 `group-XX/capture/` 保存原始测量，
`group-XX/recovery/` 保存全部最佳猜测、置信度、格结果、公钥验收及事后审计。
完整成功组的 `recovery/verification.json` 是随机 KEM/AEAD 验收记录。
完整采集失败会保留日志并标记失败，不重跑到成功；后续组照常执行。
总 wall time 见 results.json；未完成期间不要把各组进度估计当成最终总耗时。
'''
    (out/'REPORT.md').write_text(text)


def preflight():
    check_hashes(json.loads((PREVIOUS/'build.json').read_text()))
    paths = [PREVIOUS/'collector', PREVIOUS/'collector.c', PREVIOUS/'sprt.h', Path(__file__),
             HERE/'postprocess.py', ROOT/'scripts/sprt_soft_decision.py', ROOT/'scripts/sequential_oracle.py',
             ROOT/'scripts/public_embedding_cpu.py', ROOT/'scripts/sparse_correction_lattice.py',
             ROOT/'scripts/postprocess_recover.py', ROOT/'scripts/evaluate_confidence.py',
             ROOT/'scripts/verify_public_recovery.py', HERE/'PROTOCOL.md']
    groups = []
    for g in range(1,11):
        fixture = FIXTURES/f'experiment-{g:02d}'
        seed = int(json.loads((fixture/'pipeline.json').read_text())['args'][0])
        groups.append(dict(group=g, seed=seed, fixture=str(fixture), status='pending'))
        paths += [fixture/name for name in ('public.pk','public.npz','probes.cache','public-recovery','pipeline.json')]
        if not (fixture/'256-run1.log').is_file():
            raise ValueError('missing post-acceptance audit labels')
    # Confirm the solver dependency is available before starting many hours of capture.
    sys.path.insert(0, str(ROOT.parent/'BLASter/src'))
    import blaster.blaster
    import cryptography.hazmat.primitives.ciphers.aead
    return groups, {str(path):digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'runs')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    groups, hashes = preflight()
    if args.preflight:
        print(json.dumps(dict(groups=groups, files_verified=len(hashes)), indent=2)); return
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    lock = (out/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out/'runner.pid').write_text(str(os.getpid())+'\n')
    started = time.monotonic()
    run = dict(status='running', started_at_utc=now(), pid=os.getpid(), groups=groups,
               fresh_captures=10, reuse_previous_captures=False, vote3_used=False,
               alpha_per_axis=.001, min_samples=32, batch_samples=16, max_samples=1000,
               workers_per_capture=6, group_concurrency=1, source_hashes=hashes)
    def save():
        run.update(updated_at_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_json(out/'results.json', run); render(out, run)
    save()
    snapshot = out/'source-snapshot'; snapshot.mkdir()
    for path in hashes:
        if Path(path).suffix in ('.py','.c','.h','.md'):
            shutil.copyfile(path, snapshot/Path(path).name)
    # Pin the acquisition executable used by all ten measurements.
    binary = out/'collector'
    shutil.copy2(PREVIOUS/'collector', binary)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SPRT_', 'TIMING_'))
           and k not in ('RETEST_CSV','PROBE_VALIDATE_TOL','PROBE_VALIDATE','PROBE_CACHE',
                         'PUBLIC_KEY_OUT','ORACLE','SEARCH_ONLY','PROFILE_ONLY')}
    env.update(TIMING_TSC='1', TIMING_ABBA='0', TIMING_INTERLEAVE='1', JOINT_DECODE='0',
               BATCH_WORKERS='6', SERIAL_CAL='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               ORACLE='0', SEARCH_ONLY='0', PROFILE_ONLY='0', SPRT_ALPHA='.001')
    try:
        for group in groups:
            fixture = Path(group['fixture'])
            dest = out/f"group-{group['group']:02d}"; dest.mkdir()
            capture = dest/'capture'; capture.mkdir()
            group_start = time.monotonic()
            try:
                check_hashes(hashes)
                if digest(binary)!=hashes[str(PREVIOUS/'collector')]:
                    raise ValueError('pinned collector changed')
                group.update(status='capturing', started_at_utc=now()); save()
                print(f"{now()} group {group['group']} capture started", flush=True)
                command = [str(binary),str(group['seed']),'1184','1000','3000','2000000','11','6','0']
                group_env = {**env,'SPRT_OUT':str(capture),'PROBE_CACHE':str(fixture/'probes.cache'),
                             'PUBLIC_KEY_OUT':str(capture/'public.pk')}
                capture_start = time.monotonic()
                with (capture/'capture.log').open('w') as stdout, (capture/'capture.err').open('w') as stderr:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=group_env)
                    group['capture_pid'] = proc.pid; group['capture_command'] = command; save()
                    while proc.poll() is None:
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            group['capture_elapsed_seconds'] = time.monotonic()-capture_start; save()
                    if proc.returncode:
                        raise RuntimeError(f'capture exited {proc.returncode}; partial run preserved')
                group['capture_process_seconds'] = time.monotonic()-capture_start
                if (capture/'public.pk').read_bytes() != (fixture/'public.pk').read_bytes():
                    raise ValueError('captured public key identity mismatch')
                shutil.copyfile(fixture/'public.npz',capture/'public.npz')
                group.update(capture_stats(capture), status='postprocessing')
                write_json(capture/'measurement-summary.json', group); save()
                check_hashes(hashes)
                command = [sys.executable,str(HERE/'postprocess.py'),'--capture',str(capture),
                           '--fixture',str(fixture),'--out',str(dest/'recovery')]
                post_start = time.monotonic()
                with (dest/'postprocess.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((dest/'recovery/results.json').read_text())
                if result['status']!='complete' or not result['input_hashes_verified']:
                    raise ValueError('postprocessing incomplete or changed inputs')
                for key in ('public_accepted_columns','forced_guesses','trusted_guesses','forced_guess_errors',
                            'trusted_guess_errors','initial_wrong_non_erased','corrected_wrong_non_erased',
                            'accepted_truth_mismatches','correction_seconds'):
                    group[key] = result[key]
                group['postprocess_process_seconds'] = time.monotonic()-post_start
                group['kem_verified'] = False
                if result['public_accepted_columns']==11:
                    v=json.loads((dest/'recovery/verification.json').read_text())
                    if v['public_equations_passed']!=13024 or not all(v[k] for k in
                          ('fresh_kem_shared_secret_match','authenticated_message_decrypted',
                           'tampered_aead_rejected','corrupted_secret_rejected')):
                        raise ValueError('full verification failed')
                    group['kem_verified'] = True
                check_hashes(hashes)
                group.update(status='complete',input_hashes_verified=True)
            except Exception as exc:
                group.update(status='failed', error=repr(exc))
                print(f"{now()} group {group['group']} failed: {exc!r}", flush=True)
            group.update(finished_at_utc=now(),total_seconds=time.monotonic()-group_start)
            write_json(dest/'summary.json',group); save()
            print(f"{now()} group {group['group']} {group['status']} columns={group.get('public_accepted_columns')}",flush=True)
        run['status']='complete' if all(g['status']=='complete' for g in groups) else 'complete_with_failures'
        run['complete_keys']=sum(g.get('kem_verified',False) for g in groups)
        run['accepted_columns']=sum(g.get('public_accepted_columns',0) for g in groups)
        run['finished_at_utc']=now(); save()
    except BaseException as exc:
        run.update(status='interrupted',error=repr(exc)); save(); raise


if __name__=='__main__':
    main()
