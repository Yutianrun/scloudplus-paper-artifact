"""One fresh 384 then one fresh 512 capture, alpha .01, capped soft decisions."""
import argparse
import csv
from datetime import datetime,timezone
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def check(hashes):
    for path,h in hashes.items():
        if digest(path)!=h:raise ValueError(f'input changed: {path}')


def capture_stats(capture,n,nbar):
    with (capture/'capture.log').open() as f:
        rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
    total=n*nbar
    if len(rows)!=total or {(int(r['coord']),int(r['position'])) for r in rows}!={(c,p) for c in range(nbar) for p in range(n)}:
        raise ValueError('incomplete or duplicate capture')
    ns=np.array([[int(r['n_a']),int(r['n_b'])] for r in rows])
    if np.any(ns<0) or np.any(ns>1000):raise ValueError('N cap violated')
    log=(capture/'capture.err').read_text()
    workers=re.findall(r'SPRT_DONE worker=(\d+) coefficients=(\d+) calls=(\d+) seconds=([\d.]+)',log)
    if len(workers)!=6 or {int(w[0]) for w in workers}!=set(range(6)) or sum(int(w[1]) for w in workers)!=total:
        raise ValueError('worker completion mismatch')
    calls=sum(int(w[2]) for w in workers)
    cal=len(re.findall(r'CAL worker=\d+ attempt=',log))
    costs=dict(warmup=2400,calibration=cal*6000,target_and_paired_reference=2*int(ns.sum()),local_reference=total*64)
    if sum(costs.values())!=calls:raise ValueError('call accounting mismatch')
    finish=re.search(rf'SPRT_FINISHED coefficients={total} undone=0 cap=1000 alpha=([\d.e+-]+) workers=6 wall=([\d.]+)',log)
    if finish is None or float(finish[1])!=.01:raise ValueError('completion or alpha mismatch')
    return dict(physical_capture_seconds=float(finish[2]),timing_calls=calls,timing_costs=costs,
                coefficients=total,mean_samples_per_axis=float(ns.mean()),capped_axes=int((ns==1000).sum()),
                collector_abstentions=sum(r['guess']=='99' for r in rows),
                sample_quantiles=np.quantile(ns,[0,.5,.9,.95,.99,1]).tolist(),
                reference_rejected_coefficients=sum(r['ref_ok']=='0' for r in rows),
                workers=[dict(worker=int(w),coefficients=int(c),timing_calls=int(k),seconds=float(s)) for w,c,k,s in workers])


def render(out,run):
    rows=[]
    for g in run['groups']:
        mins=f"{g['physical_capture_seconds']/60:.1f}" if 'physical_capture_seconds' in g else '—'
        total=f"{g['total_seconds']/60:.1f}" if 'total_seconds' in g else '—'
        rows.append(f"| {g['level']} | {g['status']} | {mins} | {total} | {g.get('public_accepted_columns','—')}/16 | {g.get('initial_wrong_non_erased','—')} | {g.get('kem_verified','—')} |")
    (out/'REPORT.md').write_text(f'''# 384 / 512 SPRT α=0.01 实测

状态：{run['status']}；更新：{run['updated_at_utc']}。

| 参数级别 | 状态 | 实际采集分钟 | 采集与后处理总分钟 | 公钥通过列数 | 初始错误（事后审计） | KEM/AEAD |
|---:|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

每个级别一次全新采集，无 vote3；六核单组，384 完成后运行 512。
α=.01，N≥32，每16次检查，上限1000；上限未决仍给最佳猜测并保留置信度。
按低置信度优先使用既有稀疏嵌入格预算，独立全部公钥方程验收后才读取真值。
完整16列接受后执行随机 KEM、AES-GCM 和负例检查；不宣称恢复拒绝秘密 z。
沿用既有公开探针缓存，启动时核对公钥/参数/桶表绑定；不是43,046,721种向量的穷举不变性证明。
探针重搜索不计入本次时间；预热、校准、局部参考和重复测量计入调用数及实际采集时间。
构建与公开预检耗时单独保存在 build/manifest.json。
校准失败或未恢复列均按实际记录；不重采到成功，不用真值调参。
尚未完成不是成功或失败。详见各级别 capture/、recovery/ 和 summary.json。
''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((HERE/'build/manifest.json').read_text())
    hashes=manifest['hashes']
    for path in [Path(__file__),HERE/'postprocess.py',HERE/'PROTOCOL.md',
                 *[ROOT/'scripts'/f for f in ['sprt_soft_decision.py','sequential_oracle.py','public_embedding_cpu.py',
                    'sparse_correction_lattice.py','postprocess_recover.py','evaluate_confidence.py','verify_public_recovery.py']]]:
        hashes[str(path)]=digest(path)
    check(hashes)
    sys.path.insert(0,str(ROOT.parent/'BLASter/src'))
    import blaster.blaster
    import cryptography.hazmat.primitives.ciphers.aead
    for g in manifest['groups']:
        if not Path(g['audit_log']).is_file():raise ValueError('missing later audit labels')
        with np.load(Path(g['fixture'])/'public.npz') as p:
            if p['A'].shape!=(g['n'],g['n']) or p['B'].shape!=(g['n'],g['nbar']):raise ValueError('public shape')
    if args.preflight:
        print(json.dumps(dict(levels=[g['level'] for g in manifest['groups']],alpha=.01,files_verified=len(hashes))));return
    out=HERE/'runs';out.mkdir(exist_ok=False)
    lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'runner.pid').write_text(str(os.getpid())+'\n')
    start=time.monotonic()
    run=dict(status='running',started_at_utc=now(),pid=os.getpid(),alpha_per_axis=.01,max_samples=1000,
             min_samples=32,batch_samples=16,vote3_used=False,source_hashes=hashes,
             groups=[{**g,'status':'pending'} for g in manifest['groups']])
    def save():
        run.update(updated_at_utc=now(),elapsed_seconds=time.monotonic()-start)
        write(out/'results.json',run);render(out,run)
    save()
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for f in ['run.py','build.py','postprocess.py','collector.c','sprt.h','PROTOCOL.md']:shutil.copyfile(HERE/f,snapshot/f)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPRT_','TIMING_')) and k not in
         ('ORACLE','SEARCH_ONLY','PROFILE_ONLY','PROBE_VALIDATE','PROBE_VALIDATE_TOL','PROBE_CACHE','PUBLIC_KEY_OUT','RETEST_CSV')}
    env.update(SPRT_ALPHA='.01',TIMING_TSC='1',TIMING_ABBA='0',TIMING_INTERLEAVE='1',JOINT_DECODE='0',
               BATCH_WORKERS='6',SERIAL_CAL='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',ORACLE='0',SEARCH_ONLY='0',PROFILE_ONLY='0')
    try:
        for g in run['groups']:
            dest=out/str(g['level']);dest.mkdir();capture=dest/'capture';capture.mkdir()
            fixture=Path(g['fixture']);gstart=time.monotonic()
            try:
                check(hashes)
                g.update(status='capturing',started_at_utc=now());save()
                command=[g['binary'],str(g['seed']),str(g['n']),'1000','3000','2000000',str(g['nbar']),'6','0']
                genv={**env,'SPRT_OUT':str(capture),'PROBE_CACHE':str(fixture/'probes.cache'),'PUBLIC_KEY_OUT':str(capture/'public.pk')}
                print(now(),g['level'],'capture started',flush=True)
                cstart=time.monotonic()
                with (capture/'capture.log').open('w') as stdout,(capture/'capture.err').open('w') as stderr:
                    proc=subprocess.Popen(command,cwd=ROOT,env=genv,stdout=stdout,stderr=stderr)
                    g.update(capture_pid=proc.pid,capture_command=command);save()
                    while proc.poll() is None:
                        try:proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            g['capture_elapsed_seconds']=time.monotonic()-cstart;save()
                    if proc.returncode:raise RuntimeError(f'capture failed: {proc.returncode}')
                g['capture_process_seconds']=time.monotonic()-cstart
                if (capture/'public.pk').read_bytes()!=(fixture/'public.pk').read_bytes():raise ValueError('key identity mismatch')
                shutil.copyfile(fixture/'public.npz',capture/'public.npz')
                g.update(capture_stats(capture,g['n'],g['nbar']),status='postprocessing');save()
                write(capture/'measurement-summary.json',g)
                check(hashes)
                cmd=[sys.executable,str(HERE/'postprocess.py'),'--capture',str(capture),'--fixture',str(fixture),
                     '--level',str(g['level']),'--audit-log',g['audit_log'],'--out',str(dest/'recovery')]
                pstart=time.monotonic()
                with (dest/'postprocess.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                g['postprocess_process_seconds']=time.monotonic()-pstart
                result=json.loads((dest/'recovery/results.json').read_text())
                if result['status']!='complete' or not result['input_hashes_verified']:raise ValueError('recovery incomplete or modified')
                for key in ['public_accepted_columns','initial_wrong_non_erased','corrected_wrong_non_erased',
                            'accepted_truth_mismatches','correction_seconds','forced_guesses','trusted_guesses','forced_guess_errors','trusted_guess_errors']:
                    g[key]=result[key]
                g['kem_verified']=False
                if g['public_accepted_columns']==g['nbar']:
                    v=json.loads((dest/'recovery/verification.json').read_text())
                    if v['level']!=g['level'] or v['public_equations_passed']!=g['n']*g['nbar'] or not all(v[k] for k in
                            ['fresh_kem_shared_secret_match','authenticated_message_decrypted','tampered_aead_rejected','corrupted_secret_rejected']):raise ValueError('KEM verification failed')
                    g['kem_verified']=True
                check(hashes);g.update(status='complete',input_hashes_verified=True)
            except Exception as exc:
                g.update(status='failed',error=repr(exc));print(now(),g['level'],'failed',repr(exc),flush=True)
            g.update(total_seconds=time.monotonic()-gstart,finished_at_utc=now())
            write(dest/'summary.json',g);save();print(now(),g['level'],g['status'],g.get('public_accepted_columns'),flush=True)
        run.update(status='complete' if all(g['status']=='complete' for g in run['groups']) else 'complete_with_failures',finished_at_utc=now());save()
    except BaseException as exc:
        run.update(status='interrupted',error=repr(exc));save();raise


if __name__=='__main__':main()
