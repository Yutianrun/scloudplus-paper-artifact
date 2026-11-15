"""Ten fixed distinct keys, fresh per-core templates, capture then blind recovery."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'sprt-384-distinct10-a0001-20261101'
CORES=[1,2,4,5,7,6]
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    def check():
        for p,h in manifest['hashes'].items():
            if digest(p)!=h:raise ValueError(f'pinned input changed: {p}')
    check();out=HERE/'runs';out.mkdir(exist_ok=False)
    start=time.monotonic()
    report=dict(status='running',pid=os.getpid(),started_at_utc=now(),level=384,alpha=.001,
                max_samples=1000,template_samples=[256,512,1024],vote3_used=False,
                distinct_keys=10,cores=CORES,groups=manifest['groups'])
    def save():
        report.update(updated_at_utc=now(),elapsed_seconds=time.monotonic()-start)
        tmp=out/'results.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'results.json')
        lines=['# 384 新模板：10把不同密钥','',f"状态：{report['status']}；更新 UTC：{report['updated_at_utc']}",'',
               '|组|seed|状态|已采系数|模板+采集分钟|公钥通过列|KEM|','|---:|---:|---|---:|---:|---:|---|']
        for g in report['groups']:
            minutes=f"{g['capture_seconds']/60:.1f}" if 'capture_seconds' in g else '—'
            lines.append(f"|{g['group']}|{g['seed']}|{g['status']}|{g.get('captured_coefficients',0)}/26624|{minutes}|{g.get('public_accepted_columns','—')}/16|{g.get('kem_verified','—')}|")
        lines+=['','α=.001；N≤1000；模板256→512→1024；每列/每128位置重校准；六核各自模板。',
                '先原始样本重放、格纠错和完整公钥验收，再审计真值。oracle结果不参与选择。','']
        (out/'REPORT.md').write_text('\n'.join(lines))
    save()
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPRT_','TIMING_','TEMPLATE_')) and k not in
         ['ORACLE','SEARCH_ONLY','PROFILE_ONLY','PROBE_VALIDATE','PROBE_VALIDATE_TOL','PROBE_CACHE','PUBLIC_KEY_OUT','RETEST_CSV']}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    active=[];stopping=threading.Event()
    def stop(signum,frame):
        stopping.set()
        for proc in list(active):
            if proc.poll() is None:proc.terminate()
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def command(cmd,stdout,stderr,genv,heartbeat=True):
        with stdout.open('w') as log,stderr.open('w') as err:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=genv,stdout=log,stderr=err)
            active.append(proc)
            try:
                while True:
                    try:code=proc.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        if heartbeat:save()
                if code:raise RuntimeError(f'exit {code}: {stderr}')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
                active.remove(proc)
    try:
        for g in report['groups']:
            check();group_start=time.monotonic()
            dest=out/f"group-{g['group']:02d}";dest.mkdir()
            fixture=Path(g['fixture']);capture=dest/'capture';capture.mkdir()
            try:
                g['status']='preparing_probes';save();probe_start=time.monotonic()
                cache=dest/'probes.cache'
                if g.get('existing_cache'):
                    shutil.copyfile(g['existing_cache'],cache);g['probe_source']='unchanged original public search'
                else:
                    genv={**env,'SPRT_ALPHA':'.001','SPRT_OUT':str(capture),'SEARCH_ONLY':'1','PROBE_CACHE':str(cache),
                          'PUBLIC_KEY_OUT':str(dest/'search-public.pk')}
                    command([str(OLD/'build/collector'),str(g['seed']),'1664','1000','3000','2000000','16','6','0'],
                            dest/'probe.log',dest/'probe.err',genv)
                    if digest(dest/'search-public.pk')!=g['public_key_sha256']:raise ValueError('search public key mismatch')
                    g['probe_source']='fresh public search'
                g.update(probe_seconds=time.monotonic()-probe_start,probe_sha256=digest(cache),status='capturing');save()
                capture_start=time.monotonic();jobs=queue.Queue()
                for col in range(16):jobs.put(col)
                def worker(core):
                    while not stopping.is_set():
                        try:col=jobs.get_nowait()
                        except queue.Empty:return
                        part=capture/f'col-{col:02d}';part.mkdir()
                        command([str(HERE/'collector'),str(g['seed']),str(cache),str(fixture/'public.pk'),
                                 str(part),str(col),str(col),'1664'],part/'capture.csv',part/'capture.err',
                                {**env,'TEMPLATE_CORE':str(core)},heartbeat=False)
                with ThreadPoolExecutor(max_workers=6) as pool:
                    futures=[pool.submit(worker,c) for c in CORES]
                    while True:
                        count=0;finished=0
                        for path in capture.glob('col-*/capture.csv'):
                            with path.open() as f:count+=max(0,sum(1 for _ in f)-1)
                            err=path.with_suffix('.err')
                            finished+=err.exists() and 'finished calls=' in err.read_text()
                        g.update(captured_coefficients=count,columns_captured=finished,
                                 capture_elapsed_seconds=time.monotonic()-capture_start);save()
                        if all(f.done() for f in futures):break
                        time.sleep(30)
                    for f in futures:f.result()
                g.update(capture_seconds=time.monotonic()-capture_start,status='postprocessing');save()
                if g['captured_coefficients']!=26624:raise ValueError('incomplete capture')
                calls=0;shots=0;template_calls=0;trusted=0
                for part in sorted(capture.glob('col-*')):
                    log=(part/'capture.err').read_text();match=re.search(r'finished calls=(\d+)',log)
                    if not match:raise ValueError('missing completion')
                    calls+=int(match[1])
                    for row in csv.DictReader((part/'capture.csv').open()):
                        shots+=int(row['n_a'])+int(row['n_b']);trusted+=int(row['trusted'])
                    with (part/'template-raw.csv').open() as f:template_calls+=2*(sum(1 for _ in f)-1)
                if calls!=2*shots+template_calls:raise ValueError('query accounting mismatch')
                g.update(timing_calls=calls,target_calls=2*shots,template_calls=template_calls,trusted=trusted,forced=26624-trusted)
                check();t=time.monotonic()
                command([sys.executable,str(HERE/'postprocess.py'),str(capture),str(fixture),str(dest/'recovery')],
                        dest/'postprocess.log',dest/'postprocess.err',env)
                g['postprocess_seconds']=time.monotonic()-t
                rec=json.loads((dest/'recovery/results.json').read_text())
                if rec['status']!='complete':raise ValueError('incomplete public recovery')
                g.update(public_accepted_columns=rec['public_accepted_columns'],kem_verified=False,status='auditing');save()
                if rec['public_accepted_columns']==16:
                    v=json.loads((dest/'recovery/verification.json').read_text())
                    if v['public_equations_passed']!=26624 or not all(v[k] for k in
                       ['fresh_kem_shared_secret_match','authenticated_message_decrypted','tampered_aead_rejected','corrupted_secret_rejected']):
                        raise ValueError('KEM/AEAD failed')
                    g['kem_verified']=True
                audit=dest/'audit';audit.mkdir()
                subprocess.run([str(OLD/'build/key-fixture'),str(g['seed']),str(audit/'public.pk'),str(audit/'truth.bin')],check=True)
                if digest(audit/'public.pk')!=g['public_key_sha256']:raise ValueError('audit key mismatch')
                truth=np.fromfile(audit/'truth.bin',dtype=np.uint16).astype(np.int64).reshape(16,1664);truth=(truth+512)%1024-512
                with np.load(dest/'recovery/columns/input-state.npz') as p:
                    wrong=p['guesses']!=truth
                    stats=dict(initial_errors=int(wrong.sum()),trusted_errors=int((wrong&p['trusted']).sum()),
                               forced_errors=int((wrong&~p['trusted']).sum()),errors_per_column=wrong.sum(axis=1).tolist())
                stats['accepted_truth_mismatches']=0
                for c in rec['columns']:
                    if c['public_valid']:
                        s=np.load(dest/f"recovery/columns/column-{c['column']}.npy")
                        stats['accepted_truth_mismatches']+=int((s!=truth[c['column']]).sum())
                (audit/'results.json').write_text(json.dumps(stats,indent=2)+'\n')
                if digest(cache)!=g['probe_sha256']:raise ValueError('probe changed')
                check();g.update(stats,status='complete',input_hashes_verified=True)
            except Exception as exc:
                g.update(status='failed',error=repr(exc));print(repr(exc),flush=True)
            g['total_seconds']=time.monotonic()-group_start
            (dest/'summary.json').write_text(json.dumps(g,indent=2)+'\n');save()
        report['status']='complete' if all(g['status']=='complete' for g in report['groups']) else 'complete_with_failures';save()
    except BaseException as exc:
        report.update(status='interrupted',error=repr(exc));save()
        for proc in list(active):
            if proc.poll() is None:proc.terminate()
        raise

if __name__=='__main__':main()
