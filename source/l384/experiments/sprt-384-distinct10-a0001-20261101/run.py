"""Ten different 384 keys, fresh alpha .001 captures, public acceptance then audit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
import numpy as np
from build import digest,HERE,ROOT
from accounting import capture_stats

def main():
    manifest=json.loads((HERE/'build/manifest.json').read_text())
    hashes=manifest['hashes']
    def check():
        for p,h in hashes.items():
            if digest(p)!=h:raise ValueError(f'input changed: {p}')
    check()
    if len(manifest['groups'])!=10 or len({g['public_key_sha256'] for g in manifest['groups']})!=10:
        raise ValueError('ten different keys required')
    out=HERE/'runs';out.mkdir(exist_ok=False)
    start=time.monotonic()
    report=dict(status='running',pid=os.getpid(),level=384,alpha=.001,max_samples=1000,
                vote3_used=False,distinct_keys=10,groups=manifest['groups'],
                blaster='isolated GH radius fix',started_at_utc=datetime.now(timezone.utc).isoformat())
    def save():
        report.update(elapsed_seconds=time.monotonic()-start,updated_at_utc=datetime.now(timezone.utc).isoformat())
        tmp=out/'results.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'results.json')
        rows=[]
        for g in report['groups']:
            def minutes(key):return f'{g[key]/60:.1f}' if key in g else '—'
            rows.append(f"|{g['group']}|{g['seed']}|{g['public_key_sha256'][:12]}|{g['status']}|{minutes('probe_seconds')}|{minutes('physical_capture_seconds')}|{g.get('public_accepted_columns','—')}/16|{g.get('kem_verified','—')}|")
        (out/'REPORT.md').write_text('# 384：10 把不同密钥，SPRT α=0.001\n\n'+
            f"状态：{report['status']}；更新：{report['updated_at_utc']}。\n\n"+
            '|组|seed|公钥 SHA256 前缀|状态|探针分钟|采集分钟|公钥通过列|KEM|\n|---:|---:|---|---|---:|---:|---:|---|\n'+
            '\n'.join(rows)+'\n\n10 个公钥哈希已核对互异。每组新探针、新采集；N≤1000，不用vote3。\n'+
            '失败保留，不换钥重试到成功。先验收后审计真值。构建及公开导出时间另见 build/manifest.json。\n')
    save()
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPRT_','TIMING_')) and k not in
         ('ORACLE','SEARCH_ONLY','PROFILE_ONLY','PROBE_VALIDATE','PROBE_VALIDATE_TOL','PROBE_CACHE','PUBLIC_KEY_OUT','RETEST_CSV')}
    env.update(SPRT_ALPHA='.001',TIMING_TSC='1',TIMING_ABBA='0',TIMING_INTERLEAVE='1',JOINT_DECODE='0',
               BATCH_WORKERS='6',SERIAL_CAL='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',ORACLE='0',SEARCH_ONLY='0',PROFILE_ONLY='0')
    def run(command,stdout,stderr,genv,group):
        with stdout.open('w') as log,stderr.open('w') as err:
            proc=subprocess.Popen(command,cwd=ROOT,env=genv,stdout=log,stderr=err)
            group.update(active_pid=proc.pid,active_command=command);save()
            while proc.poll() is None:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:save()
            if proc.returncode:raise RuntimeError(f'exit {proc.returncode}; see {stderr}')
    try:
        for g in report['groups']:
            dest=out/f"group-{g['group']:02d}";dest.mkdir();fixture=Path(g['fixture']);gstart=time.monotonic()
            cap=dest/'capture';cap.mkdir()
            cmd=[str(HERE/'build/collector'),str(g['seed']),'1664','1000','3000','2000000','16','6','0']
            genv={**env,'SPRT_OUT':str(cap),'PROBE_CACHE':str(dest/'probes.cache'),'PUBLIC_KEY_OUT':str(cap/'public.pk')}
            try:
                check();g['status']='preparing_probes';save();t=time.monotonic()
                run(cmd,dest/'probe.log',dest/'probe.err',{**genv,'SEARCH_ONLY':'1'},g)
                g['probe_seconds']=time.monotonic()-t
                if digest(cap/'public.pk')!=g['public_key_sha256']:raise ValueError('search key mismatch')
                shutil.copyfile(dest/'probes.cache',fixture/'probes.cache')
                probe_hash=digest(fixture/'probes.cache')
                g.update(status='capturing',probe_sha256=probe_hash);save()
                run(cmd,cap/'capture.log',cap/'capture.err',genv,g)
                if digest(cap/'public.pk')!=g['public_key_sha256']:raise ValueError('capture key mismatch')
                if digest(dest/'probes.cache')!=probe_hash:raise ValueError('probe changed')
                shutil.copyfile(fixture/'public.npz',cap/'public.npz')
                g.update(capture_stats(cap,1664,16),status='postprocessing');save();check()
                t=time.monotonic()
                run([sys.executable,str(HERE/'postprocess.py'),'--capture',str(cap),'--fixture',str(fixture),
                     '--level','384','--out',str(dest/'recovery')],dest/'postprocess.log',dest/'postprocess.err',env,g)
                g['postprocess_seconds']=time.monotonic()-t
                result=json.loads((dest/'recovery/results.json').read_text())
                if result['status']!='complete' or not result['input_hashes_verified']:raise ValueError('postprocess incomplete')
                g['public_accepted_columns']=result['public_accepted_columns'];g['kem_verified']=False
                if g['public_accepted_columns']==16:
                    v=json.loads((dest/'recovery/verification.json').read_text())
                    if v['public_equations_passed']!=26624 or not all(v[k] for k in
                       ['fresh_kem_shared_secret_match','authenticated_message_decrypted','tampered_aead_rejected','corrupted_secret_rejected']):
                        raise ValueError('verification failed')
                    g['kem_verified']=True
                # Generate test truth only after recovery and independent public acceptance.
                g['status']='auditing';save()
                audit=dest/'audit';audit.mkdir()
                subprocess.run([str(HERE/'build/key-fixture'),str(g['seed']),str(audit/'public.pk'),str(audit/'truth.bin')],check=True)
                if digest(audit/'public.pk')!=g['public_key_sha256']:raise ValueError('audit key mismatch')
                truth=np.fromfile(audit/'truth.bin',dtype=np.uint16).astype(np.int64).reshape(16,1664)
                truth=(truth+512)%1024-512
                with np.load(fixture/'public.npz') as p:
                    residual=(p['B'].astype(np.int64)-p['A'].astype(np.int64)@truth.T+512)%1024-512
                    if abs(residual).max()>1:raise ValueError('audit public mismatch')
                with np.load(dest/'recovery/columns/input-state.npz') as p:
                    wrong=p['guesses']!=truth
                    stats=dict(initial_errors=int(wrong.sum()),trusted_errors=int((wrong&p['trusted']).sum()),
                               forced_errors=int((wrong&~p['trusted']).sum()),errors_per_column=wrong.sum(axis=1).tolist())
                stats['accepted_truth_mismatches']=0
                for c in result['columns']:
                    if c['public_valid']:
                        s=np.load(dest/f"recovery/columns/column-{c['column']}.npy")
                        stats['accepted_truth_mismatches']+=int((s!=truth[c['column']]).sum())
                (audit/'results.json').write_text(json.dumps(stats,indent=2)+'\n')
                g.update(stats,status='complete');check()
            except Exception as exc:
                g.update(status='failed',error=repr(exc));print(g['group'],repr(exc),flush=True)
            g.update(total_seconds=time.monotonic()-gstart)
            (dest/'summary.json').write_text(json.dumps(g,indent=2)+'\n');save()
        report['status']='complete' if all(g['status']=='complete' for g in report['groups']) else 'complete_with_failures';save()
    except BaseException as exc:
        report.update(status='interrupted',error=repr(exc));save();raise

if __name__=='__main__':main()
