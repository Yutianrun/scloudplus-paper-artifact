"""Full 256 live-SPRT capture, blind public correction, then separate truth audit."""
from concurrent.futures import ThreadPoolExecutor
import csv
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
BASE=ROOT/'experiments/freeze80-256-e2e10-20261027-113105/experiment-01'
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_confidence import read_log,confidence_scores


def main():
    out=HERE/'full'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    paths=[HERE/'collector',HERE/'collector.c',HERE/'sprt.h',Path(__file__),
           BASE/'probes.cache',BASE/'public.pk',BASE/'public.npz',
           ROOT/'scripts/evaluate_confidence.py',ROOT/'scripts/public_embedding_cpu.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policy=[(60,80,1,25),(120,128,1,25),(180,140,2,25),(236,160,2,25),
            (360,200,2,45),(640,256,2,45)]
    report=dict(status='running',stage='capture',level=256,seed=101,vote3_used=False,
                alpha_per_axis=.001,min_samples=32,batch_samples=16,max_samples=1000,
                workers=6,coefficients=13024,calibration_refresh_coefficients=128,
                local_reference_pairs_per_coefficient=32,policy=policy,inputs_sha256=hashes)
    def save():
        report['total_seconds']=time.monotonic()-start
        temp=out/'results.json.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'results.json')
    save()
    try:
        env={**os.environ,'SPRT_OUT':str(out),'SPRT_ALPHA':'.001','PROBE_CACHE':str(BASE/'probes.cache'),
             'PUBLIC_KEY_OUT':str(out/'public.pk'),'TIMING_TSC':'1','TIMING_ABBA':'0',
             'TIMING_INTERLEAVE':'1','JOINT_DECODE':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
             'BATCH_WORKERS':'6','SERIAL_CAL':'0','ORACLE':'0','SEARCH_ONLY':'0','PROFILE_ONLY':'0'}
        env.pop('RETEST_CSV',None);env.pop('PROBE_VALIDATE',None);env.pop('PROBE_VALIDATE_TOL',None)
        args=[str(HERE/'collector'),'101','1184','1000','3000','2000000','11','6','0']
        report['capture_command']=args;save()
        capture_start=time.monotonic()
        with (out/'capture.log').open('w') as stdout,(out/'capture.err').open('w') as stderr:
            subprocess.run(args,stdout=stdout,stderr=stderr,env=env,cwd=ROOT,check=True)
        report['capture_process_seconds']=time.monotonic()-capture_start
        if (out/'public.pk').read_bytes() != (BASE/'public.pk').read_bytes():raise ValueError('key mismatch')
        shutil.copyfile(BASE/'public.npz',out/'public.npz')
        lines=(out/'capture.log').read_text().splitlines()
        rows=list(csv.DictReader([line for line in lines if not line.startswith('#')]))
        if len(rows)!=13024:raise ValueError('incomplete capture')
        na=np.array([int(r['n_a']) for r in rows]);nb=np.array([int(r['n_b']) for r in rows])
        err=(out/'capture.err').read_text()
        workers=re.findall(r'SPRT_DONE worker=(\d+) coefficients=(\d+) calls=(\d+) seconds=([\d.]+)',err)
        if len(workers)!=6:raise ValueError('missing worker costs')
        total_calls=sum(int(w[2]) for w in workers)
        cal_attempts=len(re.findall(r'CAL worker=\d+ attempt=',err))
        costs=dict(warmup=6*400,calibration=cal_attempts*6000,
                   target_and_paired_reference=2*int((na+nb).sum()),local_reference=13024*64)
        if sum(costs.values())!=total_calls:raise ValueError('timing call accounting mismatch')
        wall=float(re.search(r'SPRT_FINISHED .* wall=([\d.]+)',err)[1])
        report.update(stage='public correction',physical_capture_seconds=wall,timing_calls=total_calls,
                      timing_costs=costs,mean_samples_per_axis=float(np.r_[na,nb].mean()),
                      sample_quantiles=np.quantile(np.r_[na,nb],[0,.5,.9,.95,.99,1]).tolist(),
                      capped_axes=int(np.count_nonzero(np.r_[na,nb]==1000)),
                      capture_abstentions=sum(int(r['guess'])==99 for r in rows),
                      reference_rejected_coefficients=sum(r['ref_ok']=='0' for r in rows))
        save()
        f,_=read_log(out/'capture.log',include_truth=False)
        scores=confidence_scores(f)['l2sq_margin']
        guesses=np.zeros((11,1184),dtype=np.int64);orders=guesses.copy();erasures=np.zeros_like(guesses,dtype=bool)
        for col in range(11):
            ids=np.flatnonzero(f['coord']==col);pos=f['position'][ids].astype(int)
            erased=f['guess'][ids]==99
            guesses[col,pos]=np.where(erased,0,f['guess'][ids]).astype(int)
            erasures[col,pos]=erased;rank=scores[ids].copy();rank[erased]=-np.inf
            orders[col]=pos[np.lexsort((pos,rank))]
        columns=out/'columns';columns.mkdir()
        np.savez_compressed(columns/'input-state.npz',guesses=guesses,orders=orders,erasures=erasures)
        solve_start=time.monotonic()
        def solve(col):
            attempts=[]
            for window,equations,weight,seconds in policy:
                if int(erasures[col].sum())>window:
                    attempts.append(dict(window=window,status='erasures_exceed_support'));continue
                dest=columns/f'col-{col}-w{window}'
                cmd=[sys.executable,str(ROOT/'scripts/public_embedding_cpu.py'),'--public',str(out/'public.npz'),
                     '--state',str(columns/'input-state.npz'),'--out',str(dest),'--column',str(col),
                     '--window',str(window),'--equations',str(equations),'--weight',str(weight),
                     '--beta','40','--cores','1','--seconds',str(seconds)]
                with dest.with_suffix('.log').open('w') as log:
                    try:status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=seconds+20).returncode
                    except subprocess.TimeoutExpired:status='timeout'
                attempts.append(dict(window=window,status=status))
                if status==0:
                    shutil.copyfile(dest/f'column-{col}.npy',columns/f'column-{col}.npy');break
            return dict(column=col,public_valid=(columns/f'column-{col}.npy').exists(),attempts=attempts)
        report['columns']=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for record in pool.map(solve,range(11)):
                report['columns'].append(record);save()
        report['correction_seconds']=time.monotonic()-solve_start
        with np.load(out/'public.npz',allow_pickle=False) as p:A,B=p['A'].astype(np.int64),p['B'].astype(np.int64)
        accepted={}
        for record in report['columns']:
            if not record['public_valid']:continue
            col=record['column'];s=np.load(columns/f'column-{col}.npy',allow_pickle=False)
            if (s.shape!=(1184,) or not np.issubdtype(s.dtype,np.integer) or not np.isin(s,[-1,0,1]).all()
                    or np.any(abs((B[:,col]-A@s+512)%1024-512)>1)):raise ValueError('public rejection')
            accepted[col]=s
        report['public_accepted_columns']=len(accepted)
        report['stage']='public acceptance complete';save()
        (out/'public-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
        if len(accepted)==11:
            with (out/'verification.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/verify_public_recovery.py'),
                                '--public',str(out/'public.npz'),'--pk',str(out/'public.pk'),
                                '--columns',str(columns),'--verifier',str(BASE/'public-recovery'),
                                '--out',str(out/'verification.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        # Only now load the existing same-key labels. Never used to solve or rank.
        audit_f,truth=read_log(BASE/'256-run1.log',include_truth=True)
        labels=np.empty((11,1184),dtype=np.int64)
        labels[audit_f['coord'].astype(int),audit_f['position'].astype(int)]=truth
        initial_wrong=int(((guesses!=labels)&~erasures).sum())
        corrected=filled=remaining=0
        for record in report['columns']:
            col=record['column']
            record.update(initial_wrong_non_erased=int(((guesses[col]!=labels[col])&~erasures[col]).sum()),
                          erasures=int(erasures[col].sum()))
            if col in accepted:
                mismatch=int(np.count_nonzero(accepted[col]!=labels[col]));remaining+=mismatch
                corrected+=record['initial_wrong_non_erased'];filled+=record['erasures']
                record['accepted_truth_mismatches']=mismatch
        report.update(status='complete',stage='complete',initial_wrong_non_erased=initial_wrong,
                      corrected_wrong_non_erased=corrected,completed_erasures=filled,
                      accepted_truth_mismatches=remaining,
                      all_public_checks_precede_truth_audit=True,
                      input_hashes_verified=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
        save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
