"""Force an output at the SPRT cap; keep confidence and correct weakest guesses first."""
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
from evaluate_confidence import read_log
from sprt_soft_decision import soft_decision


def main():
    out=HERE/'soft-decision'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    paths=[HERE/'full/capture.log',ROOT/'scripts/sprt_soft_decision.py',ROOT/'scripts/sequential_oracle.py',Path(__file__),
           BASE/'probes.cache',BASE/'public.pk',BASE/'public.npz',
           ROOT/'scripts/evaluate_confidence.py',ROOT/'scripts/public_embedding_cpu.py']
    paths += sorted((HERE/'full').glob('raw-*.csv'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    policy=[(60,80,1,25),(120,128,1,25),(180,140,2,25),(236,160,2,25),
            (360,200,2,45),(640,256,2,45)]
    report=dict(status='running',stage='soft decision',level=256,seed=101,vote3_used=False,
                extra_measurements=0,source='full live SPRT raw samples',
                confidence='best minus runner-up summed mixture log evidence; forced decisions first',
                previous_solutions_used=False,truth_used_for_selection=False,
                workers=3,coefficients=13024,policy=policy,inputs_sha256=hashes)
    def save():
        report['total_seconds']=time.monotonic()-start
        temp=out/'results.json.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'results.json')
    save()
    try:
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        for name in ('public.pk','public.npz'):
            shutil.copyfile(HERE/'full'/name,out/name)
        rows=list(csv.DictReader(line for line in (HERE/'full/capture.log').read_text().splitlines()
                                 if not line.startswith('#')))
        indexed={(int(r['coord']),int(r['position'])):r for r in rows}
        if len(indexed)!=13024:raise ValueError('incomplete capture')
        guesses=np.zeros((11,1184),dtype=np.int64);orders=guesses.copy()
        erasures=np.zeros_like(guesses,dtype=bool)
        margins=np.zeros_like(guesses,dtype=float);trusted=np.zeros_like(erasures)
        seen=np.zeros_like(erasures);score_rows=[]
        def decide(key,samples):
            r=indexed[key]
            if [len(x) for x in samples]!=[int(r['n_a']),int(r['n_b'])]:
                raise ValueError('sample counts differ')
            table=[[int(r[f'{a}_{t}']) for a in ('a','b')] for t in ('neg','zero','pos')]
            d=soft_decision(samples,table,float(r['gap_ns']),int(r['ref_bucket']),
                            int(r['guess']),bool(int(r['ref_ok'])))
            guesses[key]=d['guess'];margins[key]=d['margin'];trusted[key]=d['trusted'];seen[key]=True
            score_rows.append(dict(column=key[0],position=key[1],guess=d['guess'],
                                   margin=d['margin'],trusted=d['trusted'],
                                   original_abstention=int(r['guess'])==99,
                                   reference_valid=bool(int(r['ref_ok'])),
                                   candidate_scores=d['candidate_scores'].tolist(),
                                   axis_loge=d['axis_loge'].tolist()))
        for path in sorted((HERE/'full').glob('raw-*.csv')):
            key=None;samples=[[],[]]
            with path.open() as stream:
                for line in stream:
                    col,pos,axis,n,value=line.rstrip().split(',')
                    newkey=(int(col),int(pos));axis=int(axis)
                    if newkey!=key:
                        if key is not None:decide(key,samples)
                        key=newkey;samples=[[],[]]
                    if int(n)!=len(samples[axis])+1:raise ValueError('raw sequence mismatch')
                    samples[axis].append(float(value))
            if key is not None:decide(key,samples)
        if not seen.all():raise ValueError('missing raw observations')
        for col in range(11):
            orders[col]=np.lexsort((np.arange(1184),margins[col],trusted[col].astype(int)))
        with (out/'confidence.jsonl').open('w') as stream:
            for row in sorted(score_rows,key=lambda r:(r['column'],r['position'])):
                stream.write(json.dumps(row)+'\n')
        report.update(forced_guesses=int((~trusted).sum()),trusted_guesses=int(trusted.sum()),
                      soft_decision_seconds=time.monotonic()-start,stage='public correction')
        save()
        columns=out/'columns';columns.mkdir()
        np.savez_compressed(columns/'input-state.npz',guesses=guesses,orders=orders,erasures=erasures,
                            confidence_margin=margins,trusted=trusted)
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
        initial_wrong=int((guesses!=labels).sum())
        report['forced_guess_errors']=int(((guesses!=labels)&~trusted).sum())
        report['trusted_guess_errors']=int(((guesses!=labels)&trusted).sum())
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
