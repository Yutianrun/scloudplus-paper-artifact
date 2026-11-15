"""Replay template evidence, blind lattice correction, then public verification."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from replay_template_sprt import verify,export_state

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'sprt-384-distinct10-a0001-20261101'
POLICY=[(60,80,1,25),(120,128,1,25),(180,140,2,25),(236,160,2,25),(360,200,2,45),(640,256,2,45)]

def main():
    capture,fixture,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
    start=time.monotonic();columns=out/'columns';columns.mkdir()
    report=dict(status='running',stage='independent raw replay',truth_used=False,columns=[],policy=POLICY)
    def save():
        report['seconds']=time.monotonic()-start
        temp=out/'results.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'results.json')
    save();rows=[]
    for col in range(16):
        part=verify(capture/f'col-{col:02d}')
        if len(part)!=1664 or any(r['column']!=col for r in part):raise ValueError('incomplete/wrong column')
        rows.extend(part);report['columns_replayed']=col+1;save()
    export_state(rows,columns/'input-state.npz',1664,16)
    report.update(stage='public correction',trusted=sum(r['trusted'] for r in rows),
                  forced=sum(not r['trusted'] for r in rows),replay_seconds=time.monotonic()-start)
    with np.load(fixture/'public.npz') as p:A,B=p['A'].astype(np.int64),p['B'].astype(np.int64)
    with np.load(columns/'input-state.npz') as p:guesses=p['guesses'].copy()
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def valid(s,col):return s.shape==(1664,) and np.isin(s,[-1,0,1]).all() and np.all(abs((B[:,col]-A@s+512)%1024-512)<=1)
    def solve(col):
        attempts=[]
        if valid(guesses[col],col):
            np.save(columns/f'column-{col}.npy',guesses[col])
            return dict(column=col,public_valid=True,method='initial public check',attempts=attempts)
        for window,equations,weight,seconds in POLICY:
            dest=columns/f'col-{col}-w{window}'
            cmd=[sys.executable,str(OLD/'solver.py'),'--public',str(fixture/'public.npz'),
                 '--state',str(columns/'input-state.npz'),'--out',str(dest),'--column',str(col),
                 '--window',str(window),'--equations',str(equations),'--weight',str(weight),
                 '--beta','40','--cores','1','--seconds',str(seconds)]
            with dest.with_suffix('.log').open('w') as log:
                try:status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=seconds+20).returncode
                except subprocess.TimeoutExpired:status='timeout'
            attempts.append(dict(window=window,status=status))
            if status==0:
                s=np.load(dest/f'column-{col}.npy')
                if not valid(s,col):raise ValueError('independent public rejection')
                np.save(columns/f'column-{col}.npy',s);break
        return dict(column=col,public_valid=(columns/f'column-{col}.npy').exists(),method='BLASter GH-fixed',attempts=attempts)
    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(solve,c) for c in range(16)]):
            report['columns'].append(future.result());report['columns'].sort(key=lambda r:r['column']);save()
    report['public_accepted_columns']=sum(r['public_valid'] for r in report['columns'])
    report['stage']='public acceptance complete';save()
    (out/'public-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['public_accepted_columns']==16:
        with (out/'verification.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_public_recovery.py'),
                '--level','384','--public',str(fixture/'public.npz'),'--pk',str(fixture/'public.pk'),
                '--columns',str(columns),'--verifier',str(OLD/'build/public-recovery'),
                '--out',str(out/'verification.json')],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    report.update(status='complete',stage='complete',truth_audit_deferred=True);save()

if __name__=='__main__':main()
