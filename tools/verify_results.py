#!/usr/bin/env python3
"""Check frozen results and verify recovered keys using only public inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'.work/data')
    parser.add_argument('--build',type=Path,default=ROOT/'.work/build')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/results-verification.json')
    parser.add_argument('--fresh-kem',action='store_true')
    args=parser.parse_args()
    index=json.loads((ROOT/'experiment-index.json').read_text())
    report={'scope':'Original paper-selected results; fresh verification does not rerun timing acquisition.','levels':{}}
    mapping={};source_checks=[]
    for level in ['256','384','512']:
        config=index['levels'][level]
        base=args.data/f'l{level}'/config['experiment']
        runs=sorted((base/'runs').glob('group-*')) if level!='512' else [base/'runs/512']
        if len(runs)!=config['keys']:raise ValueError(f'wrong run count for L{level}')
        times=[];calls=[];rows=[]
        for run in runs:
            summary=json.loads((run/'summary.json').read_text())
            recovery=json.loads((run/'recovery/results.json').read_text())
            verification=json.loads((run/'recovery/verification.json').read_text())
            if not summary['kem_verified'] or summary['accepted_truth_mismatches']!=0:raise ValueError('unsuccessful original run')
            if not verification['fresh_kem_shared_secret_match']:raise ValueError('missing original KEM verification')
            seconds=summary['capture_seconds'] if level=='384' else summary['physical_capture_seconds']
            times.append(seconds/60);calls.append(summary['timing_calls'])
            dest=ROOT/'results'/f'l{level}'/run.name;dest.mkdir(parents=True,exist_ok=True)
            for source,name in [(run/'summary.json','summary.json'),(run/'recovery/results.json','recovery.json'),(run/'recovery/verification.json','verification.json')]:
                shutil.copyfile(source,dest/name)
                mapping[str((dest/name).relative_to(ROOT))]={'archive_member':str(source.relative_to(args.data/f'l{level}')),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            for original,expected in recovery.get('inputs_sha256',{}).items():
                marker='/scloudplus-timing-sca/'
                if marker not in original:continue
                relative=original.split(marker,1)[1]
                source=ROOT/'source'/f'l{level}'/relative
                if source.is_file() and source.suffix in ('.py','.c','.h'):
                    actual=hashlib.sha256(source.read_bytes()).hexdigest()
                    if actual!=expected:raise ValueError(f'historical source mismatch: {source}')
                    source_checks.append({'level':int(level),'run':run.name,'source':relative,'sha256':actual})
            if level=='384':
                public=base/'fixtures'/run.name/'public.npz';pk=public.with_name('public.pk')
            else:
                public=run/'capture/public.npz';pk=public.with_name('public.pk')
            fresh=None
            if args.fresh_kem:
                out=ROOT/'reports/fresh-verification'/f'l{level}'/(run.name+'.json');out.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,str(ROOT/'source'/f'l{level}'/'scripts/verify_public_recovery.py'),
                         '--level',level,'--public',str(public.resolve()),'--pk',str(pk.resolve()),
                         '--columns',str((run/'recovery/columns').resolve()),
                         '--verifier',str((args.build/f'l{level}/public-recovery').resolve()),'--out',str(out)]
                with out.with_suffix('.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                fresh=json.loads(out.read_text())['fresh_kem_shared_secret_match']
            row={'seed':summary['seed'],'online_minutes':seconds/60,'timing_calls':summary['timing_calls'],
                 'initial_errors':summary.get('initial_errors',summary.get('initial_wrong_non_erased')),
                 'final_errors':summary['accepted_truth_mismatches'],'original_kem_verified':summary['kem_verified'],'fresh_kem_verified':fresh}
            rows.append(row);print(f'L{level} {run.name}: verified',flush=True)
        mean=statistics.mean(times)
        if round(mean,1)!=config['online_minutes']:raise ValueError('abstract time mismatch')
        report['levels'][level]={'keys':len(runs),'successes':len(runs),'mean_online_minutes':mean,'rounded_online_minutes':round(mean,1),'mean_timing_calls':statistics.mean(calls),'runs':rows}
    report['historical_source_hash_checks']=len(source_checks)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'provenance/record-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (ROOT/'reports/historical-source-checks.json').write_text(json.dumps(source_checks,indent=2)+'\n')
if __name__=='__main__':main()
