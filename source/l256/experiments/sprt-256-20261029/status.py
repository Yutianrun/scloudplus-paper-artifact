"""Read live progress without touching oracle samples or audit labels."""
import json
from pathlib import Path
import re
import time

HERE=Path(__file__).resolve().parent


def status():
    out=HERE/'full'
    if not (out/'results.json').exists():return {'status':'not_started'}
    run=json.loads((out/'results.json').read_text())
    result={k:run[k] for k in ('status','stage','total_seconds','public_accepted_columns') if k in run}
    if run['status']=='running':
        result.pop('total_seconds',None)
        if run['stage']=='capture':
            result['elapsed_capture_process_seconds']=round(time.time()-(out/'results.json').stat().st_mtime,1)
    text=(out/'capture.err').read_text() if (out/'capture.err').exists() else ''
    workers={}
    for match in re.finditer(r'SPRT_(PROGRESS|DONE) worker=(\d+) coefficients=(\d+) calls=(\d+) seconds=([\d.]+)',text):
        kind,w,n,calls,seconds=match.groups()
        workers[int(w)]={'done':kind=='DONE','coefficients':int(n),'calls':int(calls),'seconds':float(seconds)}
    if workers:
        n=sum(w['coefficients'] for w in workers.values())
        result.update(completed_coefficients_lower_bound=n,total_coefficients=13024,
                      last_progress_seconds=max(w['seconds'] for w in workers.values()),workers=workers)
        if len(workers)==6:
            estimates=[]
            for w,item in workers.items():
                assigned=len(range(w,13024,6))
                estimates.append(item['seconds']*assigned/item['coefficients'])
            result['estimated_capture_seconds_if_current_worker_rates_persist']=max(estimates)
    else:result['latest_log_lines']=text.splitlines()[-3:]
    return result


if __name__=='__main__':print(json.dumps(status(),indent=2))
