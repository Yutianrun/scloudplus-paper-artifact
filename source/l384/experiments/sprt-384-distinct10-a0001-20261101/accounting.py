import csv,re
import numpy as np

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
    if finish is None or float(finish[1])!=.001:raise ValueError('completion or alpha mismatch')
    return dict(physical_capture_seconds=float(finish[2]),timing_calls=calls,timing_costs=costs,
                coefficients=total,mean_samples_per_axis=float(ns.mean()),capped_axes=int((ns==1000).sum()),
                collector_abstentions=sum(r['guess']=='99' for r in rows),
                sample_quantiles=np.quantile(ns,[0,.5,.9,.95,.99,1]).tolist(),
                reference_rejected_coefficients=sum(r['ref_ok']=='0' for r in rows),
                workers=[dict(worker=int(w),coefficients=int(c),timing_calls=int(k),seconds=float(s)) for w,c,k,s in workers])

