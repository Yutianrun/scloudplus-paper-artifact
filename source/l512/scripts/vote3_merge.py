#!/usr/bin/env python3
"""Majority-vote merge of three independent N=100 captures into one log.

Three runs with the SAME key measure each (coord,position) independently. The
physical guess is a ternary class in {-1,0,+1}. Per coordinate we take the
majority vote across the three runs; this cancels the random single-shot
misclassifications that make a single N=100 run ~96.7% accurate.

Output is ONE CSV with the e2e-log fields plus vote_runs/vote_count markers,
so postprocess_recover.py preserves the voted classes. For the
feature columns (median_*, ref_bucket, a_*/b_* buckets, gap_ns, ...) we emit the
row from the run whose guess WON the vote and, among the winners, has the
largest bucket margin (most confident) -- so the confidence ranking downstream
reflects the measurement that actually decided the class.

Runs must have identical coordinate sets and public candidate bucket tables.
Also pin the public key and complete probe cache when collecting captures:
matching bucket tables alone cannot prove that the full probes are identical.
Invalid guesses (99) abstain. All-abstained coordinates require remeasurement.
Representative margins are ranking heuristics, not calibrated vote confidence.

Tie handling (all three guesses distinct, 1:1:1): pick the single most-confident
run by bucket margin. This is rare; it degrades to best-single for that coord.

Truth (`true` column) is passed through from run1 for OFFLINE scoring only; it
is never read by the recovery path. No secret, no truth is used to decide votes.
"""
import argparse,csv,sys
from pathlib import Path
from collections import defaultdict

HDR=('coord,position,true,guess,tb_base,tb_shift,nb_base,nb_shift,channel_ok,'
     'recover_ok,worker,median_a_ns,median_b_ns,gap_ns,ref_bucket,a_neg,a_zero,'
     'a_pos,b_neg,b_zero,b_pos,mad_a_ns,mad_b_ns,half_delta_a_ns,half_delta_b_ns')
COLS=HDR.split(',')
PROBE_FIELDS = ('ref_bucket', 'a_neg', 'a_zero', 'a_pos', 'b_neg', 'b_zero', 'b_pos')

def read_rows(path):
    rows={}
    hdr=None
    with open(path) as f:
        for line in f:
            if line.startswith('coord,position,true,guess,'):
                hdr=line.rstrip('\n').split(','); continue
            if hdr and line[:1].isdigit():
                parts=line.rstrip('\n').split(',')
                if len(parts)!=len(hdr):
                    raise ValueError(f'{path}: malformed data row')
                d=dict(zip(hdr,parts))
                key=(int(d['coord']),int(d['position']))
                if key in rows:
                    raise ValueError(f'{path}: duplicate coordinate {key}')
                rows[key]=d
    if not rows:
        raise SystemExit(f'{path}: no e2e rows')
    return rows

def validate_runs(runs):
    """Check observable compatibility before voting; never inspect truth.

    Matching bucket tables are necessary, not sufficient to identify a probe.
    Capture provenance must additionally pin the key and full probe cache.
    """
    if len(runs) < 2:
        raise ValueError('need at least two runs')
    keys=set(runs[0])
    for index,run in enumerate(runs[1:],2):
        if set(run)!=keys:
            raise ValueError(f'run {index}: coordinate set differs from run 1')
        for key in sorted(keys):
            if any(run[key][field]!=runs[0][key][field] for field in PROBE_FIELDS):
                raise ValueError(f'run {index}: probe table differs at {key}; '
                                 'cannot pool these runs as repeated measurements')
    return sorted(keys)

def choose_vote(cands):
    """Invalid decoder result 99 abstains; it is not a fourth secret class."""
    votes=defaultdict(list)
    for d in cands:
        cls=int(d['guess'])
        if cls in (-1,0,1):
            votes[cls].append(d)
        elif cls!=99:
            raise ValueError(f'unknown guess class: {cls}')
    if not votes:
        raise ValueError('all runs abstained at a coordinate; remeasurement required')
    best_cls=max(votes,key=lambda cls:(len(votes[cls]),max(margin(d) for d in votes[cls])))
    winners=votes[best_cls]
    return max(winners,key=margin),len(winners)

def margin(d):
    """Bucket-distance margin of the winning class: larger = more confident.
    Uses the a/b bucket triplets vs ref_bucket, same geometry as base_prediction.
    """
    try:
        ref=float(d['ref_bucket'])
        xa=float(d['median_a_ns']);xb=float(d['median_b_ns']);g=float(d['gap_ns'])
        if g<=0: return -1e9
        xa/=g; xb/=g
        cents=[]
        for t in ('neg','zero','pos'):
            ca=float(d['a_'+t])-ref; cb=float(d['b_'+t])-ref
            cents.append((xa-ca)**2+(xb-cb)**2)
        cls=int(d['guess'])
        if cls not in (-1,0,1): return -1e9
        own=cls+1
        return min(x for i,x in enumerate(cents) if i!=own)-cents[own]
    except (KeyError,ValueError):
        return 0.0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,type=Path,help='>=2 N=100 logs')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    runs=[read_rows(str(r)) for r in a.runs]
    keys=validate_runs(runs)
    print(f'{len(runs)} runs, {len(keys)} matching (coord,position)',file=sys.stderr)
    out_rows=[]
    changed=0; ties=0
    for k in sorted(keys):
        cands=[r[k] for r in runs]
        win,best_n=choose_vote(cands)
        if best_n==1: ties+=1
        if int(win['guess'])!=int(cands[0]['guess']): changed+=1
        out=dict(win)
        # Evaluation label has one fixed source, independent of the winning run.
        out['true']=cands[0]['true']
        out['vote_runs']=len(runs)
        out['vote_count']=best_n
        out_rows.append(out)
    # write
    with open(a.out,'w',newline='') as f:
        columns=COLS+['vote_runs','vote_count']
        f.write(','.join(columns)+'\n')
        w=csv.writer(f)
        for d in out_rows:
            w.writerow([d.get(c,'0') for c in columns])
    print(f'merged {len(out_rows)} rows -> {a.out}  '
          f'(changed vs run1: {changed}, pure-ties: {ties})',file=sys.stderr)

if __name__=='__main__':main()
