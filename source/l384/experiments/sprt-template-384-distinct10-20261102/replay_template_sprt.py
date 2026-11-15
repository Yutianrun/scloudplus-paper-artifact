"""Replay public-template SPRT logs without secret labels; export complete states.

Template training and validation remain separate. Confidence margins are ranking
evidence, not coefficient posteriors. Incomplete captures cannot be exported as
full lattice inputs.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

THETA=np.array([.51,.525,.55,.6,.7,.8,.9])
HIT=np.log(2*THETA);MISS=np.log(2*(1-THETA))

def logmix(values):
    m=np.max(values)
    return float(m+np.log(np.exp(values-m).mean()))

def replay(pairs,samples,alpha=.001,minimum=32):
    logs=np.zeros((3,2,7));side=np.full(3,-1,dtype=int)
    counts=np.zeros(2,dtype=int);winner=-1
    for axis,index,value in samples:
        assert winner<0,'observations continued after coefficient stopping'
        assert index==counts[axis]+1
        counts[axis]+=1
        for p,pair in enumerate(pairs):
            if int(pair['axis'])!=axis or side[p]>=0:continue
            boundary=float(pair['boundary'])
            if value==boundary:continue
            logs[p,0]+=HIT if value<boundary else MISS
            logs[p,1]+=HIT if value>boundary else MISS
        used={int(p['axis']) for p in pairs}
        if not all(counts[a]==max(counts) for a in used) or max(counts)%16:continue
        wins=np.zeros(3,dtype=int)
        for p,pair in enumerate(pairs):
            if not int(pair['validated']):continue
            evidence=[logmix(logs[p,s]) for s in range(2)]
            if side[p]<0 and counts[int(pair['axis'])]>=minimum and max(evidence)>=np.log(6/alpha):
                side[p]=int(evidence[1]>evidence[0])
            if side[p]>=0:
                t=int(pair['right']) if side[p]==int(pair['right_high']) else int(pair['left'])
                wins[t+1]+=1
        winners=np.flatnonzero(wins==2)
        if len(winners):winner=int(winners[0])
    score=np.full(3,np.inf)
    for p,pair in enumerate(pairs):
        delta=logmix(logs[p,1])-logmix(logs[p,0])
        right=delta if int(pair['right_high']) else -delta
        l,r=int(pair['left'])+1,int(pair['right'])+1
        score[l]=min(score[l],-right);score[r]=min(score[r],right)
    best=winner if winner>=0 else int(np.argmax(score))
    return dict(guess=best-1,trusted=winner>=0,margin=float(score[best]-np.delete(score,best).max()),
                scores=score,counts=counts)

def verify(directory):
    pairs={}
    for row in csv.DictReader((directory/'templates.csv').open()):
        key=(int(row['column']),int(row.get('epoch',0)),int(row.get('attempt',0)))
        pairs.setdefault(key,[]).append(row)
    # Each refresh must have an explicit epoch; reject ambiguous legacy logs.
    assert all(len(rows)==3 for rows in pairs.values())
    template_samples={}
    for r in csv.DictReader((directory/'template-raw.csv').open()):
        key=tuple(int(r[k]) for k in ['column','stage','axis','trit'])+(int(r.get('epoch',0)),int(r.get('attempt',0)))
        template_samples.setdefault(key,[]).append(float(r['value']))
    for (col,epoch,attempt),rows in pairs.items():
        for pair in rows:
            ncal=int(pair.get('ncal',256));attempts=3 if 'ncal' in pair else 1
            axis=int(pair['axis']);l=int(pair['left']);r=int(pair['right'])
            left=np.array(template_samples[col,0,axis,l,epoch,attempt]);right=np.array(template_samples[col,0,axis,r,epoch,attempt])
            assert len(left)==len(right)==ncal
            ml,mr=np.median(left),np.median(right)
            np.testing.assert_allclose(float(pair['boundary']),(ml+mr)/2,rtol=1e-12,atol=1e-9)
            assert int(pair['right_high'])==(mr>ml)
            boundary=float(pair['boundary']);high=int(pair['right_high'])
            correct=[]
            for t,direction in [(l,not high),(r,high)]:
                x=np.array(template_samples[col,1,axis,t,epoch,attempt]);assert len(x)==ncal
                correct.append(int(np.sum(x>boundary if direction else x<boundary)))
            assert correct==[int(pair['left_correct']),int(pair['right_correct'])]
            valid=min(correct)/ncal-np.sqrt(np.log(6*attempts/.001)/(2*ncal))>.5
            assert valid==bool(int(pair['validated']))
    raw={}
    for row in csv.DictReader((directory/'raw.csv').open()):
        key=int(row['column']),int(row['position'])
        raw.setdefault(key,[]).append((int(row['axis']),int(row['sample']),float(row['value'])))
    rows=list(csv.DictReader((directory/'capture.csv').open()))
    seen=set();results=[]
    for row in rows:
        key=int(row['column']),int(row['position']);assert key not in seen;seen.add(key)
        epoch=int(row.get('epoch',0))
        d=replay(pairs[key[0],epoch,int(row.get('attempt',0))],raw[key])
        assert d['guess']==int(row['guess']) and d['trusted']==bool(int(row['trusted']))
        np.testing.assert_allclose(d['margin'],float(row['margin']),atol=1e-9)
        np.testing.assert_allclose(d['scores'],[float(row[k]) for k in ['score_neg','score_zero','score_pos']],atol=1e-9)
        np.testing.assert_equal(d['counts'],[int(row['n_a']),int(row['n_b'])])
        assert max(d['counts'])<=1000
        if not d['trusted']:assert max(d['counts'])==1000
        results.append(dict(column=key[0],position=key[1],guess=d['guess'],trusted=d['trusted'],margin=d['margin']))
    assert seen==set(raw)
    return results

def export_state(rows,out,n,nbar):
    if len(rows)!=n*nbar or {(r['column'],r['position']) for r in rows}!={(c,k) for c in range(nbar) for k in range(n)}:
        raise ValueError('full lattice input requires exactly one decision for every coefficient')
    guesses=np.zeros((nbar,n),dtype=np.int64);trusted=np.zeros_like(guesses,dtype=bool)
    margins=np.zeros_like(guesses,dtype=float)
    for r in rows:
        key=r['column'],r['position'];guesses[key]=r['guess'];trusted[key]=r['trusted'];margins[key]=r['margin']
    orders=np.array([np.lexsort((np.arange(n),margins[c],trusted[c].astype(int))) for c in range(nbar)])
    np.savez_compressed(out,guesses=guesses,orders=orders,trusted=trusted,confidence_margin=margins,
                        erasures=np.zeros_like(trusted))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--state-out',type=Path);p.add_argument('--n',type=int,default=1664);p.add_argument('--nbar',type=int,default=16)
    args=p.parse_args();rows=verify(args.directory)
    if args.state_out:export_state(rows,args.state_out,args.n,args.nbar)
    result=dict(decisions_replayed=len(rows),trusted=sum(r['trusted'] for r in rows),
                forced=sum(not r['trusted'] for r in rows),truth_read=False,
                template_boundaries_and_holdouts_verified=True)
    (args.directory/'replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
