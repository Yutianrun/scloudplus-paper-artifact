#!/usr/bin/env python3
"""Unified post-processing: timing log -> recovered ternary S, BLASter-only.

Production recovery path wired into run_256.sh. For each of the nbar columns:
  1. preserve marked vote predictions; raw captures use nearest-bucket classes
  2. confidence = joint raw-capture evidence with --confidence-runs;
     otherwise corrected_scores coord_standardized/prior_margin_8
  3. freeze the high-confidence majority; the support = the least-confident W
  4. BLASter reduces the sparse-correction (syndrome) lattice to fix the support
  5. accept a column only when every public equation residual lands in {-1,0,1}
  6. ADAPTIVE: if no candidate satisfies the equations, grow W and retry.
     Failure can also reflect insufficient lattice reduction. Public equations are the
     oracle; no secret, no timing correctness label is ever read here.

Runs under BLASter's venv python (needs `blaster`; must NOT import fpylll).
A separate offline step may compare against truth, but this path does not.

SCOPE / LIMIT: validated for Scloudplus-256 (all 11 columns; worst col2 needs
support W=384, within the adaptive ladder below). It relies on every error
living in the low-confidence tail. This does NOT hold for 384/512: there,
individual columns hide an error in the high-confidence region (384 col3 has an
error at confidence rank 1661/1664; 512 col6 at 1613/2400), so freezing the
high-confidence majority welds that error in, and adaptive growth would need a
~1700-dim lattice (infeasible for BLASter). Those levels require remeasurement
of the high-confidence-but-wrong positions before the lattice step; do not run
this path for 384/512 expecting 100% recovery.
"""
import argparse,sys,time
from pathlib import Path
import numpy as np
BL=Path('/home/yu/workspace/BLASter/src')
sys.path.insert(0,str(BL))
sys.path.insert(0,'scripts')
from blaster import reduce as blaster_reduce
from evaluate_confidence import read_log, corrected_scores, recovery_prediction

def invert_mod(a,q=1024):
    a=a.copy()%q;n=len(a);inv=np.eye(n,dtype=np.int64)
    for j in range(n):
        odd=np.flatnonzero(a[j:,j]%2)
        if not len(odd):raise ValueError('singular modulo 2')
        p=j+odd[0];a[[j,p]]=a[[p,j]];inv[[j,p]]=inv[[p,j]]
        factor=pow(int(a[j,j]),-1,q);a[j]=a[j]*factor%q;inv[j]=inv[j]*factor%q
        col=a[:,j].copy();col[j]=0
        a=(a-col[:,None]*a[j])%q;inv=(inv-col[:,None]*inv[j])%q
    return inv

def independent_rows(a):
    basis={};chosen=[]
    for i,row in enumerate(a):
        bits=int.from_bytes(np.packbits(row%2,bitorder='little').tobytes(),'little')
        while bits:
            pivot=bits.bit_length()-1
            if pivot in basis:bits^=basis[pivot]
            else:
                basis[pivot]=bits;chosen.append(i);break
        if len(chosen)==a.shape[1]:return chosen
    raise ValueError('insufficient public rank')

def center(x,q=1024):return (x+q//2)%q-q//2

def build(A,b,guess,support,q=1024,extra_rows=400):
    support=np.sort(np.asarray(support,dtype=np.int64));k=len(support)
    r=center(b-A@guess);C=A[:,support]
    piv=independent_rows(C);inv=invert_mod(C[piv])
    extra=[i for i in range(len(A)) if i not in set(piv)][:extra_rows]
    rows=piv+extra;m=len(rows)
    T=(C[extra]@inv)%q
    basis=np.zeros((m+1,m+1),dtype=np.int64)
    basis[:k,:k]=np.eye(k,dtype=np.int64)
    basis[:k,k:m]=T.T
    basis[k:m,k:m]=q*np.eye(m-k,dtype=np.int64)
    basis[m,:m]=r[rows]%q;basis[m,m]=1
    return basis,piv,inv,r,m,k

def try_extract(red_cols,m,k,piv,inv,r,guess,support,A,b):
    support=np.sort(np.asarray(support,dtype=np.int64))
    for z in range(red_cols.shape[1]):
        vec=red_cols[:,z];sign=int(vec[m])
        if abs(sign)!=1:continue
        e=np.array([int(vec[t])*sign for t in range(m)],dtype=np.int64)
        if np.max(np.abs(e))>1:continue
        delta=center(inv@center(r[piv]-e[:k]))
        if np.max(np.abs(delta))>2:continue
        s=guess.copy();s[support]=center(guess[support]+delta)
        if np.max(np.abs(s))>1:continue
        if np.max(np.abs(center(b-A@s)))<=1:return s
    return None

def solve_column(A,b,guess,ranked_pos,cores,schedule,windows):
    """Adaptive: grow support window until the public equations are satisfiable."""
    for W,er,beta in windows:
        support=sorted(ranked_pos[:W])
        basis,piv,inv,r,m,k=build(A,b,guess,support,extra_rows=er)
        Bcols=basis.T.astype(np.int64)
        t=time.time()
        U,R,tp=blaster_reduce(Bcols.copy(),lll_size=64,beta=beta,bkz_prog=4,bkz_tours=2,cores=cores)
        s=try_extract(R,m,k,piv,inv,r,guess,support,A,b)
        print(f'    W={W} dim={m+1} beta={beta} {time.time()-t:.0f}s solved={s is not None}',flush=True)
        if s is not None:return s,W
    return None,None

def base_prediction(f):
    xy=np.column_stack([f['median_a_ns'],f['median_b_ns']])/f['gap_ns'][:,None]
    cen=np.stack([np.column_stack([f['a_'+t],f['b_'+t]])-f['ref_bucket'][:,None] for t in['neg','zero','pos']],axis=1)
    dd=((xy[:,None,:]-cen)**2).sum(2)
    return dd.argmin(1)-1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='directory for column-*.npy')
    p.add_argument('--cores',type=int,default=6)
    p.add_argument('--columns',type=str,default='',help='comma list; default all')
    p.add_argument('--prediction',choices=['auto','logged','nearest'],default='auto',
                   help='auto preserves marked votes; use logged for legacy vote CSVs')
    p.add_argument('--confidence-runs',type=Path,nargs='+',
                   help='raw captures for joint timing confidence (preserves voted guesses)')
    a=p.parse_args()
    with np.load(a.public,allow_pickle=False) as pub:
        A=pub['A'].astype(np.int64);B=pub['B'].astype(np.int64)
    n=A.shape[1];c=B.shape[1]
    f,_=read_log(str(a.log),include_truth=False)
    co=f['coord'].astype(int);po=f['position'].astype(int)
    pred=recovery_prediction(f,a.prediction)
    sel=np.zeros(len(co),bool);sel[np.random.default_rng(20261020).permutation(len(sel))[:len(sel)//2]]=True
    if a.confidence_runs:
        from joint_confidence import load_runs, joint_score
        score=joint_score(f,load_runs(a.confidence_runs,f))
        print('confidence: joint standardized timing evidence; no class prior',flush=True)
    else:
        score=corrected_scores(f,sel)['coord_standardized/prior_margin_8']
    a.out.mkdir(parents=True,exist_ok=True)
    cols=range(c) if not a.columns else [int(x) for x in a.columns.split(',')]
    # adaptive window ladder (W, extra_rows, beta). Grows to cover scattered errors.
    windows=[(60,400,40),(90,400,44),(119,400,44),(160,500,50),(220,500,56),(300,600,56),(384,650,58),(460,750,60),(520,800,64)]
    schedule=None
    ok_all=True
    for j in cols:
        ids=np.flatnonzero(co==j);pos=po[ids]
        guess=np.zeros(n,dtype=np.int64);guess[pos]=pred[ids]
        order=np.argsort(score[ids]);ranked=[int(pos[o]) for o in order]
        print(f'=== column {j} ===',flush=True)
        t0=time.time()
        s,W=solve_column(A,B[:,j],guess,ranked,a.cores,schedule,windows)
        if s is None:
            print(f'column {j}: FAILED (all windows)',flush=True);ok_all=False;continue
        np.save(a.out/f'column-{j}.npy',s.astype(np.int64))
        print(f'column {j}: solved at W={W} in {time.time()-t0:.0f}s',flush=True)
    print('ALL COLUMNS SOLVED' if ok_all else 'SOME COLUMNS FAILED',flush=True)
    sys.exit(0 if ok_all else 1)

if __name__=='__main__':main()
