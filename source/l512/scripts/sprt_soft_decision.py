"""Ternary best guesses and ranking evidence, including unresolved SPRT axes.

Mixture log evidence is a ranking score, not a posterior probability. Reference
rejection and an unresolved axis lower priority for freezing, never erase data.
"""
import numpy as np
from sequential_oracle import evidence


def soft_decision(samples, candidate_buckets, gap, origin, accepted_guess=99,
                  reference_valid=True):
    table=np.asarray(candidate_buckets,dtype=int)
    if table.shape!=(3,2) or not np.isfinite(gap) or gap<=0:
        raise ValueError('three two-axis candidates and positive gap required')
    logs=np.zeros((2,2))
    candidate_scores=np.zeros(3)
    for axis in range(2):
        lo,hi=int(table[:,axis].min()),int(table[:,axis].max())
        if lo==hi:continue
        if not np.isin(table[:,axis],[lo,hi]).all() or len(samples[axis])==0:
            raise ValueError('nonconstant binary axis requires samples')
        boundary=((lo+hi)/2-origin)*gap
        logs[axis]=evidence(samples[axis],boundary)[-1]
        candidate_scores+=logs[axis,(table[:,axis]==hi).astype(int)]
    best=int(np.argmax(candidate_scores))
    if accepted_guess in (-1,0,1):
        best=accepted_guess+1
    rivals=np.delete(candidate_scores,best)
    margin=float(candidate_scores[best]-rivals.max())
    trusted=bool(accepted_guess in (-1,0,1) and reference_valid)
    return dict(guess=best-1,margin=margin,trusted=trusted,
                candidate_scores=candidate_scores,axis_loge=logs)
