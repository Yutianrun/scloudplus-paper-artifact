"""Generate an isolated live-SPRT collector from the current acquisition code."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    source = (ROOT/'recover_e2e_par.c').read_text()
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'expected one source anchor: {old[:100]}')
        source = source.replace(old, new)
    replace('#include "recover_demo.c"', '#include "../../recover_demo.c"')
    replace('#include "probe_validation.h"', '#include "../../probe_validation.h"')
    replace('static volatile uint8_t sink;', 'static unsigned long long sprt_calls=0;\nstatic volatile uint8_t sink;')
    replace('uint64_t a=use_tsc?read_tsc():mono_ns(); scloud_kemdecaps',
            'sprt_calls++; uint64_t a=use_tsc?read_tsc():mono_ns(); scloud_kemdecaps')
    replace('/* work item + shared result slot */', r'''
#include "sprt.h"
static double sequential_alpha=.001;
static unsigned sequential_min=32,sequential_batch=16;
static int sequential_axis(const size_t *b,size_t *lo,size_t *hi){
    *lo=*hi=b[0];
    for(int j=1;j<3;j++){if(b[j]<*lo)*lo=b[j];if(b[j]>*hi)*hi=b[j];}
    for(int j=0;j<3;j++)if(b[j]!=*lo && b[j]!=*hi)return 0;
    return 1;
}
static void sequential_measure(const uint8_t *sk,const uint8_t *a,const uint8_t *b,
                              const uint8_t *ref,const probe_t *p,double gap,unsigned cap,
                              double *da,double *db,sprt_state state[2],int buckets[2],
                              FILE *raw,unsigned coord,unsigned pos){
    const uint8_t *ct[2]={a,b};double *samples[2]={da,db};
    const size_t *table[2]={p->pb0,p->pb1};
    size_t lo[2],hi[2];double boundary[2];
    for(int axis=0;axis<2;axis++){
        sprt_init(&state[axis]);buckets[axis]=-1;
        if(!sequential_axis(table[axis],&lo[axis],&hi[axis]))_exit(7);
        boundary[axis]=(.5*(lo[axis]+hi[axis])-B_LO)*gap;
        if(lo[axis]==hi[axis])buckets[axis]=(int)lo[axis];
    }
    for(unsigned base=0;base<cap;base+=sequential_batch){
        unsigned stop=base+sequential_batch;if(stop>cap)stop=cap;
        for(unsigned j=base;j<stop;j++){
            unsigned first=xr()&1;
            for(unsigned t=0;t<2;t++){
                unsigned axis=(first+t)%2;
                if(buckets[axis]>=0)continue;
                double x=time_difference(sk,ct[axis],ref);
                samples[axis][state[axis].n]=x;
                sprt_add(&state[axis],x,boundary[axis]);
                fprintf(raw,"%u,%u,%u,%u,%.17g\n",coord,pos,axis,state[axis].n,x);
            }
        }
        for(int axis=0;axis<2;axis++)if(buckets[axis]<0){
            int side=sprt_decide(&state[axis],sequential_alpha,sequential_min);
            if(side>=0)buckets[axis]=(int)(side?hi[axis]:lo[axis]);
        }
        if(buckets[0]>=0 && buckets[1]>=0)break;
    }
}
/* work item + shared result slot */''')
    replace('    unsigned char done;','    unsigned na,nbshots; int ref_ok; double local_ref,own_gap,loge_a,loge_b;\n    unsigned char done;')
    replace('    if(argc>1) krng=', r'''
    if(getenv("SPRT_ALPHA"))sequential_alpha=strtod(getenv("SPRT_ALPHA"),NULL);
    if(!(sequential_alpha>0 && sequential_alpha<1) || !getenv("SPRT_OUT"))return 2;
    if(getenv("ORACLE") && atoi(getenv("ORACLE")))return 2;
    if(argc>1) krng=''')
    replace('    static uint16_t S[(size_t)scloudplus_n*scloudplus_nbar]; unpack_sk(sk,S);',
            '    /* Truth and exact bucket diagnostics are excluded from this collector. */')
    replace('            for(size_t it=g; it<nit; it+=nworkers){', r'''
            char rawpath[1024];snprintf(rawpath,sizeof(rawpath),"%s/raw-%u.csv",getenv("SPRT_OUT"),g);
            FILE *raw=fopen(rawpath,"wx");if(!raw)_exit(6);
            unsigned processed=0;
            for(size_t it=g; it<nit; it+=nworkers){
                if(processed && processed%128==0){
                    gap=calibrate(sk,ct_lo,ct_hi,Ncal,g)/(B_HI-B_LO);
                    if(!isfinite(gap))_exit(5);
                }''')
    start = source.index('                measure_axes(sk,ctb,cts,ct_lo,db,ds,N,interleave);')
    end = source.index('            free(db);free(ds);free(scratch);', start)
    source = source[:start]+r'''
                sprt_state state[2];int buckets[2];
                sequential_measure(sk,ctb,cts,ct_lo,p,gap,N,db,ds,state,buckets,raw,i,k);
                double mb=0,ms=0;
                if(state[0].n)mb=summarize_samples(db,state[0].n,scratch,&res[it].mad_a,&res[it].half_delta_a);
                if(state[1].n)ms=summarize_samples(ds,state[1].n,scratch,&res[it].mad_b,&res[it].half_delta_b);
                double refs[32];for(unsigned j=0;j<32;j++)refs[j]=time_difference(sk,ct_hi,ct_lo);
                double local=med(refs,32)/(B_HI-B_LO);
                int ref_ok=reference_valid(local,gap),guess=99;
                if(ref_ok && buckets[0]>=0 && buckets[1]>=0)
                    for(int t=0;t<3;t++)if(p->pb0[t]==(size_t)buckets[0] && p->pb1[t]==(size_t)buckets[1])guess=t-1;
                res[it].guess=guess;res[it].true_v=99;res[it].nb=res[it].ns=255;
                res[it].tb=buckets[0]<0?255:buckets[0];res[it].ts=buckets[1]<0?255:buckets[1];
                res[it].ma=mb;res[it].mb=ms;res[it].own_gap=gap;res[it].local_ref=local;res[it].ref_ok=ref_ok;
                res[it].na=state[0].n;res[it].nbshots=state[1].n;
                res[it].loge_a=fmax(sprt_logmix(state[0].low),sprt_logmix(state[0].high));
                res[it].loge_b=fmax(sprt_logmix(state[1].low),sprt_logmix(state[1].high));
                res[it].done=1;processed++;
                if(processed%64==0){
                    fflush(raw);
                    fprintf(stderr,"SPRT_PROGRESS worker=%u coefficients=%u calls=%llu seconds=%.3f\n",g,processed,sprt_calls,(mono_ns()-w0)/1e9);
                }
            }
            fclose(raw);
            fprintf(stderr,"SPRT_DONE worker=%u coefficients=%u calls=%llu seconds=%.3f\n",g,processed,sprt_calls,(mono_ns()-w0)/1e9);
''' + source[end:]
    replace('half_delta_b_ns\\n");','half_delta_b_ns,n_a,n_b,ref_ok,local_ref_ns,loge_a,loge_b\\n");')
    replace('w,res[it].ma,res[it].mb,wgap[w],(size_t)B_LO,','w,res[it].ma,res[it].mb,res[it].own_gap,(size_t)B_LO,')
    replace('printf(",%.17g,%.17g,%.17g,%.17g\\n",res[it].mad_a,res[it].mad_b,\n                     res[it].half_delta_a,res[it].half_delta_b);',
            'printf(",%.17g,%.17g,%.17g,%.17g,%u,%u,%d,%.17g,%.17g,%.17g\\n",res[it].mad_a,res[it].mad_b,\n                     res[it].half_delta_a,res[it].half_delta_b,res[it].na,res[it].nbshots,res[it].ref_ok,res[it].local_ref,res[it].loge_a,res[it].loge_b);')
    # Remove inherited truth accuracy and fixed-N cost claims.
    start=source.index('    /* Two axes, each with N (probe,reference) pairs, plus two diagnostics.')
    end=source.index('    free(items); free(memo);',start)
    source=source[:start]+r'''
    fprintf(stderr,"SPRT_FINISHED coefficients=%lu undone=%lu cap=%u alpha=%.17g workers=%u wall=%.6f\n",tot,undone,N,sequential_alpha,nworkers,wall);
''' + source[end:]
    (HERE/'collector.c').write_text(source)
    subprocess.run(['bash','experiments/public-recovery-20261020/build_verifier.sh',
                    str(HERE/'collector.c'),str(HERE/'collector')],cwd=ROOT,check=True)
    paths=[ROOT/'recover_e2e_par.c',ROOT/'recover_demo.c',ROOT/'probe_validation.h',
           HERE/'sprt.h',Path(__file__),HERE/'collector.c',HERE/'collector']
    (HERE/'build.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in paths},indent=2)+'\n')


if __name__=='__main__':main()
