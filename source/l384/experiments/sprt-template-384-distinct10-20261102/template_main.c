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
