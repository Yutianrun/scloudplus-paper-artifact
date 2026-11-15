/* Pairwise sign SPRT with boundaries fitted on PUBLIC candidate templates.
 * Include after sprt.h. alpha bounds the three pair tests, conditional on
 * correct sign-null/transfer assumptions; it is not a posterior probability.
 */
typedef struct {
    int left,right,axis,right_high,validated;
    double boundary,quality;
} template_pair;
static const int template_left[3]={0,0,1},template_right[3]={1,2,2};

static int template_winner(const template_pair pairs[3],sprt_state states[3],
                           double alpha,unsigned minimum){
    int wins[3]={0};
    for(int p=0;p<3;p++){
        if(!pairs[p].validated)continue;
        int side=sprt_decide(&states[p],alpha/3,minimum);
        if(side<0)continue;
        wins[side==pairs[p].right_high?pairs[p].right:pairs[p].left]++;
    }
    for(int t=0;t<3;t++)if(wins[t]==2)return t;
    return -1;
}

/* At cap always emit a trit. Rank by worst pairwise evidence against a rival;
 * preserve the complete scores and mark unvalidated/unresolved output forced. */
static int template_best(const template_pair pairs[3],const sprt_state states[3],
                         double scores[3],double *margin){
    for(int t=0;t<3;t++)scores[t]=INFINITY;
    for(int p=0;p<3;p++){
        double delta=sprt_logmix(states[p].high)-sprt_logmix(states[p].low);
        double right=pairs[p].right_high?delta:-delta;
        if(right<scores[pairs[p].right])scores[pairs[p].right]=right;
        if(-right<scores[pairs[p].left])scores[pairs[p].left]=-right;
    }
    int best=0;for(int t=1;t<3;t++)if(scores[t]>scores[best])best=t;
    double runner=-INFINITY;
    for(int t=0;t<3;t++)if(t!=best && scores[t]>runner)runner=scores[t];
    *margin=scores[best]-runner;return best;
}
