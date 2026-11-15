/* Bernoulli mixture SPRT on timing signs. Not a bucket posterior. */
#include <math.h>
#include <string.h>
typedef struct { double low[7],high[7]; unsigned n; int side; } sprt_state;
static const double sprt_theta[7]={.51,.525,.55,.6,.7,.8,.9};
static void sprt_init(sprt_state *s){memset(s,0,sizeof(*s));s->side=-1;}
static void sprt_add(sprt_state *s,double value,double boundary){
    s->n++;
    if(value==boundary)return;
    for(int j=0;j<7;j++){
        double hit=log(2*sprt_theta[j]),miss=log(2*(1-sprt_theta[j]));
        s->high[j]+=value>boundary?hit:miss;
        s->low[j]+=value<boundary?hit:miss;
    }
}
static double sprt_logmix(const double *v){
    double m=v[0],sum=0;
    for(int j=1;j<7;j++)if(v[j]>m)m=v[j];
    for(int j=0;j<7;j++)sum+=exp(v[j]-m);
    return m+log(sum/7);
}
static int sprt_decide(sprt_state *s,double alpha,unsigned minimum){
    if(s->side>=0)return s->side;
    if(s->n<minimum)return -1;
    double low=sprt_logmix(s->low),high=sprt_logmix(s->high);
    if(fmax(low,high)>=log(2/alpha))s->side=high>low?1:0;
    return s->side;
}
