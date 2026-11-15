/* Test-only deterministic keys. Truth export is invoked AFTER public recovery. */
#define main recover_demo_unused_main
#include "../../recover_demo.c"
#undef main
int main(int argc,char **argv){
    if(argc!=3 && argc!=4)return 2;
    krng=strtoull(argv[1],0,10)|1ULL;
    uint8_t pk[scloudplus_pk],sk[scloudplus_kem_sk];
    if(scloud_kemkeygen(pk,sk))return 3;
    FILE *f=fopen(argv[2],"wb");if(!f)return 4;
    int ok=fwrite(pk,1,sizeof(pk),f)==sizeof(pk);
    if(fclose(f)||!ok)return 5;
    if(argc==4){
        static uint16_t S[scloudplus_n*scloudplus_nbar];
        unpack_sk(sk,S);
        f=fopen(argv[3],"wb");if(!f)return 6;
        ok=fwrite(S,sizeof(uint16_t),scloudplus_n*scloudplus_nbar,f)==scloudplus_n*scloudplus_nbar;
        if(fclose(f)||!ok)return 7;
    }
    return 0;
}
