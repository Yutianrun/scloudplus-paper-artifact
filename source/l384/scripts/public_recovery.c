/* Rebuild a functional decapsulation key ONLY from recovered S and public pk.
 * No key generation or original secret input. z is NOT recovered. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "hash.h"
#include "kem.h"
#include "pack.h"
#include "scloudplus_param_common.h"
int randombytes(unsigned char *buf, unsigned int size){
 FILE *f=fopen("/dev/urandom","rb");if(!f)return -1;
 int ok=fread(buf,1,size,f)==size;
 if(fclose(f))ok=0;return ok?0:-1;
}
#include "matrix.h"
#include "sample.c"
static int export_public(const char *pk_path,const char *out_path){
 uint8_t pk[scloudplus_pk];
 FILE *f=fopen(pk_path,"rb");if(!f)return 3;
 size_t count=fread(pk,1,sizeof(pk),f);int extra=fgetc(f);fclose(f);
 if(count!=sizeof(pk)||extra!=EOF)return 3;
 static uint16_t s[scloudplus_n*scloudplus_nbar],e[scloudplus_m*scloudplus_nbar];
 static uint16_t b[scloudplus_m*scloudplus_nbar],a[scloudplus_m*scloudplus_n];
 unpack_pk(pk,b);
 f=fopen(out_path,"wb");if(!f)return 4;
 int ok=fwrite(b,sizeof(uint16_t),scloudplus_m*scloudplus_nbar,f)==scloudplus_m*scloudplus_nbar;
 const uint8_t *seed=pk+scloudplus_pk-scloudplus_seedA_bytes;
 for(unsigned k=0;k<scloudplus_n;k+=scloudplus_nbar){
  memset(s,0,sizeof(s));
  for(unsigned j=0;j<scloudplus_nbar&&k+j<scloudplus_n;j++)s[j*scloudplus_n+k+j]=1;
  mul_as_e(seed,s,e,b);
  for(unsigned i=0;i<scloudplus_m;i++)for(unsigned j=0;j<scloudplus_nbar&&k+j<scloudplus_n;j++)
   a[i*scloudplus_n+k+j]=b[i*scloudplus_nbar+j]&scloudplus_q_mask;
 }
 ok &= fwrite(a,sizeof(uint16_t),scloudplus_m*scloudplus_n,f)==scloudplus_m*scloudplus_n;
 if(fclose(f)||!ok)return 5;
 return 0;
}
int main(int argc,char **argv){
 if(argc==4 && !strcmp(argv[1],"--export"))return export_public(argv[2],argv[3]);
 if(argc!=4)return 2;
 uint8_t pk[scloudplus_pk],rebuilt[scloudplus_kem_sk]={0};
 FILE *pf=fopen(argv[3],"rb");if(!pf)return 3;
 size_t pk_count=fread(pk,1,sizeof(pk),pf);int pk_extra=fgetc(pf);fclose(pf);
 if(pk_count!=sizeof(pk)||pk_extra!=EOF)return 3;
 uint16_t S[scloudplus_n*scloudplus_nbar],B[scloudplus_m*scloudplus_nbar],P[scloudplus_m*scloudplus_nbar],E[scloudplus_m*scloudplus_nbar]={0};
 FILE *f=fopen(argv[1],"rb");if(!f)return 4;
 size_t count=fread(S,sizeof(uint16_t),scloudplus_n*scloudplus_nbar,f);int extra=fgetc(f);fclose(f);
 if(count!=scloudplus_n*scloudplus_nbar||extra!=EOF)return 5;
 for(size_t i=0;i<count;i++)if(S[i]!=0&&S[i]!=1&&S[i]!=65535)return 6;
 unpack_pk(pk,B);mul_as_e(pk+scloudplus_pk-scloudplus_seedA_bytes,S,E,P);
 for(size_t i=0;i<scloudplus_m*scloudplus_nbar;i++){
  unsigned r=(B[i]-P[i])&scloudplus_q_mask;if(r!=0&&r!=1&&r!=scloudplus_q_mask)return 7;
 }
 pack_sk(S,rebuilt);memcpy(rebuilt+scloudplus_pke_sk,pk,sizeof(pk));
 scloudplus_H(rebuilt+scloudplus_pke_sk+scloudplus_pk,pk,sizeof(pk));
 /* Encapsulation obtains fresh bytes directly from the OS. */
 uint8_t ct[scloudplus_ctx],sender[scloudplus_ss],receiver[scloudplus_ss];
 if(scloud_kemencaps(pk,ct,sender)||scloud_kemdecaps(rebuilt,ct,receiver))return 9;
 if(memcmp(sender,receiver,sizeof(sender)))return 10;
 f=fopen(argv[2],"wb");if(!f)return 11;
 int ok=fwrite(sender,1,sizeof(sender),f)==sizeof(sender);
 ok &= fwrite(receiver,1,sizeof(receiver),f)==sizeof(receiver);
 if(fclose(f)||!ok)return 12;
 puts("ALL PUBLIC EQUATIONS PASS; FRESH KEM SHARED SECRETS MATCH (z not recovered)");return 0;
}
