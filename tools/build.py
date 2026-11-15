#!/usr/bin/env python3
"""Build original frozen collectors and public verifiers; does not start capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
COLLECTORS={256:'sprt-256-20261029',384:'sprt-template-384-distinct10-20261102',512:'sprt-wide-alpha-0.01-20261030'}

def build(level,out):
    snap=ROOT/'source'/f'l{level}'
    manifest=json.loads((ROOT/'provenance'/f'l{level}-source.json').read_text())
    for name,record in manifest['files'].items():
        if hashlib.sha256((snap/name).read_bytes()).hexdigest()!=record['sha256']:raise ValueError(f'source changed: {name}')
    impl=snap/'latest-download/Implementations and Test_Vectors/Implementations'
    core=impl/'_shared/scloudplus_core'
    flags=['gcc','-O3','-std=c99','-mavx2','-mbmi2','-mpopcnt','-maes',
           '-DSCLOUDPLUS_FAMILY_AES','-DSCLOUDPLUS_TIER_OPTIMIZED',
           '-DSCLOUDPLUS_BACKEND_AVX2','-DSCLOUDPLUS_AVX2_FAMILY_AES']
    for p in [impl/f'Optimized_Implementation/Scloudplus-{level}/kem',core/'include',core/'common',impl/'_shared/api_pkc']:
        flags+=['-I',str(p)]
    common=[core/p for p in ['common/encode.c','common/hash_aes_shake.c','common/kem.c','common/pke.c','common/util.c','avx2/aes_aesni.c','avx2/sample_avx2.c','avx2/matrix_avx2.c','ref/pack_reference.c']]
    dest=out/f'l{level}';dest.mkdir(parents=True,exist_ok=False)
    commands=[]
    for source,name in [(snap/'experiments'/COLLECTORS[level]/'collector.c','collector'),(snap/'scripts/public_recovery.c','public-recovery')]:
        command=[*flags,str(source),*map(str,common),'-lm','-o',str(dest/name)]
        with (dest/f'{name}.build.log').open('w') as log:
            subprocess.run(command,check=True,cwd=snap,stdout=log,stderr=subprocess.STDOUT)
        commands.append(command)
    (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(f'L{level}: original collector and public verifier built',flush=True)
    return {'level':level,'source_commit':manifest['commit'],'source_hashes_verified':len(manifest['files']),'collector_build':'passed','public_verifier_build':'passed'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--level',choices=['256','384','512','all'],default='all')
    p.add_argument('--out',type=Path,default=ROOT/'.work/build');p.add_argument('--report',type=Path)
    a=p.parse_args();levels=[256,384,512] if a.level=='all' else [int(a.level)]
    results=[build(n,a.out) for n in levels]
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
