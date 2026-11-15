"""Build isolated SPRT collectors and public verifiers for 384 and 512."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMPL=ROOT/'latest-download/Implementations and Test_Vectors/Implementations'
CORE=IMPL/'_shared/scloudplus_core'
FIXTURES={384:ROOT/'experiments/vote3-384-full-20261022',
          512:ROOT/'experiments/vote3-512-full-20261024'}
SOURCES=['common/encode.c','common/hash_aes_shake.c','common/kem.c','common/pke.c',
         'common/util.c','avx2/aes_aesni.c','avx2/sample_avx2.c','avx2/matrix_avx2.c','ref/pack_reference.c']


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    start=time.monotonic()
    build=HERE/'build';build.mkdir(exist_ok=False)
    fixtures=HERE/'fixtures';fixtures.mkdir(exist_ok=False)
    paths=[HERE/'collector.c',HERE/'sprt.h',Path(__file__),ROOT/'recover_demo.c',ROOT/'probe_validation.h',
           ROOT/'scripts/public_recovery.c']
    paths+=list(CORE.rglob('*.c'))+list(CORE.rglob('*.h'))
    groups=[]
    for level,n in [(384,1664),(512,2400)]:
        include=IMPL/f'Optimized_Implementation/Scloudplus-{level}/kem'
        paths+=list(include.rglob('*.h'))
        flags=['gcc','-O3','-std=c99','-mavx2','-mbmi2','-mpopcnt','-maes',
               '-DSCLOUDPLUS_FAMILY_AES','-DSCLOUDPLUS_TIER_OPTIMIZED',
               '-DSCLOUDPLUS_BACKEND_AVX2','-DSCLOUDPLUS_AVX2_FAMILY_AES']
        for path in [include,CORE/'include',CORE/'common',IMPL/'_shared/api_pkc']:
            flags+=['-I',str(path)]
        fixture=fixtures/str(level);fixture.mkdir()
        for source,dest in [(HERE/'collector.c',build/f'collector-{level}'),
                            (ROOT/'scripts/public_recovery.c',fixture/'public-recovery')]:
            subprocess.run([*flags,str(source),*[str(CORE/p) for p in SOURCES],'-lm','-o',str(dest)],check=True,cwd=ROOT)
            paths.append(dest)
        original=FIXTURES[level]
        shutil.copyfile(original/'probes.cache',fixture/'probes.cache')
        shutil.copyfile(original/('public.pk' if level==384 else 'run1.pk'),fixture/'public.pk')
        (fixture/'pipeline.json').write_text(json.dumps(dict(level=level,args=[1,n,1000,3000,2000000,16,6,0]))+'\n')
        subprocess.run([str(fixture/'public-recovery'),'--export',str(fixture/'public.pk'),str(fixture/'public.bin')],check=True)
        values=np.fromfile(fixture/'public.bin',dtype=np.uint16)
        if len(values)!=n*n+n*16:raise ValueError('wrong public export shape')
        np.savez(fixture/'public.npz',B=values[:n*16].reshape(n,16),A=values[n*16:].reshape(n,n))
        # Public cache/key checks only: SEARCH_ONLY returns before calibration or measurements.
        env={k:v for k,v in os.environ.items() if not k.startswith(('TIMING_','SPRT_')) and
             k not in ('PROBE_VALIDATE','PROBE_VALIDATE_TOL','RETEST_CSV','PUBLIC_KEY_OUT')}
        env.update(SPRT_ALPHA='.01',SPRT_OUT=str(build),ORACLE='0',SEARCH_ONLY='1',PROFILE_ONLY='0',
                   PROBE_CACHE=str(fixture/'probes.cache'),PUBLIC_KEY_OUT=str(build/f'checked-{level}.pk'))
        with (build/f'cache-check-{level}.log').open('w') as log:
            subprocess.run([str(build/f'collector-{level}'),'1',str(n),'1000','3000','2000000','16','6','0'],
                           env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if (build/f'checked-{level}.pk').read_bytes()!=(fixture/'public.pk').read_bytes():
            raise ValueError('fixture key does not match seed 1')
        paths += [fixture/f for f in ['public.pk','public.npz','probes.cache','pipeline.json']]
        groups.append(dict(level=level,n=n,nbar=16,seed=1,fixture=str(fixture),
                           audit_log=str(original/f'{level}-run1.log'),binary=str(build/f'collector-{level}')))
    manifest=dict(groups=groups,hashes={str(p):digest(p) for p in sorted(set(paths))},
                  build_and_public_preflight_seconds=time.monotonic()-start)
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(build_seconds=manifest['build_and_public_preflight_seconds'],levels=[384,512],
                         pinned_files=len(manifest['hashes'])),indent=2))


if __name__=='__main__':main()
