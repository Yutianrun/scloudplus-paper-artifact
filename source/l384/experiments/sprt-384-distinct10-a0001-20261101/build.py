"""Build fresh 384 tools; pin sources and create ten distinct PUBLIC fixtures."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMPL=ROOT/'latest-download/Implementations and Test_Vectors/Implementations'
CORE=IMPL/'_shared/scloudplus_core'
SEEDS=list(range(101,121,2))

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    start=time.monotonic()
    build=HERE/'build';build.mkdir(exist_ok=False)
    fixtures=HERE/'fixtures';fixtures.mkdir(exist_ok=False)
    include=IMPL/'Optimized_Implementation/Scloudplus-384/kem'
    flags=['gcc','-O3','-std=c99','-mavx2','-mbmi2','-mpopcnt','-maes',
           '-DSCLOUDPLUS_FAMILY_AES','-DSCLOUDPLUS_TIER_OPTIMIZED',
           '-DSCLOUDPLUS_BACKEND_AVX2','-DSCLOUDPLUS_AVX2_FAMILY_AES']
    for path in [include,CORE/'include',CORE/'common',IMPL/'_shared/api_pkc']:
        flags+=['-I',str(path)]
    sources=['common/encode.c','common/hash_aes_shake.c','common/kem.c','common/pke.c',
             'common/util.c','avx2/aes_aesni.c','avx2/sample_avx2.c','avx2/matrix_avx2.c','ref/pack_reference.c']
    for source,name in [(HERE/'collector.c','collector'),(HERE/'key_fixture.c','key-fixture'),
                        (ROOT/'scripts/public_recovery.c','public-recovery')]:
        subprocess.run([*flags,str(source),*[str(CORE/p) for p in sources],'-lm','-o',str(build/name)],cwd=ROOT,check=True)
    groups=[]
    for group,seed in enumerate(SEEDS,1):
        fixture=fixtures/f'group-{group:02d}';fixture.mkdir()
        subprocess.run([str(build/'key-fixture'),str(seed),str(fixture/'public.pk')],check=True)
        subprocess.run([str(build/'public-recovery'),'--export',str(fixture/'public.pk'),str(fixture/'public.bin')],check=True)
        values=np.fromfile(fixture/'public.bin',dtype=np.uint16);n=1664;nbar=16
        if len(values)!=n*n+n*nbar:raise ValueError('bad export size')
        np.savez(fixture/'public.npz',B=values[:n*nbar].reshape(n,nbar),A=values[n*nbar:].reshape(n,n))
        (fixture/'pipeline.json').write_text(json.dumps({'level':384,'args':[seed,n,1000,3000,2000000,nbar,6,0]})+'\n')
        (fixture/'public-recovery').symlink_to(build/'public-recovery')
        groups.append(dict(group=group,seed=seed,fixture=str(fixture),public_key_sha256=digest(fixture/'public.pk'),status='pending'))
    if len({g['public_key_sha256'] for g in groups})!=10:raise ValueError('duplicate test key')
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.c'))+[HERE/'sprt.h',HERE/'PROTOCOL.md',ROOT/'recover_demo.c',ROOT/'probe_validation.h']
    paths+=list(CORE.rglob('*.c'))+list(CORE.rglob('*.h'))+list(include.rglob('*.h'))
    paths+=[p for p in (HERE/'blaster-fixed').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths+=list(build.glob('*'))
    paths+=[p for p in fixtures.rglob('*') if p.is_file()]
    paths+=[ROOT/'scripts'/f for f in ['public_recovery.c','sprt_soft_decision.py','sequential_oracle.py',
             'sparse_correction_lattice.py','postprocess_recover.py','evaluate_confidence.py','verify_public_recovery.py']]
    manifest=dict(level=384,alpha=.001,groups=groups,unique_public_keys=10,
                  build_and_public_fixture_seconds=time.monotonic()-start,
                  hashes={str(p):digest(p) for p in sorted(set(paths))})
    (build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'unique_public_keys':10,'seeds':SEEDS,'seconds':manifest['build_and_public_fixture_seconds']}))

if __name__=='__main__':main()
