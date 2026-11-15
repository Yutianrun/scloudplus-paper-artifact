"""Snapshot the pinned collector; append a public-template SPRT entrypoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments/sprt-384-distinct10-a0001-20261101'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('original_build',SOURCE/'build.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original=(SOURCE/'collector.c').read_text()
    signature='int main(int argc,char**argv){'
    assert original.count(signature)==1
    generated=original.replace(signature,'int collector_unused_main(int argc,char**argv){')
    (HERE/'collector.c').write_text(generated+'\n'+(HERE/'template_main.c').read_text())
    shutil.copyfile(SOURCE/'sprt.h',HERE/'sprt.h')
    include=old.IMPL/'Optimized_Implementation/Scloudplus-384/kem'
    flags=['gcc','-O3','-std=c99','-mavx2','-mbmi2','-mpopcnt','-maes',
           '-DSCLOUDPLUS_FAMILY_AES','-DSCLOUDPLUS_TIER_OPTIMIZED',
           '-DSCLOUDPLUS_BACKEND_AVX2','-DSCLOUDPLUS_AVX2_FAMILY_AES']
    for path in [include,old.CORE/'include',old.CORE/'common',old.IMPL/'_shared/api_pkc']:
        flags+=['-I',str(path)]
    sources=['common/encode.c','common/hash_aes_shake.c','common/kem.c','common/pke.c',
             'common/util.c','avx2/aes_aesni.c','avx2/sample_avx2.c','avx2/matrix_avx2.c','ref/pack_reference.c']
    cmd=[*flags,str(HERE/'collector.c'),*[str(old.CORE/p) for p in sources],'-lm','-o',str(HERE/'collector')]
    subprocess.run(cmd,cwd=ROOT,check=True)
    paths=[SOURCE/'collector.c',SOURCE/'sprt.h',ROOT/'recover_demo.c',ROOT/'probe_validation.h',
           HERE/'build.py',HERE/'template_sprt.h',HERE/'template_main.c',HERE/'collector.c',HERE/'collector']
    (HERE/'build-manifest.json').write_text(json.dumps(dict(command=cmd,hashes={str(p):digest(p) for p in paths}),indent=2)+'\n')

if __name__=='__main__':main()
