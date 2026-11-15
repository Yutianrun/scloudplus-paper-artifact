#!/usr/bin/env python3
"""Verify archived bytes and optionally extract to a new directory (Python 3.11+)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(level, destination=None):
    base=ROOT/'archives'/f'l{level}'
    cfg=json.loads((base/'manifest.json').read_text())['sets'][0]
    listing=base/cfg['file_manifest']
    if sha(listing)!=cfg['file_manifest_sha256']:raise ValueError('file manifest checksum mismatch')
    expected=json.loads(listing.read_text())
    parts=[]
    for rec in cfg['parts']:
        p=base/rec['path']
        if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError(f'bad part: {p.name}')
        parts.append(p)
    if destination is not None:destination.mkdir(parents=True,exist_ok=False)
    cat=subprocess.Popen(['cat',*map(str,parts)],stdout=subprocess.PIPE)
    dec=subprocess.Popen(['zstd','-d','-q','-c'],stdin=cat.stdout,stdout=subprocess.PIPE)
    cat.stdout.close()
    count=0;total=0;seen=set()
    try:
        with tarfile.open(fileobj=dec.stdout,mode='r|') as archive:
            for member in archive:
                if member.isdir():continue
                if not member.isfile():raise ValueError(f'non-file member: {member.name}')
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe archive path')
                if member.name not in expected or member.name in seen:raise ValueError('unexpected or duplicate member')
                record=expected[member.name];h=hashlib.sha256();size=0
                target=None
                if destination is not None:
                    dest=destination/Path(*path.parts);dest.parent.mkdir(parents=True,exist_ok=True)
                    target=dest.open('xb')
                try:
                    with archive.extractfile(member) as stream:
                        while data:=stream.read(1024*1024):
                            h.update(data);size+=len(data)
                            if target:target.write(data)
                finally:
                    if target:target.close()
                if size!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError(f'bad member: {member.name}')
                if destination is not None:dest.chmod(member.mode & 0o777)
                seen.add(member.name);count+=1;total+=size
        # Consume archive padding and let zstd validate the frame checksum.
        while dec.stdout.read(1024*1024):pass
        if dec.wait()!=0 or cat.wait()!=0:raise ValueError('decompression failed')
    finally:
        dec.stdout.close()
        if dec.poll() is None:dec.terminate();dec.wait()
        if cat.poll() is None:cat.terminate();cat.wait()
    if seen!=set(expected):raise ValueError('missing archive members')
    result={'level':level,'parts_verified':len(parts),'files_verified':count,'bytes_verified':total,'all_sha256_match':True}
    print(json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',choices=['256','384','512','all'],default='all')
    p.add_argument('--extract',type=Path,help='New output root; one subdirectory per level')
    p.add_argument('--report',type=Path)
    a=p.parse_args();levels=[256,384,512] if a.level=='all' else [int(a.level)]
    results=[verify(n,a.extract/f'l{n}' if a.extract else None) for n in levels]
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
