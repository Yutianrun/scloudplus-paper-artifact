#!/usr/bin/env python3
"""Validate recovered columns against public equations, a fresh KEM and AES-GCM.
No original secret matrix or timing correctness labels are read.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def validate_columns(A, B, directory):
    columns = []
    for j in range(B.shape[1]):
        s = np.load(directory / f'column-{j}.npy', allow_pickle=False)
        if s.shape != (A.shape[1],) or not np.isin(s, [-1, 0, 1]).all():
            raise ValueError(f'column {j}: invalid ternary secret')
        columns.append(s.astype(np.int64))
    S = np.column_stack(columns)
    residual = (B.astype(np.int64) - A.astype(np.int64) @ S + 512) % 1024 - 512
    if np.any(np.abs(residual) > 1):
        raise ValueError('public equations failed')
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--pk', type=Path, required=True)
    p.add_argument('--columns', type=Path, required=True)
    p.add_argument('--verifier', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--level', type=int, choices=(256,384,512), default=256)
    a = p.parse_args()
    with np.load(a.public, allow_pickle=False) as public:
        S = validate_columns(public['A'], public['B'], a.columns)
        equations = public['B'].size
    with tempfile.TemporaryDirectory() as temporary:
        d = Path(temporary)
        secret = d / 'recovered-S.bin'
        S.T.astype('<u2').tofile(secret)
        secrets = d / 'shared-secrets.bin'
        subprocess.run([str(a.verifier.resolve()), str(secret), str(secrets), str(a.pk.resolve())], check=True)
        shared = secrets.read_bytes()
        size = a.level // 8
        if len(shared) != 2*size or shared[:size] != shared[size:]:
            raise ValueError('fresh KEM shared secrets do not match')
        sender = hashlib.sha256(b'Scloudplus recovery AEAD check\0' + shared[:size]).digest()
        receiver = hashlib.sha256(b'Scloudplus recovery AEAD check\0' + shared[size:]).digest()
        nonce = os.urandom(12)
        aad = f'Scloudplus-{a.level} independent public recovery validation'.encode()
        message = b'Fresh ciphertext decrypted using only recovered S and the public key.'
        ciphertext = AESGCM(sender).encrypt(nonce, message, aad)
        plaintext = AESGCM(receiver).decrypt(nonce, ciphertext, aad)
        if plaintext != message:
            raise ValueError('authenticated message mismatch')
        damaged = ciphertext[:-1] + bytes([ciphertext[-1] ^ 1])
        try:
            AESGCM(receiver).decrypt(nonce, damaged, aad)
        except InvalidTag:
            pass
        else:
            raise ValueError('tampered authenticated ciphertext accepted')
        bad_S = S.copy()
        bad_S[0, 0] = 1 if bad_S[0, 0] != 1 else 0
        bad_S.T.astype('<u2').tofile(secret)
        negative = subprocess.run([str(a.verifier.resolve()), str(secret), str(d/'bad.bin'), str(a.pk.resolve())], capture_output=True)
        if negative.returncode != 7:
            raise ValueError(f'corrupted S did not fail public equations: {negative.returncode}')
    result = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  level=a.level,
                  recovered_secret_sha256=hashlib.sha256(S.T.astype('<u2').tobytes()).hexdigest(),
                  public_equations_passed=int(equations), coefficients=int(S.size),
                  fresh_kem_shared_secret_match=True, kem_random_source="OS /dev/urandom",
                  authenticated_message_decrypted=True,
                  tampered_aead_rejected=True, corrupted_secret_rejected=True,
                  rejection_secret_z_recovered=False, public_key_sha256=hashlib.sha256(a.pk.read_bytes()).hexdigest(),
                  nonce_hex=nonce.hex(), ciphertext_hex=ciphertext.hex(), plaintext=plaintext.decode())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.columns/'recovered-S.npy', S)
    S.T.astype('<u2').tofile(a.columns/'recovered-S.bin')
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
