#!/usr/bin/env bash
set -euo pipefail
IMPL="latest-download/Implementations and Test_Vectors/Implementations"
CORE="$IMPL/_shared/scloudplus_core"
LEVEL=${1:?level required}
case "$LEVEL" in 256|384|512) ;; *) exit 2;; esac
gcc -O3 -std=c99 -mavx2 -mbmi2 -mpopcnt -maes \
  -DSCLOUDPLUS_FAMILY_AES \
  -DSCLOUDPLUS_TIER_OPTIMIZED \
  -DSCLOUDPLUS_BACKEND_AVX2 \
  -DSCLOUDPLUS_AVX2_FAMILY_AES \
  -I"$IMPL/Optimized_Implementation/Scloudplus-$LEVEL/kem" \
  -I"$CORE/include" \
  -I"$CORE/common" \
  -I"$IMPL/_shared/api_pkc" \
  scripts/public_recovery.c \
  "$CORE/common/encode.c" \
  "$CORE/common/hash_aes_shake.c" \
  "$CORE/common/kem.c" \
  "$CORE/common/pke.c" \
  "$CORE/common/util.c" \
  "$CORE/avx2/aes_aesni.c" \
  "$CORE/avx2/sample_avx2.c" \
  "$CORE/avx2/matrix_avx2.c" \
  "$CORE/ref/pack_reference.c" \
  -lm -o "${2:?output binary required}"
