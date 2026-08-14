#define GPU_ATTEMPT_SIZE 16
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

/* Keccak state in scalar registers (not ulong[25]) — avoids private-memory
 * spill that made OpenCL SHA-3 slower than multi-CPU on Intel Arc. */

__constant ulong k_rc[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
};

#define KECCAK_ROUND(rc) \
    do { \
        const ulong c0 = a00 ^ a10 ^ a20 ^ a30 ^ a40; \
        const ulong c1 = a01 ^ a11 ^ a21 ^ a31 ^ a41; \
        const ulong c2 = a02 ^ a12 ^ a22 ^ a32 ^ a42; \
        const ulong c3 = a03 ^ a13 ^ a23 ^ a33 ^ a43; \
        const ulong c4 = a04 ^ a14 ^ a24 ^ a34 ^ a44; \
        const ulong d0 = ROTL64(c1, 1) ^ c4; \
        const ulong d1 = ROTL64(c2, 1) ^ c0; \
        const ulong d2 = ROTL64(c3, 1) ^ c1; \
        const ulong d3 = ROTL64(c4, 1) ^ c2; \
        const ulong d4 = ROTL64(c0, 1) ^ c3; \
        const ulong b00 = (a00 ^ d0); \
        const ulong b01 = ROTL64(a11 ^ d1, 44); \
        const ulong b02 = ROTL64(a22 ^ d2, 43); \
        const ulong b03 = ROTL64(a33 ^ d3, 21); \
        const ulong b04 = ROTL64(a44 ^ d4, 14); \
        const ulong b10 = ROTL64(a03 ^ d3, 28); \
        const ulong b11 = ROTL64(a14 ^ d4, 20); \
        const ulong b12 = ROTL64(a20 ^ d0, 3); \
        const ulong b13 = ROTL64(a31 ^ d1, 45); \
        const ulong b14 = ROTL64(a42 ^ d2, 61); \
        const ulong b20 = ROTL64(a01 ^ d1, 1); \
        const ulong b21 = ROTL64(a12 ^ d2, 6); \
        const ulong b22 = ROTL64(a23 ^ d3, 25); \
        const ulong b23 = ROTL64(a34 ^ d4, 8); \
        const ulong b24 = ROTL64(a40 ^ d0, 18); \
        const ulong b30 = ROTL64(a04 ^ d4, 27); \
        const ulong b31 = ROTL64(a10 ^ d0, 36); \
        const ulong b32 = ROTL64(a21 ^ d1, 10); \
        const ulong b33 = ROTL64(a32 ^ d2, 15); \
        const ulong b34 = ROTL64(a43 ^ d3, 56); \
        const ulong b40 = ROTL64(a02 ^ d2, 62); \
        const ulong b41 = ROTL64(a13 ^ d3, 55); \
        const ulong b42 = ROTL64(a24 ^ d4, 39); \
        const ulong b43 = ROTL64(a30 ^ d0, 41); \
        const ulong b44 = ROTL64(a41 ^ d1, 2); \
        a00 = b00 ^ ((~b01) & b02) ^ (rc); \
        a01 = b01 ^ ((~b02) & b03); \
        a02 = b02 ^ ((~b03) & b04); \
        a03 = b03 ^ ((~b04) & b00); \
        a04 = b04 ^ ((~b00) & b01); \
        a10 = b10 ^ ((~b11) & b12); \
        a11 = b11 ^ ((~b12) & b13); \
        a12 = b12 ^ ((~b13) & b14); \
        a13 = b13 ^ ((~b14) & b10); \
        a14 = b14 ^ ((~b10) & b11); \
        a20 = b20 ^ ((~b21) & b22); \
        a21 = b21 ^ ((~b22) & b23); \
        a22 = b22 ^ ((~b23) & b24); \
        a23 = b23 ^ ((~b24) & b20); \
        a24 = b24 ^ ((~b20) & b21); \
        a30 = b30 ^ ((~b31) & b32); \
        a31 = b31 ^ ((~b32) & b33); \
        a32 = b32 ^ ((~b33) & b34); \
        a33 = b33 ^ ((~b34) & b30); \
        a34 = b34 ^ ((~b30) & b31); \
        a40 = b40 ^ ((~b41) & b42); \
        a41 = b41 ^ ((~b42) & b43); \
        a42 = b42 ^ ((~b43) & b44); \
        a43 = b43 ^ ((~b44) & b40); \
        a44 = b44 ^ ((~b40) & b41); \
    } while (0)

#define KECCAK_F1600() \
    do { \
        KECCAK_ROUND(k_rc[0]); KECCAK_ROUND(k_rc[1]); KECCAK_ROUND(k_rc[2]); KECCAK_ROUND(k_rc[3]); \
        KECCAK_ROUND(k_rc[4]); KECCAK_ROUND(k_rc[5]); KECCAK_ROUND(k_rc[6]); KECCAK_ROUND(k_rc[7]); \
        KECCAK_ROUND(k_rc[8]); KECCAK_ROUND(k_rc[9]); KECCAK_ROUND(k_rc[10]); KECCAK_ROUND(k_rc[11]); \
        KECCAK_ROUND(k_rc[12]); KECCAK_ROUND(k_rc[13]); KECCAK_ROUND(k_rc[14]); KECCAK_ROUND(k_rc[15]); \
        KECCAK_ROUND(k_rc[16]); KECCAK_ROUND(k_rc[17]); KECCAK_ROUND(k_rc[18]); KECCAK_ROUND(k_rc[19]); \
        KECCAK_ROUND(k_rc[20]); KECCAK_ROUND(k_rc[21]); KECCAK_ROUND(k_rc[22]); KECCAK_ROUND(k_rc[23]); \
    } while (0)

static int prsha3_digest_eq(__global const uchar* k_hash, uint hash_len,
    ulong a00, ulong a01, ulong a02, ulong a03, ulong a04,
    ulong a10, ulong a11, ulong a12) {
    ulong lanes[8];
    lanes[0] = a00; lanes[1] = a01; lanes[2] = a02; lanes[3] = a03;
    lanes[4] = a04; lanes[5] = a10; lanes[6] = a11; lanes[7] = a12;
    for (uint i = 0; i < hash_len; i++) {
        const uchar b = (uchar)(lanes[i >> 3] >> ((i & 7) << 3));
        if (b != k_hash[i]) return 0;
    }
    return 1;
}

static int prsha3_compare_reg(__global const uchar* k_hash, const uchar* message, int len,
                              uint rate, uint hash_len, uchar pad) {
    ulong a00 = 0, a01 = 0, a02 = 0, a03 = 0, a04 = 0;
    ulong a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0;
    ulong a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;
    ulong a30 = 0, a31 = 0, a32 = 0, a33 = 0, a34 = 0;
    ulong a40 = 0, a41 = 0, a42 = 0, a43 = 0, a44 = 0;

    ulong lane0 = 0, lane1 = 0;
    for (int i = 0; i < len; i++) {
        if (i < 8) lane0 |= ((ulong)message[i]) << (i << 3);
        else lane1 |= ((ulong)message[i]) << ((i - 8) << 3);
    }
    if (len < 8) lane0 |= ((ulong)pad) << (len << 3);
    else lane1 |= ((ulong)pad) << ((len - 8) << 3);
    a00 = lane0;
    a01 = lane1;

    const uint last = (rate - 1u) >> 3;
    const ulong sep = 0x80UL << 56;
    if (last == 8u) a13 ^= sep;
    else if (last == 12u) a22 ^= sep;
    else if (last == 16u) a31 ^= sep;
    else a32 ^= sep;

    KECCAK_F1600();
    return prsha3_digest_eq(k_hash, hash_len, a00, a01, a02, a03, a04, a10, a11, a12);
}

static int prkeccak_384_compare(__global const uchar* k_hash, uchar* password, const int length) {
  return prsha3_compare_reg(k_hash, password, length, 104u, 48u, (uchar)1);
}

__kernel void prkeccak_384_kernel(__global uchar* result,
                          __global const uchar* k_dict,
                          __global const uchar* k_hash,
                          __global int* g_found,
                          const ulong start,
                          const uint count,
                          const uint pass_len,
                          const uint dict_length,
                          const uint min_len) {
  const uint ix = get_global_id(0);
  if (ix >= count || *g_found) return;
  ulong idx = start + (ulong)ix;
  uchar attempt[GPU_ATTEMPT_SIZE];
  for (int pos = (int)pass_len - 1; pos >= 0; --pos) {
    attempt[pos] = k_dict[idx % dict_length];
    idx /= dict_length;
  }
  for (uint i = 0; i < dict_length; ++i) {
    attempt[pass_len] = k_dict[i];
    if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {
      if (*g_found) return;
      if (prkeccak_384_compare(k_hash, attempt, (int)(pass_len + 1u))) {
        for (uint k = 0; k < pass_len + 1u; ++k) result[k] = attempt[k];
        result[pass_len + 1u] = 0;
        *g_found = 1;
        return;
      }
    }
    if (pass_len + 2u < min_len) continue;
    for (uint j = 0; j < dict_length; ++j) {
      attempt[pass_len + 1u] = k_dict[j];
      if (*g_found) return;
      if (prkeccak_384_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
