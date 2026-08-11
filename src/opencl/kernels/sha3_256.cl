#define GPU_ATTEMPT_SIZE 16
#define MAX_RATE 144
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

__constant ulong k_rc[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
};

static void prkeccak_f1600(ulong* A) {
    for (int round = 0; round < 24; round++) {
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        D[0] = ROTL64(C[1], 1) ^ C[4];
        D[1] = ROTL64(C[2], 1) ^ C[0];
        D[2] = ROTL64(C[3], 1) ^ C[1];
        D[3] = ROTL64(C[4], 1) ^ C[2];
        D[4] = ROTL64(C[0], 1) ^ C[3];
        for (int x = 0; x < 5; x++) {
            A[x] ^= D[x];
            A[x + 5] ^= D[x];
            A[x + 10] ^= D[x];
            A[x + 15] ^= D[x];
            A[x + 20] ^= D[x];
        }

        A[1] = ROTL64(A[1], 1);
        A[2] = ROTL64(A[2], 62);
        A[3] = ROTL64(A[3], 28);
        A[4] = ROTL64(A[4], 27);
        A[5] = ROTL64(A[5], 36);
        A[6] = ROTL64(A[6], 44);
        A[7] = ROTL64(A[7], 6);
        A[8] = ROTL64(A[8], 55);
        A[9] = ROTL64(A[9], 20);
        A[10] = ROTL64(A[10], 3);
        A[11] = ROTL64(A[11], 10);
        A[12] = ROTL64(A[12], 43);
        A[13] = ROTL64(A[13], 25);
        A[14] = ROTL64(A[14], 39);
        A[15] = ROTL64(A[15], 41);
        A[16] = ROTL64(A[16], 45);
        A[17] = ROTL64(A[17], 15);
        A[18] = ROTL64(A[18], 21);
        A[19] = ROTL64(A[19], 8);
        A[20] = ROTL64(A[20], 18);
        A[21] = ROTL64(A[21], 2);
        A[22] = ROTL64(A[22], 61);
        A[23] = ROTL64(A[23], 56);
        A[24] = ROTL64(A[24], 14);

        {
            ulong A1 = A[1];
            A[1] = A[6];
            A[6] = A[9];
            A[9] = A[22];
            A[22] = A[14];
            A[14] = A[20];
            A[20] = A[2];
            A[2] = A[12];
            A[12] = A[13];
            A[13] = A[19];
            A[19] = A[23];
            A[23] = A[15];
            A[15] = A[4];
            A[4] = A[24];
            A[24] = A[21];
            A[21] = A[8];
            A[8] = A[16];
            A[16] = A[5];
            A[5] = A[3];
            A[3] = A[18];
            A[18] = A[17];
            A[17] = A[11];
            A[11] = A[7];
            A[7] = A[10];
            A[10] = A1;
        }

        for (int i = 0; i < 25; i += 5) {
            ulong A0 = A[0 + i], A1 = A[1 + i];
            A[0 + i] ^= ~A1 & A[2 + i];
            A[1 + i] ^= ~A[2 + i] & A[3 + i];
            A[2 + i] ^= ~A[3 + i] & A[4 + i];
            A[3 + i] ^= ~A[4 + i] & A0;
            A[4 + i] ^= ~A0 & A1;
        }

        A[0] ^= k_rc[round];
    }
}

static void prsha3_hash(const uchar* message, uint len, uchar* out,
                                             uint rate, uint out_len, uchar pad) {
    ulong state[25]; for (int __z = 0; __z < (int)(25); ++__z) state[__z] = 0;
    uchar block[MAX_RATE]; for (int __z = 0; __z < (int)(MAX_RATE); ++__z) block[__z] = 0;
    { const uint __n = (uint)(len); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = (message)[__i]; }
    block[len] |= pad;
    block[rate - 1] |= 0x80;

    const uint nq = rate / 8;
    for (uint i = 0; i < nq; i++) {
        const uint o = i * 8;
        state[i] ^= (ulong)(block[o + 0])
            | ((ulong)(block[o + 1]) << 8)
            | ((ulong)(block[o + 2]) << 16)
            | ((ulong)(block[o + 3]) << 24)
            | ((ulong)(block[o + 4]) << 32)
            | ((ulong)(block[o + 5]) << 40)
            | ((ulong)(block[o + 6]) << 48)
            | ((ulong)(block[o + 7]) << 56);
    }
    prkeccak_f1600(state);

    for (uint i = 0; i < out_len; i++) {
        out[i] = (uchar)(state[i >> 3] >> ((i & 7) << 3));
    }
}

static int prsha3_256_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[32];
  prsha3_hash(password, (uint)length, hash, 136, 32, (uchar)6);
  for (int i = 0; i < 32; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prsha3_256_kernel(__global uchar* result,
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
      if (prsha3_256_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prsha3_256_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
