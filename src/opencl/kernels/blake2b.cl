#define GPU_ATTEMPT_SIZE 16
#define BLOCK_LEN 128
#define HASH_LEN 64
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

__constant ulong k_iv[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL,
};

__constant uchar k_sigma[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

static void prblake2b_G(ulong* v, int a, int b, int c, int d, ulong x, ulong y) {
    v[a] = v[a] + v[b] + x;
    v[d] = ROTR64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = ROTR64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = ROTR64(v[b] ^ v[c], 63);
}

static void prblake2b_compress(ulong* h, const uchar* block, ulong t, int last) {
    ulong m[16];
    for (int i = 0; i < 16; i++) {
        const int o = i * 8;
        m[i] = (ulong)(block[o + 0])
            | ((ulong)(block[o + 1]) << 8)
            | ((ulong)(block[o + 2]) << 16)
            | ((ulong)(block[o + 3]) << 24)
            | ((ulong)(block[o + 4]) << 32)
            | ((ulong)(block[o + 5]) << 40)
            | ((ulong)(block[o + 6]) << 48)
            | ((ulong)(block[o + 7]) << 56);
    }

    ulong v[16];
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = k_iv[i];
    }
    v[12] ^= t;
    /* t fits in low 64 bits for short passwords; high half stays 0 */
    if (last) {
        v[14] = ~v[14];
    }

    for (int j = 0; j < 12; j++) {
        __constant uchar* s = k_sigma[j];
        prblake2b_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        prblake2b_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        prblake2b_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        prblake2b_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        prblake2b_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        prblake2b_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        prblake2b_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        prblake2b_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

static void prblake2b_hash(const uchar* message, uint len, uchar* hash) {
    ulong h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = k_iv[i];
    }
    /* fanout=1, depth=1, keylen=0, digest_length=64 */
    h[0] ^= 0x01010000UL ^ HASH_LEN;

    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;
    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];
    prblake2b_compress(h, block, (ulong)(len), 1);

    for (int i = 0; i < 8; i++) {
        hash[i * 8 + 0] = (uchar)(h[i]);
        hash[i * 8 + 1] = (uchar)(h[i] >> 8);
        hash[i * 8 + 2] = (uchar)(h[i] >> 16);
        hash[i * 8 + 3] = (uchar)(h[i] >> 24);
        hash[i * 8 + 4] = (uchar)(h[i] >> 32);
        hash[i * 8 + 5] = (uchar)(h[i] >> 40);
        hash[i * 8 + 6] = (uchar)(h[i] >> 48);
        hash[i * 8 + 7] = (uchar)(h[i] >> 56);
    }
}

static int prblake2b_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[HASH_LEN];
  prblake2b_hash(password, (uint)length, hash);
  for (int i = 0; i < HASH_LEN; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prblake2b_kernel(__global uchar* result,
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
      if (prblake2b_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prblake2b_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
