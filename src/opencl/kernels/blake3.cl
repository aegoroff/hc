#define GPU_ATTEMPT_SIZE 16
#define BLOCK_LEN 64
#define HASH_LEN 32
#define ROOT_FLAGS 11u /* CHUNK_START|CHUNK_END|ROOT */
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__constant uint k_iv[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

#define B3_G(v, a, b, c, d, x, y) \
    do { \
        (v)[a] = (v)[a] + (v)[b] + (x); \
        (v)[d] = ROTR32((v)[d] ^ (v)[a], 16); \
        (v)[c] = (v)[c] + (v)[d]; \
        (v)[b] = ROTR32((v)[b] ^ (v)[c], 12); \
        (v)[a] = (v)[a] + (v)[b] + (y); \
        (v)[d] = ROTR32((v)[d] ^ (v)[a], 8); \
        (v)[c] = (v)[c] + (v)[d]; \
        (v)[b] = ROTR32((v)[b] ^ (v)[c], 7); \
    } while (0)

static void prblake3_compress(uint* cv, const uchar* block, uint block_len) {
    uint m[16];
    for (int i = 0; i < 16; i++) {
        const int o = i * 4;
        m[i] = (uint)(block[o + 0])
            | ((uint)(block[o + 1]) << 8)
            | ((uint)(block[o + 2]) << 16)
            | ((uint)(block[o + 3]) << 24);
    }

    uint v[16];
    for (int i = 0; i < 8; i++) {
        v[i] = cv[i];
        v[i + 8] = k_iv[i];
    }
    v[12] = 0;
    v[13] = 0;
    v[14] = block_len;
    v[15] = ROOT_FLAGS;

    /* round 0 */
    B3_G(v, 0, 4, 8, 12, m[0], m[1]);
    B3_G(v, 1, 5, 9, 13, m[2], m[3]);
    B3_G(v, 2, 6, 10, 14, m[4], m[5]);
    B3_G(v, 3, 7, 11, 15, m[6], m[7]);
    B3_G(v, 0, 5, 10, 15, m[8], m[9]);
    B3_G(v, 1, 6, 11, 12, m[10], m[11]);
    B3_G(v, 2, 7, 8, 13, m[12], m[13]);
    B3_G(v, 3, 4, 9, 14, m[14], m[15]);
    /* round 1 */
    B3_G(v, 0, 4, 8, 12, m[2], m[6]);
    B3_G(v, 1, 5, 9, 13, m[3], m[10]);
    B3_G(v, 2, 6, 10, 14, m[7], m[0]);
    B3_G(v, 3, 7, 11, 15, m[4], m[13]);
    B3_G(v, 0, 5, 10, 15, m[1], m[11]);
    B3_G(v, 1, 6, 11, 12, m[12], m[5]);
    B3_G(v, 2, 7, 8, 13, m[9], m[14]);
    B3_G(v, 3, 4, 9, 14, m[15], m[8]);
    /* round 2 */
    B3_G(v, 0, 4, 8, 12, m[3], m[4]);
    B3_G(v, 1, 5, 9, 13, m[10], m[12]);
    B3_G(v, 2, 6, 10, 14, m[13], m[2]);
    B3_G(v, 3, 7, 11, 15, m[7], m[14]);
    B3_G(v, 0, 5, 10, 15, m[6], m[5]);
    B3_G(v, 1, 6, 11, 12, m[9], m[0]);
    B3_G(v, 2, 7, 8, 13, m[11], m[15]);
    B3_G(v, 3, 4, 9, 14, m[8], m[1]);
    /* round 3 */
    B3_G(v, 0, 4, 8, 12, m[10], m[7]);
    B3_G(v, 1, 5, 9, 13, m[12], m[9]);
    B3_G(v, 2, 6, 10, 14, m[14], m[3]);
    B3_G(v, 3, 7, 11, 15, m[13], m[15]);
    B3_G(v, 0, 5, 10, 15, m[4], m[0]);
    B3_G(v, 1, 6, 11, 12, m[11], m[2]);
    B3_G(v, 2, 7, 8, 13, m[5], m[8]);
    B3_G(v, 3, 4, 9, 14, m[1], m[6]);
    /* round 4 */
    B3_G(v, 0, 4, 8, 12, m[12], m[13]);
    B3_G(v, 1, 5, 9, 13, m[9], m[11]);
    B3_G(v, 2, 6, 10, 14, m[15], m[10]);
    B3_G(v, 3, 7, 11, 15, m[14], m[8]);
    B3_G(v, 0, 5, 10, 15, m[7], m[2]);
    B3_G(v, 1, 6, 11, 12, m[5], m[3]);
    B3_G(v, 2, 7, 8, 13, m[0], m[1]);
    B3_G(v, 3, 4, 9, 14, m[6], m[4]);
    /* round 5 */
    B3_G(v, 0, 4, 8, 12, m[9], m[14]);
    B3_G(v, 1, 5, 9, 13, m[11], m[5]);
    B3_G(v, 2, 6, 10, 14, m[8], m[12]);
    B3_G(v, 3, 7, 11, 15, m[15], m[1]);
    B3_G(v, 0, 5, 10, 15, m[13], m[3]);
    B3_G(v, 1, 6, 11, 12, m[0], m[10]);
    B3_G(v, 2, 7, 8, 13, m[2], m[6]);
    B3_G(v, 3, 4, 9, 14, m[4], m[7]);
    /* round 6 */
    B3_G(v, 0, 4, 8, 12, m[11], m[15]);
    B3_G(v, 1, 5, 9, 13, m[5], m[0]);
    B3_G(v, 2, 6, 10, 14, m[1], m[9]);
    B3_G(v, 3, 7, 11, 15, m[8], m[6]);
    B3_G(v, 0, 5, 10, 15, m[14], m[10]);
    B3_G(v, 1, 6, 11, 12, m[2], m[12]);
    B3_G(v, 2, 7, 8, 13, m[3], m[4]);
    B3_G(v, 3, 4, 9, 14, m[7], m[13]);

    for (int i = 0; i < 8; i++) {
        cv[i] = v[i] ^ v[i + 8];
    }
}

static void prblake3_hash(const uchar* message, uint len, uchar* hash) {
    uint cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = k_iv[i];
    }

    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;
    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];
    prblake3_compress(cv, block, len);

    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = (uchar)(cv[i]);
        hash[i * 4 + 1] = (uchar)(cv[i] >> 8);
        hash[i * 4 + 2] = (uchar)(cv[i] >> 16);
        hash[i * 4 + 3] = (uchar)(cv[i] >> 24);
    }
}

static int prblake3_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[HASH_LEN];
  prblake3_hash(password, (uint)length, hash);
  for (int i = 0; i < HASH_LEN; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prblake3_kernel(__global uchar* result,
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
      if (prblake3_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prblake3_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
