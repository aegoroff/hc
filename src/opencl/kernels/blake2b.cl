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

#define B2B_G(v, a, b, c, d, x, y) \
    do { \
        (v)[a] = (v)[a] + (v)[b] + (x); \
        (v)[d] = ROTR64((v)[d] ^ (v)[a], 32); \
        (v)[c] = (v)[c] + (v)[d]; \
        (v)[b] = ROTR64((v)[b] ^ (v)[c], 24); \
        (v)[a] = (v)[a] + (v)[b] + (y); \
        (v)[d] = ROTR64((v)[d] ^ (v)[a], 16); \
        (v)[c] = (v)[c] + (v)[d]; \
        (v)[b] = ROTR64((v)[b] ^ (v)[c], 63); \
    } while (0)

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

    /* round 0 */
    B2B_G(v, 0, 4, 8, 12, m[0], m[1]);
    B2B_G(v, 1, 5, 9, 13, m[2], m[3]);
    B2B_G(v, 2, 6, 10, 14, m[4], m[5]);
    B2B_G(v, 3, 7, 11, 15, m[6], m[7]);
    B2B_G(v, 0, 5, 10, 15, m[8], m[9]);
    B2B_G(v, 1, 6, 11, 12, m[10], m[11]);
    B2B_G(v, 2, 7, 8, 13, m[12], m[13]);
    B2B_G(v, 3, 4, 9, 14, m[14], m[15]);
    /* round 1 */
    B2B_G(v, 0, 4, 8, 12, m[14], m[10]);
    B2B_G(v, 1, 5, 9, 13, m[4], m[8]);
    B2B_G(v, 2, 6, 10, 14, m[9], m[15]);
    B2B_G(v, 3, 7, 11, 15, m[13], m[6]);
    B2B_G(v, 0, 5, 10, 15, m[1], m[12]);
    B2B_G(v, 1, 6, 11, 12, m[0], m[2]);
    B2B_G(v, 2, 7, 8, 13, m[11], m[7]);
    B2B_G(v, 3, 4, 9, 14, m[5], m[3]);
    /* round 2 */
    B2B_G(v, 0, 4, 8, 12, m[11], m[8]);
    B2B_G(v, 1, 5, 9, 13, m[12], m[0]);
    B2B_G(v, 2, 6, 10, 14, m[5], m[2]);
    B2B_G(v, 3, 7, 11, 15, m[15], m[13]);
    B2B_G(v, 0, 5, 10, 15, m[10], m[14]);
    B2B_G(v, 1, 6, 11, 12, m[3], m[6]);
    B2B_G(v, 2, 7, 8, 13, m[7], m[1]);
    B2B_G(v, 3, 4, 9, 14, m[9], m[4]);
    /* round 3 */
    B2B_G(v, 0, 4, 8, 12, m[7], m[9]);
    B2B_G(v, 1, 5, 9, 13, m[3], m[1]);
    B2B_G(v, 2, 6, 10, 14, m[13], m[12]);
    B2B_G(v, 3, 7, 11, 15, m[11], m[14]);
    B2B_G(v, 0, 5, 10, 15, m[2], m[6]);
    B2B_G(v, 1, 6, 11, 12, m[5], m[10]);
    B2B_G(v, 2, 7, 8, 13, m[4], m[0]);
    B2B_G(v, 3, 4, 9, 14, m[15], m[8]);
    /* round 4 */
    B2B_G(v, 0, 4, 8, 12, m[9], m[0]);
    B2B_G(v, 1, 5, 9, 13, m[5], m[7]);
    B2B_G(v, 2, 6, 10, 14, m[2], m[4]);
    B2B_G(v, 3, 7, 11, 15, m[10], m[15]);
    B2B_G(v, 0, 5, 10, 15, m[14], m[1]);
    B2B_G(v, 1, 6, 11, 12, m[11], m[12]);
    B2B_G(v, 2, 7, 8, 13, m[6], m[8]);
    B2B_G(v, 3, 4, 9, 14, m[3], m[13]);
    /* round 5 */
    B2B_G(v, 0, 4, 8, 12, m[2], m[12]);
    B2B_G(v, 1, 5, 9, 13, m[6], m[10]);
    B2B_G(v, 2, 6, 10, 14, m[0], m[11]);
    B2B_G(v, 3, 7, 11, 15, m[8], m[3]);
    B2B_G(v, 0, 5, 10, 15, m[4], m[13]);
    B2B_G(v, 1, 6, 11, 12, m[7], m[5]);
    B2B_G(v, 2, 7, 8, 13, m[15], m[14]);
    B2B_G(v, 3, 4, 9, 14, m[1], m[9]);
    /* round 6 */
    B2B_G(v, 0, 4, 8, 12, m[12], m[5]);
    B2B_G(v, 1, 5, 9, 13, m[1], m[15]);
    B2B_G(v, 2, 6, 10, 14, m[14], m[13]);
    B2B_G(v, 3, 7, 11, 15, m[4], m[10]);
    B2B_G(v, 0, 5, 10, 15, m[0], m[7]);
    B2B_G(v, 1, 6, 11, 12, m[6], m[3]);
    B2B_G(v, 2, 7, 8, 13, m[9], m[2]);
    B2B_G(v, 3, 4, 9, 14, m[8], m[11]);
    /* round 7 */
    B2B_G(v, 0, 4, 8, 12, m[13], m[11]);
    B2B_G(v, 1, 5, 9, 13, m[7], m[14]);
    B2B_G(v, 2, 6, 10, 14, m[12], m[1]);
    B2B_G(v, 3, 7, 11, 15, m[3], m[9]);
    B2B_G(v, 0, 5, 10, 15, m[5], m[0]);
    B2B_G(v, 1, 6, 11, 12, m[15], m[4]);
    B2B_G(v, 2, 7, 8, 13, m[8], m[6]);
    B2B_G(v, 3, 4, 9, 14, m[2], m[10]);
    /* round 8 */
    B2B_G(v, 0, 4, 8, 12, m[6], m[15]);
    B2B_G(v, 1, 5, 9, 13, m[14], m[9]);
    B2B_G(v, 2, 6, 10, 14, m[11], m[3]);
    B2B_G(v, 3, 7, 11, 15, m[0], m[8]);
    B2B_G(v, 0, 5, 10, 15, m[12], m[2]);
    B2B_G(v, 1, 6, 11, 12, m[13], m[7]);
    B2B_G(v, 2, 7, 8, 13, m[1], m[4]);
    B2B_G(v, 3, 4, 9, 14, m[10], m[5]);
    /* round 9 */
    B2B_G(v, 0, 4, 8, 12, m[10], m[2]);
    B2B_G(v, 1, 5, 9, 13, m[8], m[4]);
    B2B_G(v, 2, 6, 10, 14, m[7], m[6]);
    B2B_G(v, 3, 7, 11, 15, m[1], m[5]);
    B2B_G(v, 0, 5, 10, 15, m[15], m[11]);
    B2B_G(v, 1, 6, 11, 12, m[9], m[14]);
    B2B_G(v, 2, 7, 8, 13, m[3], m[12]);
    B2B_G(v, 3, 4, 9, 14, m[13], m[0]);
    /* round 10 */
    B2B_G(v, 0, 4, 8, 12, m[0], m[1]);
    B2B_G(v, 1, 5, 9, 13, m[2], m[3]);
    B2B_G(v, 2, 6, 10, 14, m[4], m[5]);
    B2B_G(v, 3, 7, 11, 15, m[6], m[7]);
    B2B_G(v, 0, 5, 10, 15, m[8], m[9]);
    B2B_G(v, 1, 6, 11, 12, m[10], m[11]);
    B2B_G(v, 2, 7, 8, 13, m[12], m[13]);
    B2B_G(v, 3, 4, 9, 14, m[14], m[15]);
    /* round 11 */
    B2B_G(v, 0, 4, 8, 12, m[14], m[10]);
    B2B_G(v, 1, 5, 9, 13, m[4], m[8]);
    B2B_G(v, 2, 6, 10, 14, m[9], m[15]);
    B2B_G(v, 3, 7, 11, 15, m[13], m[6]);
    B2B_G(v, 0, 5, 10, 15, m[1], m[12]);
    B2B_G(v, 1, 6, 11, 12, m[0], m[2]);
    B2B_G(v, 2, 7, 8, 13, m[11], m[7]);
    B2B_G(v, 3, 4, 9, 14, m[5], m[3]);

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
