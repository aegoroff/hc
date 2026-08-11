#define GPU_ATTEMPT_SIZE 16
#define BLOCK_LEN 64
#define HASH_LEN 20
#define NUM_ROUNDS 80
#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))

__constant uint KL[5] = {
    0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu };
__constant uint KR[5] = {
    0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u };
__constant int RL[NUM_ROUNDS] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    7,  4, 13,  1, 10,  6, 15,  3, 12,  0,  9,  5,  2, 14, 11,  8,
    3, 10, 14,  4,  9, 15,  8,  1,  2,  7,  0,  6, 13, 11,  5, 12,
    1,  9, 11, 10,  0,  8, 12,  4, 13,  3,  7, 15, 14,  5,  6,  2,
    4,  0,  5,  9,  7, 12,  2, 10, 14,  1,  3,  8, 11,  6, 15, 13 };
__constant int RR[NUM_ROUNDS] = {
    5, 14,  7,  0,  9,  2, 11,  4, 13,  6, 15,  8,  1, 10,  3, 12,
    6, 11,  3,  7,  0, 13,  5, 10, 14, 15,  8, 12,  4,  9,  1,  2,
    15,  5,  1,  3,  7, 14,  6,  9, 11,  8, 12,  2, 10,  0,  4, 13,
    8,  6,  4,  1,  3, 11, 15,  0,  5, 12,  2, 13,  9,  7, 10, 14,
    12, 15, 10,  4,  1,  5,  8,  7,  6,  2, 13, 14,  0,  3,  9, 11 };
__constant int SL[NUM_ROUNDS] = {
    11, 14, 15, 12,  5,  8,  7,  9, 11, 13, 14, 15,  6,  7,  9,  8,
    7,  6,  8, 13, 11,  9,  7, 15,  7, 12, 15,  9, 11,  7, 13, 12,
    11, 13,  6,  7, 14,  9, 13, 15, 14,  8, 13,  6,  5, 12,  7,  5,
    11, 12, 14, 15, 14, 15,  9,  8,  9, 14,  5,  6,  8,  6,  5, 12,
    9, 15,  5, 11,  6,  8, 13, 12,  5, 12, 13, 14, 11,  8,  5,  6 };
__constant int SR[NUM_ROUNDS] = {
    8,  9,  9, 11, 13, 15, 15,  5,  7,  7,  8, 11, 14, 14, 12,  6,
    9, 13, 15,  7, 12,  8,  9, 11,  7,  7, 12,  7,  6, 15, 13, 11,
    9,  7, 15, 11,  8,  6,  6, 14, 12, 13,  5, 14, 13, 13,  7,  5,
    15,  5,  8, 11, 14, 14,  6, 14,  6,  9, 12,  9, 12,  5, 15,  8,
    8,  5, 12,  9, 12,  5, 14,  6,  8, 13,  6,  5, 15, 13, 11, 11 };

static uint f(int i, uint x, uint y, uint z) {
    switch (i >> 4) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        case 4: return x ^ (y | ~z);
        default: return 0; // Dummy value to please the compiler
    }
}

static void prrmd160_compress(uint* state, const uchar* blocks, uint len);

static void prrmd160_hash(const uchar* message, uint len, uchar* hash) {
    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    uint off = len & ~(BLOCK_LEN - 1);
    prrmd160_compress(state, message, off);

    // Final blocks, padding, and length
    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;
    { const uint __n = (uint)(len - off); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = (&message[off])[__i]; }
    off = len & (BLOCK_LEN - 1);
    block[off] = 0x80;
    ++off;
    if (off + 8 > BLOCK_LEN) {
        prrmd160_compress(state, block, BLOCK_LEN);
        { const uint __n = (uint)(BLOCK_LEN); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = 0; }
    }
    block[BLOCK_LEN - 8] = ((len & 0x1FU) << 3);
    len >>= 5;
    for (int i = 1; i < 8; i++, len >>= 8)
        block[BLOCK_LEN - 8 + i] = (len);
    prrmd160_compress(state, block, BLOCK_LEN);

    // Uint32 array to bytes in little endian
    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = (state[i >> 2] >> ((i & 3) << 3));
}

static void prrmd160_compress(uint* state, const uchar* blocks, uint len) {
#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))  // Assumes that x is uint and 0 < n < 32
    uint schedule[16];
    for (uint i = 0; i < len; ) {

        // Message schedule
        for (int j = 0; j < 16; j++, i += 4) {
            schedule[j] = (blocks[i + 0]) << 0
                | (blocks[i + 1]) << 8
                | (blocks[i + 2]) << 16
                | (blocks[i + 3]) << 24;
        }

        // The 80 rounds
        uint al = state[0], ar = state[0];
        uint bl = state[1], br = state[1];
        uint cl = state[2], cr = state[2];
        uint dl = state[3], dr = state[3];
        uint el = state[4], er = state[4];
        for (int j = 0; j < NUM_ROUNDS; j++) {
            uint temp = 0U + ROTL32(0U + al + f(j, bl, cl, dl) + schedule[RL[j]] + KL[j >> 4], SL[j]) + el;
            al = el;
            el = dl;
            dl = ROTL32(cl, 10);
            cl = bl;
            bl = temp;
            temp = 0U + ROTL32(0U + ar + f(NUM_ROUNDS - 1 - j, br, cr, dr) + schedule[RR[j]] + KR[j >> 4], SR[j]) + er;
            ar = er;
            er = dr;
            dr = ROTL32(cr, 10);
            cr = br;
            br = temp;
        }
        uint temp = 0U + state[1] + cl + dr;
        state[1] = 0U + state[2] + dl + er;
        state[2] = 0U + state[3] + el + ar;
        state[3] = 0U + state[4] + al + br;
        state[4] = 0U + state[0] + bl + cr;
        state[0] = temp;
    }
}

static int prrmd160_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[20];
  prrmd160_hash(password, (uint)length, hash);
  for (int i = 0; i < 20; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prrmd160_kernel(__global uchar* result,
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
      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
