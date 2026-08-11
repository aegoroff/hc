"#define GPU_ATTEMPT_SIZE 16\n"\
"#define BLOCK_LEN 64\n"\
"#define HASH_LEN 20\n"\
"#define NUM_ROUNDS 80\n"\
"#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))\n"\
"\n"\
"__constant uint KL[5] = {\n"\
"    0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu };\n"\
"__constant uint KR[5] = {\n"\
"    0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u };\n"\
"__constant int RL[NUM_ROUNDS] = {\n"\
"    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,\n"\
"    7,  4, 13,  1, 10,  6, 15,  3, 12,  0,  9,  5,  2, 14, 11,  8,\n"\
"    3, 10, 14,  4,  9, 15,  8,  1,  2,  7,  0,  6, 13, 11,  5, 12,\n"\
"    1,  9, 11, 10,  0,  8, 12,  4, 13,  3,  7, 15, 14,  5,  6,  2,\n"\
"    4,  0,  5,  9,  7, 12,  2, 10, 14,  1,  3,  8, 11,  6, 15, 13 };\n"\
"__constant int RR[NUM_ROUNDS] = {\n"\
"    5, 14,  7,  0,  9,  2, 11,  4, 13,  6, 15,  8,  1, 10,  3, 12,\n"\
"    6, 11,  3,  7,  0, 13,  5, 10, 14, 15,  8, 12,  4,  9,  1,  2,\n"\
"    15,  5,  1,  3,  7, 14,  6,  9, 11,  8, 12,  2, 10,  0,  4, 13,\n"\
"    8,  6,  4,  1,  3, 11, 15,  0,  5, 12,  2, 13,  9,  7, 10, 14,\n"\
"    12, 15, 10,  4,  1,  5,  8,  7,  6,  2, 13, 14,  0,  3,  9, 11 };\n"\
"__constant int SL[NUM_ROUNDS] = {\n"\
"    11, 14, 15, 12,  5,  8,  7,  9, 11, 13, 14, 15,  6,  7,  9,  8,\n"\
"    7,  6,  8, 13, 11,  9,  7, 15,  7, 12, 15,  9, 11,  7, 13, 12,\n"\
"    11, 13,  6,  7, 14,  9, 13, 15, 14,  8, 13,  6,  5, 12,  7,  5,\n"\
"    11, 12, 14, 15, 14, 15,  9,  8,  9, 14,  5,  6,  8,  6,  5, 12,\n"\
"    9, 15,  5, 11,  6,  8, 13, 12,  5, 12, 13, 14, 11,  8,  5,  6 };\n"\
"__constant int SR[NUM_ROUNDS] = {\n"\
"    8,  9,  9, 11, 13, 15, 15,  5,  7,  7,  8, 11, 14, 14, 12,  6,\n"\
"    9, 13, 15,  7, 12,  8,  9, 11,  7,  7, 12,  7,  6, 15, 13, 11,\n"\
"    9,  7, 15, 11,  8,  6,  6, 14, 12, 13,  5, 14, 13, 13,  7,  5,\n"\
"    15,  5,  8, 11, 14, 14,  6, 14,  6,  9, 12,  9, 12,  5, 15,  8,\n"\
"    8,  5, 12,  9, 12,  5, 14,  6,  8, 13,  6,  5, 15, 13, 11, 11 };\n"\
"\n"\
"static uint f(int i, uint x, uint y, uint z) {\n"\
"    switch (i >> 4) {\n"\
"        case 0: return x ^ y ^ z;\n"\
"        case 1: return (x & y) | (~x & z);\n"\
"        case 2: return (x | ~y) ^ z;\n"\
"        case 3: return (x & z) | (y & ~z);\n"\
"        case 4: return x ^ (y | ~z);\n"\
"        default: return 0; // Dummy value to please the compiler\n"\
"    }\n"\
"}\n"\
"\n"\
"static void prrmd160_compress(uint* state, const uchar* blocks, uint len);\n"\
"\n"\
"static void prrmd160_hash(const uchar* message, uint len, uchar* hash) {\n"\
"    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };\n"\
"    uint off = len & ~(BLOCK_LEN - 1);\n"\
"    prrmd160_compress(state, message, off);\n"\
"\n"\
"    // Final blocks, padding, and length\n"\
"    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;\n"\
"    { const uint __n = (uint)(len - off); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = (&message[off])[__i]; }\n"\
"    off = len & (BLOCK_LEN - 1);\n"\
"    block[off] = 0x80;\n"\
"    ++off;\n"\
"    if (off + 8 > BLOCK_LEN) {\n"\
"        prrmd160_compress(state, block, BLOCK_LEN);\n"\
"        { const uint __n = (uint)(BLOCK_LEN); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = 0; }\n"\
"    }\n"\
"    block[BLOCK_LEN - 8] = ((len & 0x1FU) << 3);\n"\
"    len >>= 5;\n"\
"    for (int i = 1; i < 8; i++, len >>= 8)\n"\
"        block[BLOCK_LEN - 8 + i] = (len);\n"\
"    prrmd160_compress(state, block, BLOCK_LEN);\n"\
"\n"\
"    // Uint32 array to bytes in little endian\n"\
"    for (int i = 0; i < HASH_LEN; i++)\n"\
"        hash[i] = (state[i >> 2] >> ((i & 3) << 3));\n"\
"}\n"\
"\n"\
"static void prrmd160_compress(uint* state, const uchar* blocks, uint len) {\n"\
"#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))  // Assumes that x is uint and 0 < n < 32\n"\
"    uint schedule[16];\n"\
"    for (uint i = 0; i < len; ) {\n"\
"\n"\
"        // Message schedule\n"\
"        for (int j = 0; j < 16; j++, i += 4) {\n"\
"            schedule[j] = (blocks[i + 0]) << 0\n"\
"                | (blocks[i + 1]) << 8\n"\
"                | (blocks[i + 2]) << 16\n"\
"                | (blocks[i + 3]) << 24;\n"\
"        }\n"\
"\n"\
"        // The 80 rounds\n"\
"        uint al = state[0], ar = state[0];\n"\
"        uint bl = state[1], br = state[1];\n"\
"        uint cl = state[2], cr = state[2];\n"\
"        uint dl = state[3], dr = state[3];\n"\
"        uint el = state[4], er = state[4];\n"\
"        for (int j = 0; j < NUM_ROUNDS; j++) {\n"\
"            uint temp = 0U + ROTL32(0U + al + f(j, bl, cl, dl) + schedule[RL[j]] + KL[j >> 4], SL[j]) + el;\n"\
"            al = el;\n"\
"            el = dl;\n"\
"            dl = ROTL32(cl, 10);\n"\
"            cl = bl;\n"\
"            bl = temp;\n"\
"            temp = 0U + ROTL32(0U + ar + f(NUM_ROUNDS - 1 - j, br, cr, dr) + schedule[RR[j]] + KR[j >> 4], SR[j]) + er;\n"\
"            ar = er;\n"\
"            er = dr;\n"\
"            dr = ROTL32(cr, 10);\n"\
"            cr = br;\n"\
"            br = temp;\n"\
"        }\n"\
"        uint temp = 0U + state[1] + cl + dr;\n"\
"        state[1] = 0U + state[2] + dl + er;\n"\
"        state[2] = 0U + state[3] + el + ar;\n"\
"        state[3] = 0U + state[4] + al + br;\n"\
"        state[4] = 0U + state[0] + bl + cr;\n"\
"        state[0] = temp;\n"\
"    }\n"\
"}\n"\
"\n"\
"static int prrmd160_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[20];\n"\
"  prrmd160_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < 20; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prrmd160_kernel(__global uchar* result,\n"\
"                          __global const uchar* k_dict,\n"\
"                          __global const uchar* k_hash,\n"\
"                          __global int* g_found,\n"\
"                          const ulong start,\n"\
"                          const uint count,\n"\
"                          const uint pass_len,\n"\
"                          const uint dict_length,\n"\
"                          const uint min_len) {\n"\
"  const uint ix = get_global_id(0);\n"\
"  if (ix >= count || *g_found) return;\n"\
"  ulong idx = start + (ulong)ix;\n"\
"  uchar attempt[GPU_ATTEMPT_SIZE];\n"\
"  for (int pos = (int)pass_len - 1; pos >= 0; --pos) {\n"\
"    attempt[pos] = k_dict[idx % dict_length];\n"\
"    idx /= dict_length;\n"\
"  }\n"\
"  for (uint i = 0; i < dict_length; ++i) {\n"\
"    attempt[pass_len] = k_dict[i];\n"\
"    if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {\n"\
"      if (*g_found) return;\n"\
"      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
"        for (uint k = 0; k < pass_len + 1u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 1u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"    if (pass_len + 2u < min_len) continue;\n"\
"    for (uint j = 0; j < dict_length; ++j) {\n"\
"      attempt[pass_len + 1u] = k_dict[j];\n"\
"      if (*g_found) return;\n"\
"      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
