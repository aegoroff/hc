"#define GPU_ATTEMPT_SIZE 16\n"\
"#define BLOCK_LEN 128\n"\
"#define HASH_LEN 64\n"\
"#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))\n"\
"\n"\
"__constant ulong k_iv[8] = {\n"\
"    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,\n"\
"    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,\n"\
"    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,\n"\
"    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL,\n"\
"};\n"\
"\n"\
"__constant uchar k_sigma[12][16] = {\n"\
"    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },\n"\
"    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },\n"\
"    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },\n"\
"    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },\n"\
"    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },\n"\
"    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },\n"\
"    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },\n"\
"    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },\n"\
"    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },\n"\
"    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },\n"\
"    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },\n"\
"    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },\n"\
"};\n"\
"\n"\
"static void prblake2b_G(ulong* v, int a, int b, int c, int d, ulong x, ulong y) {\n"\
"    v[a] = v[a] + v[b] + x;\n"\
"    v[d] = ROTR64(v[d] ^ v[a], 32);\n"\
"    v[c] = v[c] + v[d];\n"\
"    v[b] = ROTR64(v[b] ^ v[c], 24);\n"\
"    v[a] = v[a] + v[b] + y;\n"\
"    v[d] = ROTR64(v[d] ^ v[a], 16);\n"\
"    v[c] = v[c] + v[d];\n"\
"    v[b] = ROTR64(v[b] ^ v[c], 63);\n"\
"}\n"\
"\n"\
"static void prblake2b_compress(ulong* h, const uchar* block, ulong t, int last) {\n"\
"    ulong m[16];\n"\
"    for (int i = 0; i < 16; i++) {\n"\
"        const int o = i * 8;\n"\
"        m[i] = (ulong)(block[o + 0])\n"\
"            | ((ulong)(block[o + 1]) << 8)\n"\
"            | ((ulong)(block[o + 2]) << 16)\n"\
"            | ((ulong)(block[o + 3]) << 24)\n"\
"            | ((ulong)(block[o + 4]) << 32)\n"\
"            | ((ulong)(block[o + 5]) << 40)\n"\
"            | ((ulong)(block[o + 6]) << 48)\n"\
"            | ((ulong)(block[o + 7]) << 56);\n"\
"    }\n"\
"\n"\
"    ulong v[16];\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        v[i] = h[i];\n"\
"        v[i + 8] = k_iv[i];\n"\
"    }\n"\
"    v[12] ^= t;\n"\
"    /* t fits in low 64 bits for short passwords; high half stays 0 */\n"\
"    if (last) {\n"\
"        v[14] = ~v[14];\n"\
"    }\n"\
"\n"\
"    for (int j = 0; j < 12; j++) {\n"\
"        __constant uchar* s = k_sigma[j];\n"\
"        prblake2b_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);\n"\
"        prblake2b_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);\n"\
"        prblake2b_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);\n"\
"        prblake2b_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);\n"\
"        prblake2b_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);\n"\
"        prblake2b_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);\n"\
"        prblake2b_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);\n"\
"        prblake2b_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);\n"\
"    }\n"\
"\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        h[i] ^= v[i] ^ v[i + 8];\n"\
"    }\n"\
"}\n"\
"\n"\
"static void prblake2b_hash(const uchar* message, uint len, uchar* hash) {\n"\
"    ulong h[8];\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        h[i] = k_iv[i];\n"\
"    }\n"\
"    /* fanout=1, depth=1, keylen=0, digest_length=64 */\n"\
"    h[0] ^= 0x01010000UL ^ HASH_LEN;\n"\
"\n"\
"    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;\n"\
"    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];\n"\
"    prblake2b_compress(h, block, (ulong)(len), 1);\n"\
"\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        hash[i * 8 + 0] = (uchar)(h[i]);\n"\
"        hash[i * 8 + 1] = (uchar)(h[i] >> 8);\n"\
"        hash[i * 8 + 2] = (uchar)(h[i] >> 16);\n"\
"        hash[i * 8 + 3] = (uchar)(h[i] >> 24);\n"\
"        hash[i * 8 + 4] = (uchar)(h[i] >> 32);\n"\
"        hash[i * 8 + 5] = (uchar)(h[i] >> 40);\n"\
"        hash[i * 8 + 6] = (uchar)(h[i] >> 48);\n"\
"        hash[i * 8 + 7] = (uchar)(h[i] >> 56);\n"\
"    }\n"\
"}\n"\
"\n"\
"static int prblake2b_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[HASH_LEN];\n"\
"  prblake2b_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < HASH_LEN; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prblake2b_kernel(__global uchar* result,\n"\
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
"      if (prblake2b_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prblake2b_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
