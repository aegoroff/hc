"#define GPU_ATTEMPT_SIZE 16\n"\
"#define BLOCK_LEN 64\n"\
"#define HASH_LEN 32\n"\
"#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))\n"\
"\n"\
"__constant uint k_iv[8] = {\n"\
"    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,\n"\
"    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,\n"\
"};\n"\
"\n"\
"__constant uchar k_sigma[10][16] = {\n"\
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
"};\n"\
"\n"\
"static void prblake2s_G(uint* v, int a, int b, int c, int d, uint x, uint y) {\n"\
"    v[a] = v[a] + v[b] + x;\n"\
"    v[d] = ROTR32(v[d] ^ v[a], 16);\n"\
"    v[c] = v[c] + v[d];\n"\
"    v[b] = ROTR32(v[b] ^ v[c], 12);\n"\
"    v[a] = v[a] + v[b] + y;\n"\
"    v[d] = ROTR32(v[d] ^ v[a], 8);\n"\
"    v[c] = v[c] + v[d];\n"\
"    v[b] = ROTR32(v[b] ^ v[c], 7);\n"\
"}\n"\
"\n"\
"static void prblake2s_compress(uint* h, const uchar* block, ulong t, int last) {\n"\
"    uint m[16];\n"\
"    for (int i = 0; i < 16; i++) {\n"\
"        const int o = i * 4;\n"\
"        m[i] = (uint)(block[o + 0])\n"\
"            | ((uint)(block[o + 1]) << 8)\n"\
"            | ((uint)(block[o + 2]) << 16)\n"\
"            | ((uint)(block[o + 3]) << 24);\n"\
"    }\n"\
"\n"\
"    uint v[16];\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        v[i] = h[i];\n"\
"        v[i + 8] = k_iv[i];\n"\
"    }\n"\
"    v[12] ^= (uint)(t);\n"\
"    v[13] ^= (uint)(t >> 32);\n"\
"    if (last) {\n"\
"        v[14] = ~v[14];\n"\
"    }\n"\
"\n"\
"    for (int j = 0; j < 10; j++) {\n"\
"        __constant uchar* s = k_sigma[j];\n"\
"        prblake2s_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);\n"\
"        prblake2s_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);\n"\
"        prblake2s_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);\n"\
"        prblake2s_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);\n"\
"        prblake2s_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);\n"\
"        prblake2s_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);\n"\
"        prblake2s_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);\n"\
"        prblake2s_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);\n"\
"    }\n"\
"\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        h[i] ^= v[i] ^ v[i + 8];\n"\
"    }\n"\
"}\n"\
"\n"\
"static void prblake2s_hash(const uchar* message, uint len, uchar* hash) {\n"\
"    uint h[8];\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        h[i] = k_iv[i];\n"\
"    }\n"\
"    /* fanout=1, depth=1, keylen=0, digest_length=32 */\n"\
"    h[0] ^= 0x01010000u ^ HASH_LEN;\n"\
"\n"\
"    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;\n"\
"    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];\n"\
"    prblake2s_compress(h, block, (ulong)(len), 1);\n"\
"\n"\
"    for (int i = 0; i < 8; i++) {\n"\
"        hash[i * 4 + 0] = (uchar)(h[i]);\n"\
"        hash[i * 4 + 1] = (uchar)(h[i] >> 8);\n"\
"        hash[i * 4 + 2] = (uchar)(h[i] >> 16);\n"\
"        hash[i * 4 + 3] = (uchar)(h[i] >> 24);\n"\
"    }\n"\
"}\n"\
"\n"\
"static int prblake2s_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[HASH_LEN];\n"\
"  prblake2s_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < HASH_LEN; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prblake2s_kernel(__global uchar* result,\n"\
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
"  if (pass_len >= min_len) {\n"\
"    if (*g_found) return;\n"\
"    if (prblake2s_compare(k_hash, attempt, (int)pass_len)) {\n"\
"      for (uint k = 0; k < pass_len; ++k) result[k] = attempt[k];\n"\
"      result[pass_len] = 0;\n"\
"      *g_found = 1;\n"\
"      return;\n"\
"    }\n"\
"  }\n"\
"  const uint attempt_len = pass_len + 1u;\n"\
"  if (attempt_len < min_len) return;\n"\
"  for (uint i = 0; i < dict_length; ++i) {\n"\
"    attempt[pass_len] = k_dict[i];\n"\
"    if (*g_found) return;\n"\
"    if (prblake2s_compare(k_hash, attempt, (int)attempt_len)) {\n"\
"      for (uint k = 0; k < attempt_len; ++k) result[k] = attempt[k];\n"\
"      result[attempt_len] = 0;\n"\
"      *g_found = 1;\n"\
"      return;\n"\
"    }\n"\
"  }\n"\
"}\n"
