"#define GPU_ATTEMPT_SIZE 16\n"\
"#define BLOCK_LEN 64\n"\
"#define HASH_LEN 16\n"\
"\n"\
"#define F(x, y, z) ((x) ^ (y) ^ (z))\n"\
"#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))\n"\
"#define H(x, y, z) (((x) | ~(y)) ^ (z))\n"\
"#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))\n"\
"#define ROTL32(x, n) (((0U + (x)) << (n)) | ((x) >> (32 - (n))))\n"\
"#define FF(a, b, c, d, x, s) \\\n"\
"    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }\n"\
"#define GG(a, b, c, d, x, s) \\\n"\
"    { (a) += G((b), (c), (d)) + (x) + 0x5a827999u; (a) = ROTL32((a), (s)); }\n"\
"#define HH(a, b, c, d, x, s) \\\n"\
"    { (a) += H((b), (c), (d)) + (x) + 0x6ed9eba1u; (a) = ROTL32((a), (s)); }\n"\
"#define II(a, b, c, d, x, s) \\\n"\
"    { (a) += I((b), (c), (d)) + (x) + 0x8f1bbcdcu; (a) = ROTL32((a), (s)); }\n"\
"#define FFF(a, b, c, d, x, s) \\\n"\
"    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }\n"\
"#define GGG(a, b, c, d, x, s) \\\n"\
"    { (a) += G((b), (c), (d)) + (x) + 0x6d703ef3u; (a) = ROTL32((a), (s)); }\n"\
"#define HHH(a, b, c, d, x, s) \\\n"\
"    { (a) += H((b), (c), (d)) + (x) + 0x5c4dd124u; (a) = ROTL32((a), (s)); }\n"\
"#define III(a, b, c, d, x, s) \\\n"\
"    { (a) += I((b), (c), (d)) + (x) + 0x50a28be6u; (a) = ROTL32((a), (s)); }\n"\
"\n"\
"\n"\
"\n"\
"static void prrmd128_compress(uint* state, const uchar* block) {\n"\
"    uint X[16];\n"\
"    for (int j = 0; j < 16; j++) {\n"\
"        const int i = j * 4;\n"\
"        X[j] = (uint)(block[i + 0])\n"\
"            | ((uint)(block[i + 1]) << 8)\n"\
"            | ((uint)(block[i + 2]) << 16)\n"\
"            | ((uint)(block[i + 3]) << 24);\n"\
"    }\n"\
"\n"\
"    uint aa = state[0], bb = state[1], cc = state[2], dd = state[3];\n"\
"    uint aaa = state[0], bbb = state[1], ccc = state[2], ddd = state[3];\n"\
"\n"\
"    FF(aa, bb, cc, dd, X[0], 11);\n"\
"    FF(dd, aa, bb, cc, X[1], 14);\n"\
"    FF(cc, dd, aa, bb, X[2], 15);\n"\
"    FF(bb, cc, dd, aa, X[3], 12);\n"\
"    FF(aa, bb, cc, dd, X[4], 5);\n"\
"    FF(dd, aa, bb, cc, X[5], 8);\n"\
"    FF(cc, dd, aa, bb, X[6], 7);\n"\
"    FF(bb, cc, dd, aa, X[7], 9);\n"\
"    FF(aa, bb, cc, dd, X[8], 11);\n"\
"    FF(dd, aa, bb, cc, X[9], 13);\n"\
"    FF(cc, dd, aa, bb, X[10], 14);\n"\
"    FF(bb, cc, dd, aa, X[11], 15);\n"\
"    FF(aa, bb, cc, dd, X[12], 6);\n"\
"    FF(dd, aa, bb, cc, X[13], 7);\n"\
"    FF(cc, dd, aa, bb, X[14], 9);\n"\
"    FF(bb, cc, dd, aa, X[15], 8);\n"\
"\n"\
"    GG(aa, bb, cc, dd, X[7], 7);\n"\
"    GG(dd, aa, bb, cc, X[4], 6);\n"\
"    GG(cc, dd, aa, bb, X[13], 8);\n"\
"    GG(bb, cc, dd, aa, X[1], 13);\n"\
"    GG(aa, bb, cc, dd, X[10], 11);\n"\
"    GG(dd, aa, bb, cc, X[6], 9);\n"\
"    GG(cc, dd, aa, bb, X[15], 7);\n"\
"    GG(bb, cc, dd, aa, X[3], 15);\n"\
"    GG(aa, bb, cc, dd, X[12], 7);\n"\
"    GG(dd, aa, bb, cc, X[0], 12);\n"\
"    GG(cc, dd, aa, bb, X[9], 15);\n"\
"    GG(bb, cc, dd, aa, X[5], 9);\n"\
"    GG(aa, bb, cc, dd, X[2], 11);\n"\
"    GG(dd, aa, bb, cc, X[14], 7);\n"\
"    GG(cc, dd, aa, bb, X[11], 13);\n"\
"    GG(bb, cc, dd, aa, X[8], 12);\n"\
"\n"\
"    HH(aa, bb, cc, dd, X[3], 11);\n"\
"    HH(dd, aa, bb, cc, X[10], 13);\n"\
"    HH(cc, dd, aa, bb, X[14], 6);\n"\
"    HH(bb, cc, dd, aa, X[4], 7);\n"\
"    HH(aa, bb, cc, dd, X[9], 14);\n"\
"    HH(dd, aa, bb, cc, X[15], 9);\n"\
"    HH(cc, dd, aa, bb, X[8], 13);\n"\
"    HH(bb, cc, dd, aa, X[1], 15);\n"\
"    HH(aa, bb, cc, dd, X[2], 14);\n"\
"    HH(dd, aa, bb, cc, X[7], 8);\n"\
"    HH(cc, dd, aa, bb, X[0], 13);\n"\
"    HH(bb, cc, dd, aa, X[6], 6);\n"\
"    HH(aa, bb, cc, dd, X[13], 5);\n"\
"    HH(dd, aa, bb, cc, X[11], 12);\n"\
"    HH(cc, dd, aa, bb, X[5], 7);\n"\
"    HH(bb, cc, dd, aa, X[12], 5);\n"\
"\n"\
"    II(aa, bb, cc, dd, X[1], 11);\n"\
"    II(dd, aa, bb, cc, X[9], 12);\n"\
"    II(cc, dd, aa, bb, X[11], 14);\n"\
"    II(bb, cc, dd, aa, X[10], 15);\n"\
"    II(aa, bb, cc, dd, X[0], 14);\n"\
"    II(dd, aa, bb, cc, X[8], 15);\n"\
"    II(cc, dd, aa, bb, X[12], 9);\n"\
"    II(bb, cc, dd, aa, X[4], 8);\n"\
"    II(aa, bb, cc, dd, X[13], 9);\n"\
"    II(dd, aa, bb, cc, X[3], 14);\n"\
"    II(cc, dd, aa, bb, X[7], 5);\n"\
"    II(bb, cc, dd, aa, X[15], 6);\n"\
"    II(aa, bb, cc, dd, X[14], 8);\n"\
"    II(dd, aa, bb, cc, X[5], 6);\n"\
"    II(cc, dd, aa, bb, X[6], 5);\n"\
"    II(bb, cc, dd, aa, X[2], 12);\n"\
"\n"\
"    III(aaa, bbb, ccc, ddd, X[5], 8);\n"\
"    III(ddd, aaa, bbb, ccc, X[14], 9);\n"\
"    III(ccc, ddd, aaa, bbb, X[7], 9);\n"\
"    III(bbb, ccc, ddd, aaa, X[0], 11);\n"\
"    III(aaa, bbb, ccc, ddd, X[9], 13);\n"\
"    III(ddd, aaa, bbb, ccc, X[2], 15);\n"\
"    III(ccc, ddd, aaa, bbb, X[11], 15);\n"\
"    III(bbb, ccc, ddd, aaa, X[4], 5);\n"\
"    III(aaa, bbb, ccc, ddd, X[13], 7);\n"\
"    III(ddd, aaa, bbb, ccc, X[6], 7);\n"\
"    III(ccc, ddd, aaa, bbb, X[15], 8);\n"\
"    III(bbb, ccc, ddd, aaa, X[8], 11);\n"\
"    III(aaa, bbb, ccc, ddd, X[1], 14);\n"\
"    III(ddd, aaa, bbb, ccc, X[10], 14);\n"\
"    III(ccc, ddd, aaa, bbb, X[3], 12);\n"\
"    III(bbb, ccc, ddd, aaa, X[12], 6);\n"\
"\n"\
"    HHH(aaa, bbb, ccc, ddd, X[6], 9);\n"\
"    HHH(ddd, aaa, bbb, ccc, X[11], 13);\n"\
"    HHH(ccc, ddd, aaa, bbb, X[3], 15);\n"\
"    HHH(bbb, ccc, ddd, aaa, X[7], 7);\n"\
"    HHH(aaa, bbb, ccc, ddd, X[0], 12);\n"\
"    HHH(ddd, aaa, bbb, ccc, X[13], 8);\n"\
"    HHH(ccc, ddd, aaa, bbb, X[5], 9);\n"\
"    HHH(bbb, ccc, ddd, aaa, X[10], 11);\n"\
"    HHH(aaa, bbb, ccc, ddd, X[14], 7);\n"\
"    HHH(ddd, aaa, bbb, ccc, X[15], 7);\n"\
"    HHH(ccc, ddd, aaa, bbb, X[8], 12);\n"\
"    HHH(bbb, ccc, ddd, aaa, X[12], 7);\n"\
"    HHH(aaa, bbb, ccc, ddd, X[4], 6);\n"\
"    HHH(ddd, aaa, bbb, ccc, X[9], 15);\n"\
"    HHH(ccc, ddd, aaa, bbb, X[1], 13);\n"\
"    HHH(bbb, ccc, ddd, aaa, X[2], 11);\n"\
"\n"\
"    GGG(aaa, bbb, ccc, ddd, X[15], 9);\n"\
"    GGG(ddd, aaa, bbb, ccc, X[5], 7);\n"\
"    GGG(ccc, ddd, aaa, bbb, X[1], 15);\n"\
"    GGG(bbb, ccc, ddd, aaa, X[3], 11);\n"\
"    GGG(aaa, bbb, ccc, ddd, X[7], 8);\n"\
"    GGG(ddd, aaa, bbb, ccc, X[14], 6);\n"\
"    GGG(ccc, ddd, aaa, bbb, X[6], 6);\n"\
"    GGG(bbb, ccc, ddd, aaa, X[9], 14);\n"\
"    GGG(aaa, bbb, ccc, ddd, X[11], 12);\n"\
"    GGG(ddd, aaa, bbb, ccc, X[8], 13);\n"\
"    GGG(ccc, ddd, aaa, bbb, X[12], 5);\n"\
"    GGG(bbb, ccc, ddd, aaa, X[2], 14);\n"\
"    GGG(aaa, bbb, ccc, ddd, X[10], 13);\n"\
"    GGG(ddd, aaa, bbb, ccc, X[0], 13);\n"\
"    GGG(ccc, ddd, aaa, bbb, X[4], 7);\n"\
"    GGG(bbb, ccc, ddd, aaa, X[13], 5);\n"\
"\n"\
"    FFF(aaa, bbb, ccc, ddd, X[8], 15);\n"\
"    FFF(ddd, aaa, bbb, ccc, X[6], 5);\n"\
"    FFF(ccc, ddd, aaa, bbb, X[4], 8);\n"\
"    FFF(bbb, ccc, ddd, aaa, X[1], 11);\n"\
"    FFF(aaa, bbb, ccc, ddd, X[3], 14);\n"\
"    FFF(ddd, aaa, bbb, ccc, X[11], 14);\n"\
"    FFF(ccc, ddd, aaa, bbb, X[15], 6);\n"\
"    FFF(bbb, ccc, ddd, aaa, X[0], 14);\n"\
"    FFF(aaa, bbb, ccc, ddd, X[5], 6);\n"\
"    FFF(ddd, aaa, bbb, ccc, X[12], 9);\n"\
"    FFF(ccc, ddd, aaa, bbb, X[2], 12);\n"\
"    FFF(bbb, ccc, ddd, aaa, X[13], 9);\n"\
"    FFF(aaa, bbb, ccc, ddd, X[9], 12);\n"\
"    FFF(ddd, aaa, bbb, ccc, X[7], 5);\n"\
"    FFF(ccc, ddd, aaa, bbb, X[10], 15);\n"\
"    FFF(bbb, ccc, ddd, aaa, X[14], 8);\n"\
"\n"\
"    ddd += cc + state[1];\n"\
"    state[1] = state[2] + dd + aaa;\n"\
"    state[2] = state[3] + aa + bbb;\n"\
"    state[3] = state[0] + bb + ccc;\n"\
"    state[0] = ddd;\n"\
"}\n"\
"\n"\
"static void prrmd128_hash(const uchar* message, uint len, uchar* hash) {\n"\
"    uint state[4] = {\n"\
"        0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u\n"\
"    };\n"\
"\n"\
"    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;\n"\
"    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];\n"\
"    block[len] = 0x80;\n"\
"    const ulong bitlen = (ulong)(len) << 3;\n"\
"    block[56] = (uchar)(bitlen);\n"\
"    block[57] = (uchar)(bitlen >> 8);\n"\
"    block[58] = (uchar)(bitlen >> 16);\n"\
"    block[59] = (uchar)(bitlen >> 24);\n"\
"    block[60] = (uchar)(bitlen >> 32);\n"\
"    block[61] = (uchar)(bitlen >> 40);\n"\
"    block[62] = (uchar)(bitlen >> 48);\n"\
"    block[63] = (uchar)(bitlen >> 56);\n"\
"\n"\
"    prrmd128_compress(state, block);\n"\
"\n"\
"    for (int i = 0; i < HASH_LEN; i++)\n"\
"        hash[i] = (uchar)(state[i >> 2] >> ((i & 3) << 3));\n"\
"}\n"\
"\n"\
"static int prrmd128_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[16];\n"\
"  prrmd128_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < 16; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"\n"\
"__kernel void prrmd128_kernel(__global uchar* result,\n"\
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
"    if (prrmd128_compare(k_hash, attempt, (int)pass_len)) {\n"\
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
"    if (prrmd128_compare(k_hash, attempt, (int)attempt_len)) {\n"\
"      for (uint k = 0; k < attempt_len; ++k) result[k] = attempt[k];\n"\
"      result[attempt_len] = 0;\n"\
"      *g_found = 1;\n"\
"      return;\n"\
"    }\n"\
"  }\n"\
"}\n"
