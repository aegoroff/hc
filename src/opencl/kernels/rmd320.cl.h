"#define GPU_ATTEMPT_SIZE 16\n"\
"#define BLOCK_LEN 64\n"\
"#define HASH_LEN 40\n"\
"\n"\
"#define F(x, y, z) ((x) ^ (y) ^ (z))\n"\
"#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))\n"\
"#define H(x, y, z) (((x) | ~(y)) ^ (z))\n"\
"#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))\n"\
"#define J(x, y, z) ((x) ^ ((y) | ~(z)))\n"\
"#define ROTL32(x, n) (((0U + (x)) << (n)) | ((x) >> (32 - (n))))\n"\
"#define FF(a, b, c, d, e, x, s) \\\n"\
"    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define GG(a, b, c, d, e, x, s) \\\n"\
"    { (a) += G((b), (c), (d)) + (x) + 0x5a827999u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define HH(a, b, c, d, e, x, s) \\\n"\
"    { (a) += H((b), (c), (d)) + (x) + 0x6ed9eba1u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define II(a, b, c, d, e, x, s) \\\n"\
"    { (a) += I((b), (c), (d)) + (x) + 0x8f1bbcdcu; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define JJ(a, b, c, d, e, x, s) \\\n"\
"    { (a) += J((b), (c), (d)) + (x) + 0xa953fd4eu; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define FFF(a, b, c, d, e, x, s) \\\n"\
"    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define GGG(a, b, c, d, e, x, s) \\\n"\
"    { (a) += G((b), (c), (d)) + (x) + 0x7a6d76e9u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define HHH(a, b, c, d, e, x, s) \\\n"\
"    { (a) += H((b), (c), (d)) + (x) + 0x6d703ef3u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define III(a, b, c, d, e, x, s) \\\n"\
"    { (a) += I((b), (c), (d)) + (x) + 0x5c4dd124u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"#define JJJ(a, b, c, d, e, x, s) \\\n"\
"    { (a) += J((b), (c), (d)) + (x) + 0x50a28be6u; (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }\n"\
"\n"\
"\n"\
"\n"\
"static void prrmd320_compress(uint* state, const uchar* block) {\n"\
"    uint X[16];\n"\
"    for (int j = 0; j < 16; j++) {\n"\
"        const int i = j * 4;\n"\
"        X[j] = (uint)(block[i + 0])\n"\
"            | ((uint)(block[i + 1]) << 8)\n"\
"            | ((uint)(block[i + 2]) << 16)\n"\
"            | ((uint)(block[i + 3]) << 24);\n"\
"    }\n"\
"\n"\
"    uint aa = state[0], bb = state[1], cc = state[2], dd = state[3], ee = state[4];\n"\
"    uint aaa = state[5], bbb = state[6], ccc = state[7], ddd = state[8], eee = state[9];\n"\
"    uint tmp;\n"\
"\n"\
"    FF(aa, bb, cc, dd, ee, X[ 0], 11);\n"\
"    FF(ee, aa, bb, cc, dd, X[ 1], 14);\n"\
"    FF(dd, ee, aa, bb, cc, X[ 2], 15);\n"\
"    FF(cc, dd, ee, aa, bb, X[ 3], 12);\n"\
"    FF(bb, cc, dd, ee, aa, X[ 4],  5);\n"\
"    FF(aa, bb, cc, dd, ee, X[ 5],  8);\n"\
"    FF(ee, aa, bb, cc, dd, X[ 6],  7);\n"\
"    FF(dd, ee, aa, bb, cc, X[ 7],  9);\n"\
"    FF(cc, dd, ee, aa, bb, X[ 8], 11);\n"\
"    FF(bb, cc, dd, ee, aa, X[ 9], 13);\n"\
"    FF(aa, bb, cc, dd, ee, X[10], 14);\n"\
"    FF(ee, aa, bb, cc, dd, X[11], 15);\n"\
"    FF(dd, ee, aa, bb, cc, X[12],  6);\n"\
"    FF(cc, dd, ee, aa, bb, X[13],  7);\n"\
"    FF(bb, cc, dd, ee, aa, X[14],  9);\n"\
"    FF(aa, bb, cc, dd, ee, X[15],  8);\n"\
"    JJJ(aaa, bbb, ccc, ddd, eee, X[ 5],  8);\n"\
"    JJJ(eee, aaa, bbb, ccc, ddd, X[14],  9);\n"\
"    JJJ(ddd, eee, aaa, bbb, ccc, X[ 7],  9);\n"\
"    JJJ(ccc, ddd, eee, aaa, bbb, X[ 0], 11);\n"\
"    JJJ(bbb, ccc, ddd, eee, aaa, X[ 9], 13);\n"\
"    JJJ(aaa, bbb, ccc, ddd, eee, X[ 2], 15);\n"\
"    JJJ(eee, aaa, bbb, ccc, ddd, X[11], 15);\n"\
"    JJJ(ddd, eee, aaa, bbb, ccc, X[ 4],  5);\n"\
"    JJJ(ccc, ddd, eee, aaa, bbb, X[13],  7);\n"\
"    JJJ(bbb, ccc, ddd, eee, aaa, X[ 6],  7);\n"\
"    JJJ(aaa, bbb, ccc, ddd, eee, X[15],  8);\n"\
"    JJJ(eee, aaa, bbb, ccc, ddd, X[ 8], 11);\n"\
"    JJJ(ddd, eee, aaa, bbb, ccc, X[ 1], 14);\n"\
"    JJJ(ccc, ddd, eee, aaa, bbb, X[10], 14);\n"\
"    JJJ(bbb, ccc, ddd, eee, aaa, X[ 3], 12);\n"\
"    JJJ(aaa, bbb, ccc, ddd, eee, X[12],  6);\n"\
"    tmp = aa; aa = aaa; aaa = tmp;\n"\
"    GG(ee, aa, bb, cc, dd, X[ 7],  7);\n"\
"    GG(dd, ee, aa, bb, cc, X[ 4],  6);\n"\
"    GG(cc, dd, ee, aa, bb, X[13],  8);\n"\
"    GG(bb, cc, dd, ee, aa, X[ 1], 13);\n"\
"    GG(aa, bb, cc, dd, ee, X[10], 11);\n"\
"    GG(ee, aa, bb, cc, dd, X[ 6],  9);\n"\
"    GG(dd, ee, aa, bb, cc, X[15],  7);\n"\
"    GG(cc, dd, ee, aa, bb, X[ 3], 15);\n"\
"    GG(bb, cc, dd, ee, aa, X[12],  7);\n"\
"    GG(aa, bb, cc, dd, ee, X[ 0], 12);\n"\
"    GG(ee, aa, bb, cc, dd, X[ 9], 15);\n"\
"    GG(dd, ee, aa, bb, cc, X[ 5],  9);\n"\
"    GG(cc, dd, ee, aa, bb, X[ 2], 11);\n"\
"    GG(bb, cc, dd, ee, aa, X[14],  7);\n"\
"    GG(aa, bb, cc, dd, ee, X[11], 13);\n"\
"    GG(ee, aa, bb, cc, dd, X[ 8], 12);\n"\
"    III(eee, aaa, bbb, ccc, ddd, X[ 6],  9);\n"\
"    III(ddd, eee, aaa, bbb, ccc, X[11], 13);\n"\
"    III(ccc, ddd, eee, aaa, bbb, X[ 3], 15);\n"\
"    III(bbb, ccc, ddd, eee, aaa, X[ 7],  7);\n"\
"    III(aaa, bbb, ccc, ddd, eee, X[ 0], 12);\n"\
"    III(eee, aaa, bbb, ccc, ddd, X[13],  8);\n"\
"    III(ddd, eee, aaa, bbb, ccc, X[ 5],  9);\n"\
"    III(ccc, ddd, eee, aaa, bbb, X[10], 11);\n"\
"    III(bbb, ccc, ddd, eee, aaa, X[14],  7);\n"\
"    III(aaa, bbb, ccc, ddd, eee, X[15],  7);\n"\
"    III(eee, aaa, bbb, ccc, ddd, X[ 8], 12);\n"\
"    III(ddd, eee, aaa, bbb, ccc, X[12],  7);\n"\
"    III(ccc, ddd, eee, aaa, bbb, X[ 4],  6);\n"\
"    III(bbb, ccc, ddd, eee, aaa, X[ 9], 15);\n"\
"    III(aaa, bbb, ccc, ddd, eee, X[ 1], 13);\n"\
"    III(eee, aaa, bbb, ccc, ddd, X[ 2], 11);\n"\
"    tmp = bb; bb = bbb; bbb = tmp;\n"\
"    HH(dd, ee, aa, bb, cc, X[ 3], 11);\n"\
"    HH(cc, dd, ee, aa, bb, X[10], 13);\n"\
"    HH(bb, cc, dd, ee, aa, X[14],  6);\n"\
"    HH(aa, bb, cc, dd, ee, X[ 4],  7);\n"\
"    HH(ee, aa, bb, cc, dd, X[ 9], 14);\n"\
"    HH(dd, ee, aa, bb, cc, X[15],  9);\n"\
"    HH(cc, dd, ee, aa, bb, X[ 8], 13);\n"\
"    HH(bb, cc, dd, ee, aa, X[ 1], 15);\n"\
"    HH(aa, bb, cc, dd, ee, X[ 2], 14);\n"\
"    HH(ee, aa, bb, cc, dd, X[ 7],  8);\n"\
"    HH(dd, ee, aa, bb, cc, X[ 0], 13);\n"\
"    HH(cc, dd, ee, aa, bb, X[ 6],  6);\n"\
"    HH(bb, cc, dd, ee, aa, X[13],  5);\n"\
"    HH(aa, bb, cc, dd, ee, X[11], 12);\n"\
"    HH(ee, aa, bb, cc, dd, X[ 5],  7);\n"\
"    HH(dd, ee, aa, bb, cc, X[12],  5);\n"\
"    HHH(ddd, eee, aaa, bbb, ccc, X[15],  9);\n"\
"    HHH(ccc, ddd, eee, aaa, bbb, X[ 5],  7);\n"\
"    HHH(bbb, ccc, ddd, eee, aaa, X[ 1], 15);\n"\
"    HHH(aaa, bbb, ccc, ddd, eee, X[ 3], 11);\n"\
"    HHH(eee, aaa, bbb, ccc, ddd, X[ 7],  8);\n"\
"    HHH(ddd, eee, aaa, bbb, ccc, X[14],  6);\n"\
"    HHH(ccc, ddd, eee, aaa, bbb, X[ 6],  6);\n"\
"    HHH(bbb, ccc, ddd, eee, aaa, X[ 9], 14);\n"\
"    HHH(aaa, bbb, ccc, ddd, eee, X[11], 12);\n"\
"    HHH(eee, aaa, bbb, ccc, ddd, X[ 8], 13);\n"\
"    HHH(ddd, eee, aaa, bbb, ccc, X[12],  5);\n"\
"    HHH(ccc, ddd, eee, aaa, bbb, X[ 2], 14);\n"\
"    HHH(bbb, ccc, ddd, eee, aaa, X[10], 13);\n"\
"    HHH(aaa, bbb, ccc, ddd, eee, X[ 0], 13);\n"\
"    HHH(eee, aaa, bbb, ccc, ddd, X[ 4],  7);\n"\
"    HHH(ddd, eee, aaa, bbb, ccc, X[13],  5);\n"\
"    tmp = cc; cc = ccc; ccc = tmp;\n"\
"    II(cc, dd, ee, aa, bb, X[ 1], 11);\n"\
"    II(bb, cc, dd, ee, aa, X[ 9], 12);\n"\
"    II(aa, bb, cc, dd, ee, X[11], 14);\n"\
"    II(ee, aa, bb, cc, dd, X[10], 15);\n"\
"    II(dd, ee, aa, bb, cc, X[ 0], 14);\n"\
"    II(cc, dd, ee, aa, bb, X[ 8], 15);\n"\
"    II(bb, cc, dd, ee, aa, X[12],  9);\n"\
"    II(aa, bb, cc, dd, ee, X[ 4],  8);\n"\
"    II(ee, aa, bb, cc, dd, X[13],  9);\n"\
"    II(dd, ee, aa, bb, cc, X[ 3], 14);\n"\
"    II(cc, dd, ee, aa, bb, X[ 7],  5);\n"\
"    II(bb, cc, dd, ee, aa, X[15],  6);\n"\
"    II(aa, bb, cc, dd, ee, X[14],  8);\n"\
"    II(ee, aa, bb, cc, dd, X[ 5],  6);\n"\
"    II(dd, ee, aa, bb, cc, X[ 6],  5);\n"\
"    II(cc, dd, ee, aa, bb, X[ 2], 12);\n"\
"    GGG(ccc, ddd, eee, aaa, bbb, X[ 8], 15);\n"\
"    GGG(bbb, ccc, ddd, eee, aaa, X[ 6],  5);\n"\
"    GGG(aaa, bbb, ccc, ddd, eee, X[ 4],  8);\n"\
"    GGG(eee, aaa, bbb, ccc, ddd, X[ 1], 11);\n"\
"    GGG(ddd, eee, aaa, bbb, ccc, X[ 3], 14);\n"\
"    GGG(ccc, ddd, eee, aaa, bbb, X[11], 14);\n"\
"    GGG(bbb, ccc, ddd, eee, aaa, X[15],  6);\n"\
"    GGG(aaa, bbb, ccc, ddd, eee, X[ 0], 14);\n"\
"    GGG(eee, aaa, bbb, ccc, ddd, X[ 5],  6);\n"\
"    GGG(ddd, eee, aaa, bbb, ccc, X[12],  9);\n"\
"    GGG(ccc, ddd, eee, aaa, bbb, X[ 2], 12);\n"\
"    GGG(bbb, ccc, ddd, eee, aaa, X[13],  9);\n"\
"    GGG(aaa, bbb, ccc, ddd, eee, X[ 9], 12);\n"\
"    GGG(eee, aaa, bbb, ccc, ddd, X[ 7],  5);\n"\
"    GGG(ddd, eee, aaa, bbb, ccc, X[10], 15);\n"\
"    GGG(ccc, ddd, eee, aaa, bbb, X[14],  8);\n"\
"    tmp = dd; dd = ddd; ddd = tmp;\n"\
"    JJ(bb, cc, dd, ee, aa, X[ 4],  9);\n"\
"    JJ(aa, bb, cc, dd, ee, X[ 0], 15);\n"\
"    JJ(ee, aa, bb, cc, dd, X[ 5],  5);\n"\
"    JJ(dd, ee, aa, bb, cc, X[ 9], 11);\n"\
"    JJ(cc, dd, ee, aa, bb, X[ 7],  6);\n"\
"    JJ(bb, cc, dd, ee, aa, X[12],  8);\n"\
"    JJ(aa, bb, cc, dd, ee, X[ 2], 13);\n"\
"    JJ(ee, aa, bb, cc, dd, X[10], 12);\n"\
"    JJ(dd, ee, aa, bb, cc, X[14],  5);\n"\
"    JJ(cc, dd, ee, aa, bb, X[ 1], 12);\n"\
"    JJ(bb, cc, dd, ee, aa, X[ 3], 13);\n"\
"    JJ(aa, bb, cc, dd, ee, X[ 8], 14);\n"\
"    JJ(ee, aa, bb, cc, dd, X[11], 11);\n"\
"    JJ(dd, ee, aa, bb, cc, X[ 6],  8);\n"\
"    JJ(cc, dd, ee, aa, bb, X[15],  5);\n"\
"    JJ(bb, cc, dd, ee, aa, X[13],  6);\n"\
"    FFF(bbb, ccc, ddd, eee, aaa, X[12] ,  8);\n"\
"    FFF(aaa, bbb, ccc, ddd, eee, X[15] ,  5);\n"\
"    FFF(eee, aaa, bbb, ccc, ddd, X[10] , 12);\n"\
"    FFF(ddd, eee, aaa, bbb, ccc, X[ 4] ,  9);\n"\
"    FFF(ccc, ddd, eee, aaa, bbb, X[ 1] , 12);\n"\
"    FFF(bbb, ccc, ddd, eee, aaa, X[ 5] ,  5);\n"\
"    FFF(aaa, bbb, ccc, ddd, eee, X[ 8] , 14);\n"\
"    FFF(eee, aaa, bbb, ccc, ddd, X[ 7] ,  6);\n"\
"    FFF(ddd, eee, aaa, bbb, ccc, X[ 6] ,  8);\n"\
"    FFF(ccc, ddd, eee, aaa, bbb, X[ 2] , 13);\n"\
"    FFF(bbb, ccc, ddd, eee, aaa, X[13] ,  6);\n"\
"    FFF(aaa, bbb, ccc, ddd, eee, X[14] ,  5);\n"\
"    FFF(eee, aaa, bbb, ccc, ddd, X[ 0] , 15);\n"\
"    FFF(ddd, eee, aaa, bbb, ccc, X[ 3] , 13);\n"\
"    FFF(ccc, ddd, eee, aaa, bbb, X[ 9] , 11);\n"\
"    FFF(bbb, ccc, ddd, eee, aaa, X[11] , 11);\n"\
"    tmp = ee; ee = eee; eee = tmp;\n"\
"\n"\
"    state[0] += aa;\n"\
"    state[1] += bb;\n"\
"    state[2] += cc;\n"\
"    state[3] += dd;\n"\
"    state[4] += ee;\n"\
"    state[5] += aaa;\n"\
"    state[6] += bbb;\n"\
"    state[7] += ccc;\n"\
"    state[8] += ddd;\n"\
"    state[9] += eee;\n"\
"}\n"\
"\n"\
"static void prrmd320_hash(const uchar* message, uint len, uchar* hash) {\n"\
"    uint state[10] = {\n"\
"        0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u,\n"\
"        0x76543210u, 0xFEDCBA98u, 0x89ABCDEFu, 0x01234567u, 0x3C2D1E0Fu,\n"\
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
"    prrmd320_compress(state, block);\n"\
"\n"\
"    for (int i = 0; i < HASH_LEN; i++)\n"\
"        hash[i] = (uchar)(state[i >> 2] >> ((i & 3) << 3));\n"\
"}\n"\
"\n"\
"static int prrmd320_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[40];\n"\
"  prrmd320_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < 40; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"\n"\
"__kernel void prrmd320_kernel(__global uchar* result,\n"\
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
"      if (prrmd320_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prrmd320_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
