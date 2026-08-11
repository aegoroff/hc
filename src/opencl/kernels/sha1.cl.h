"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 20\n"\
"\n"\
"/* Short-password SHA-1: single block, big-endian words (len <= GPU_ATTEMPT_SIZE). */\n"\
"static int prsha1_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  const uint h0 = (uint)k_hash[3] | (uint)k_hash[2] << 8 | (uint)k_hash[1] << 16 | (uint)k_hash[0] << 24;\n"\
"  const uint h1 = (uint)k_hash[7] | (uint)k_hash[6] << 8 | (uint)k_hash[5] << 16 | (uint)k_hash[4] << 24;\n"\
"  const uint h2 = (uint)k_hash[11] | (uint)k_hash[10] << 8 | (uint)k_hash[9] << 16 | (uint)k_hash[8] << 24;\n"\
"  const uint h3 = (uint)k_hash[15] | (uint)k_hash[14] << 8 | (uint)k_hash[13] << 16 | (uint)k_hash[12] << 24;\n"\
"  const uint h4 = (uint)k_hash[19] | (uint)k_hash[18] << 8 | (uint)k_hash[17] << 16 | (uint)k_hash[16] << 24;\n"\
"\n"\
"  const uint a0 = 0x67452301u;\n"\
"  const uint b0 = 0xEFCDAB89u;\n"\
"  const uint c0 = 0x98BADCFEu;\n"\
"  const uint d0 = 0x10325476u;\n"\
"  const uint e0 = 0xC3D2E1F0u;\n"\
"\n"\
"  uint schedule[16];\n"\
"  int i;\n"\
"  for (i = 0; i < 16; ++i) schedule[i] = 0;\n"\
"  for (i = 0; i < length; ++i)\n"\
"    schedule[i / 4] |= ((uint)password[i]) << (24 - (i % 4) * 8);\n"\
"  schedule[i / 4] |= 0x80u << (24 - (i % 4) * 8);\n"\
"  schedule[15] = (uint)length * 8u;\n"\
"\n"\
"#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))\n"\
"#define SCHEDULE(i)  \\\n"\
"                temp = schedule[(i - 3) & 0xF] ^ schedule[(i - 8) & 0xF] ^ schedule[(i - 14) & 0xF] ^ schedule[(i - 16) & 0xF];  \\\n"\
"                schedule[i & 0xF] = ROTL32(temp, 1);\n"\
"#define ROUND0(a, b, c, d, e, i)  ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999u)\n"\
"#define ROUND0b(a, b, c, d, e, i) SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999u)\n"\
"#define ROUND1(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0x6ED9EBA1u)\n"\
"#define ROUND2(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDCu)\n"\
"#define ROUND3(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0xCA62C1D6u)\n"\
"#define ROUNDTAIL(a, b, e, f, i, k)  \\\n"\
"                e = 0U + e + ROTL32(a, 5) + f + (uint)(k) + schedule[i & 0xF];  \\\n"\
"                b = ROTL32(b, 30);\n"\
"\n"\
"  uint a = a0, b = b0, c = c0, d = d0, e = e0;\n"\
"  uint temp;\n"\
"  ROUND0(a, b, c, d, e, 0)\n"\
"  ROUND0(e, a, b, c, d, 1)\n"\
"  ROUND0(d, e, a, b, c, 2)\n"\
"  ROUND0(c, d, e, a, b, 3)\n"\
"  ROUND0(b, c, d, e, a, 4)\n"\
"  ROUND0(a, b, c, d, e, 5)\n"\
"  ROUND0(e, a, b, c, d, 6)\n"\
"  ROUND0(d, e, a, b, c, 7)\n"\
"  ROUND0(c, d, e, a, b, 8)\n"\
"  ROUND0(b, c, d, e, a, 9)\n"\
"  ROUND0(a, b, c, d, e, 10)\n"\
"  ROUND0(e, a, b, c, d, 11)\n"\
"  ROUND0(d, e, a, b, c, 12)\n"\
"  ROUND0(c, d, e, a, b, 13)\n"\
"  ROUND0(b, c, d, e, a, 14)\n"\
"  ROUND0(a, b, c, d, e, 15)\n"\
"  ROUND0b(e, a, b, c, d, 16)\n"\
"  ROUND0b(d, e, a, b, c, 17)\n"\
"  ROUND0b(c, d, e, a, b, 18)\n"\
"  ROUND0b(b, c, d, e, a, 19)\n"\
"  ROUND1(a, b, c, d, e, 20)\n"\
"  ROUND1(e, a, b, c, d, 21)\n"\
"  ROUND1(d, e, a, b, c, 22)\n"\
"  ROUND1(c, d, e, a, b, 23)\n"\
"  ROUND1(b, c, d, e, a, 24)\n"\
"  ROUND1(a, b, c, d, e, 25)\n"\
"  ROUND1(e, a, b, c, d, 26)\n"\
"  ROUND1(d, e, a, b, c, 27)\n"\
"  ROUND1(c, d, e, a, b, 28)\n"\
"  ROUND1(b, c, d, e, a, 29)\n"\
"  ROUND1(a, b, c, d, e, 30)\n"\
"  ROUND1(e, a, b, c, d, 31)\n"\
"  ROUND1(d, e, a, b, c, 32)\n"\
"  ROUND1(c, d, e, a, b, 33)\n"\
"  ROUND1(b, c, d, e, a, 34)\n"\
"  ROUND1(a, b, c, d, e, 35)\n"\
"  ROUND1(e, a, b, c, d, 36)\n"\
"  ROUND1(d, e, a, b, c, 37)\n"\
"  ROUND1(c, d, e, a, b, 38)\n"\
"  ROUND1(b, c, d, e, a, 39)\n"\
"  ROUND2(a, b, c, d, e, 40)\n"\
"  ROUND2(e, a, b, c, d, 41)\n"\
"  ROUND2(d, e, a, b, c, 42)\n"\
"  ROUND2(c, d, e, a, b, 43)\n"\
"  ROUND2(b, c, d, e, a, 44)\n"\
"  ROUND2(a, b, c, d, e, 45)\n"\
"  ROUND2(e, a, b, c, d, 46)\n"\
"  ROUND2(d, e, a, b, c, 47)\n"\
"  ROUND2(c, d, e, a, b, 48)\n"\
"  ROUND2(b, c, d, e, a, 49)\n"\
"  ROUND2(a, b, c, d, e, 50)\n"\
"  ROUND2(e, a, b, c, d, 51)\n"\
"  ROUND2(d, e, a, b, c, 52)\n"\
"  ROUND2(c, d, e, a, b, 53)\n"\
"  ROUND2(b, c, d, e, a, 54)\n"\
"  ROUND2(a, b, c, d, e, 55)\n"\
"  ROUND2(e, a, b, c, d, 56)\n"\
"  ROUND2(d, e, a, b, c, 57)\n"\
"  ROUND2(c, d, e, a, b, 58)\n"\
"  ROUND2(b, c, d, e, a, 59)\n"\
"  ROUND3(a, b, c, d, e, 60)\n"\
"  ROUND3(e, a, b, c, d, 61)\n"\
"  ROUND3(d, e, a, b, c, 62)\n"\
"  ROUND3(c, d, e, a, b, 63)\n"\
"  ROUND3(b, c, d, e, a, 64)\n"\
"  ROUND3(a, b, c, d, e, 65)\n"\
"  ROUND3(e, a, b, c, d, 66)\n"\
"  ROUND3(d, e, a, b, c, 67)\n"\
"  ROUND3(c, d, e, a, b, 68)\n"\
"  ROUND3(b, c, d, e, a, 69)\n"\
"  ROUND3(a, b, c, d, e, 70)\n"\
"  ROUND3(e, a, b, c, d, 71)\n"\
"  ROUND3(d, e, a, b, c, 72)\n"\
"  ROUND3(c, d, e, a, b, 73)\n"\
"  ROUND3(b, c, d, e, a, 74)\n"\
"  ROUND3(a, b, c, d, e, 75)\n"\
"  ROUND3(e, a, b, c, d, 76)\n"\
"  ROUND3(d, e, a, b, c, 77)\n"\
"  ROUND3(c, d, e, a, b, 78)\n"\
"  ROUND3(b, c, d, e, a, 79)\n"\
"\n"\
"  a = 0U + a0 + a;\n"\
"  b = 0U + b0 + b;\n"\
"  c = 0U + c0 + c;\n"\
"  d = 0U + d0 + d;\n"\
"  e = 0U + e0 + e;\n"\
"\n"\
"#undef ROUNDTAIL\n"\
"#undef ROUND3\n"\
"#undef ROUND2\n"\
"#undef ROUND1\n"\
"#undef ROUND0b\n"\
"#undef ROUND0\n"\
"#undef SCHEDULE\n"\
"#undef ROTL32\n"\
"\n"\
"  return a == h0 && b == h1 && c == h2 && d == h3 && e == h4;\n"\
"}\n"\
"\n"\
"\n"\
"__kernel void prsha1_kernel(__global uchar* result,\n"\
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
"      if (prsha1_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prsha1_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
