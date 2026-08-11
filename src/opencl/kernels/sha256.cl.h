"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 32\n"\
"#define BLOCK_LEN 64\n"\
"#define STATE_LEN 8\n"\
"#define LENGTH_SIZE 8\n"\
"\n"\
"static void prsha256_compress(uint state[], const uchar block[]) {\n"\
"#define ROTR32(x, n)  (((0U + (x)) << (32 - (n))) | ((x) >> (n)))\n"\
"#define LOADSCHEDULE(i)  \\\n"\
"                schedule[i] = (uint)block[i * 4 + 0] << 24  \\\n"\
"                            | (uint)block[i * 4 + 1] << 16  \\\n"\
"                            | (uint)block[i * 4 + 2] <<  8  \\\n"\
"                            | (uint)block[i * 4 + 3] <<  0;\n"\
"#define SCHEDULE(i)  \\\n"\
"                schedule[i] = 0U + schedule[i - 16] + schedule[i - 7]  \\\n"\
"                        + (ROTR32(schedule[i - 15], 7) ^ ROTR32(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3))  \\\n"\
"                        + (ROTR32(schedule[i - 2], 17) ^ ROTR32(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10));\n"\
"#define ROUND(a, b, c, d, e, f, g, h, i, k) \\\n"\
"                h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + (uint)(k) + schedule[i];  \\\n"\
"                d = 0U + d + h;  \\\n"\
"                h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));\n"\
"  uint schedule[64];\n"\
"  LOADSCHEDULE(0)\n"\
"  LOADSCHEDULE(1)\n"\
"  LOADSCHEDULE(2)\n"\
"  LOADSCHEDULE(3)\n"\
"  LOADSCHEDULE(4)\n"\
"  LOADSCHEDULE(5)\n"\
"  LOADSCHEDULE(6)\n"\
"  LOADSCHEDULE(7)\n"\
"  LOADSCHEDULE(8)\n"\
"  LOADSCHEDULE(9)\n"\
"  LOADSCHEDULE(10)\n"\
"  LOADSCHEDULE(11)\n"\
"  LOADSCHEDULE(12)\n"\
"  LOADSCHEDULE(13)\n"\
"  LOADSCHEDULE(14)\n"\
"  LOADSCHEDULE(15)\n"\
"  SCHEDULE(16)\n"\
"  SCHEDULE(17)\n"\
"  SCHEDULE(18)\n"\
"  SCHEDULE(19)\n"\
"  SCHEDULE(20)\n"\
"  SCHEDULE(21)\n"\
"  SCHEDULE(22)\n"\
"  SCHEDULE(23)\n"\
"  SCHEDULE(24)\n"\
"  SCHEDULE(25)\n"\
"  SCHEDULE(26)\n"\
"  SCHEDULE(27)\n"\
"  SCHEDULE(28)\n"\
"  SCHEDULE(29)\n"\
"  SCHEDULE(30)\n"\
"  SCHEDULE(31)\n"\
"  SCHEDULE(32)\n"\
"  SCHEDULE(33)\n"\
"  SCHEDULE(34)\n"\
"  SCHEDULE(35)\n"\
"  SCHEDULE(36)\n"\
"  SCHEDULE(37)\n"\
"  SCHEDULE(38)\n"\
"  SCHEDULE(39)\n"\
"  SCHEDULE(40)\n"\
"  SCHEDULE(41)\n"\
"  SCHEDULE(42)\n"\
"  SCHEDULE(43)\n"\
"  SCHEDULE(44)\n"\
"  SCHEDULE(45)\n"\
"  SCHEDULE(46)\n"\
"  SCHEDULE(47)\n"\
"  SCHEDULE(48)\n"\
"  SCHEDULE(49)\n"\
"  SCHEDULE(50)\n"\
"  SCHEDULE(51)\n"\
"  SCHEDULE(52)\n"\
"  SCHEDULE(53)\n"\
"  SCHEDULE(54)\n"\
"  SCHEDULE(55)\n"\
"  SCHEDULE(56)\n"\
"  SCHEDULE(57)\n"\
"  SCHEDULE(58)\n"\
"  SCHEDULE(59)\n"\
"  SCHEDULE(60)\n"\
"  SCHEDULE(61)\n"\
"  SCHEDULE(62)\n"\
"  SCHEDULE(63)\n"\
"  uint a = state[0];\n"\
"  uint b = state[1];\n"\
"  uint c = state[2];\n"\
"  uint d = state[3];\n"\
"  uint e = state[4];\n"\
"  uint f = state[5];\n"\
"  uint g = state[6];\n"\
"  uint h = state[7];\n"\
"  ROUND(a, b, c, d, e, f, g, h, 0, 0x428A2F98u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 1, 0x71374491u)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 2, 0xB5C0FBCFu)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 3, 0xE9B5DBA5u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 4, 0x3956C25Bu)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 5, 0x59F111F1u)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 6, 0x923F82A4u)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 7, 0xAB1C5ED5u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 8, 0xD807AA98u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 9, 0x12835B01u)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 10, 0x243185BEu)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 11, 0x550C7DC3u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 12, 0x72BE5D74u)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 13, 0x80DEB1FEu)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 14, 0x9BDC06A7u)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 15, 0xC19BF174u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 16, 0xE49B69C1u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 17, 0xEFBE4786u)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 18, 0x0FC19DC6u)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 19, 0x240CA1CCu)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 20, 0x2DE92C6Fu)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 21, 0x4A7484AAu)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 22, 0x5CB0A9DCu)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 23, 0x76F988DAu)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 24, 0x983E5152u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 25, 0xA831C66Du)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 26, 0xB00327C8u)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 27, 0xBF597FC7u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 28, 0xC6E00BF3u)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 29, 0xD5A79147u)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 30, 0x06CA6351u)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 31, 0x14292967u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 32, 0x27B70A85u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 33, 0x2E1B2138u)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 34, 0x4D2C6DFCu)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 35, 0x53380D13u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 36, 0x650A7354u)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 37, 0x766A0ABBu)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 38, 0x81C2C92Eu)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 39, 0x92722C85u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 40, 0xA2BFE8A1u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 41, 0xA81A664Bu)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 42, 0xC24B8B70u)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 43, 0xC76C51A3u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 44, 0xD192E819u)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 45, 0xD6990624u)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 46, 0xF40E3585u)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 47, 0x106AA070u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 48, 0x19A4C116u)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 49, 0x1E376C08u)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 50, 0x2748774Cu)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 51, 0x34B0BCB5u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 52, 0x391C0CB3u)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 53, 0x4ED8AA4Au)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 54, 0x5B9CCA4Fu)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 55, 0x682E6FF3u)\n"\
"  ROUND(a, b, c, d, e, f, g, h, 56, 0x748F82EEu)\n"\
"  ROUND(h, a, b, c, d, e, f, g, 57, 0x78A5636Fu)\n"\
"  ROUND(g, h, a, b, c, d, e, f, 58, 0x84C87814u)\n"\
"  ROUND(f, g, h, a, b, c, d, e, 59, 0x8CC70208u)\n"\
"  ROUND(e, f, g, h, a, b, c, d, 60, 0x90BEFFFAu)\n"\
"  ROUND(d, e, f, g, h, a, b, c, 61, 0xA4506CEBu)\n"\
"  ROUND(c, d, e, f, g, h, a, b, 62, 0xBEF9A3F7u)\n"\
"  ROUND(b, c, d, e, f, g, h, a, 63, 0xC67178F2u)\n"\
"  state[0] = 0U + state[0] + a;\n"\
"  state[1] = 0U + state[1] + b;\n"\
"  state[2] = 0U + state[2] + c;\n"\
"  state[3] = 0U + state[3] + d;\n"\
"  state[4] = 0U + state[4] + e;\n"\
"  state[5] = 0U + state[5] + f;\n"\
"  state[6] = 0U + state[6] + g;\n"\
"  state[7] = 0U + state[7] + h;\n"\
"#undef ROUND\n"\
"#undef SCHEDULE\n"\
"#undef LOADSCHEDULE\n"\
"#undef ROTR32\n"\
"}\n"\
"\n"\
"\n"\
"static void prsha256_hash(const uchar* message, uint len, uint* hash) {\n"\
"  hash[0] = 0x6A09E667u;\n"\
"  hash[1] = 0xBB67AE85u;\n"\
"  hash[2] = 0x3C6EF372u;\n"\
"  hash[3] = 0xA54FF53Au;\n"\
"  hash[4] = 0x510E527Fu;\n"\
"  hash[5] = 0x9B05688Cu;\n"\
"  hash[6] = 0x1F83D9ABu;\n"\
"  hash[7] = 0x5BE0CD19u;\n"\
"  uint off;\n"\
"  for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)\n"\
"    prsha256_compress(hash, &message[off]);\n"\
"  uchar block[BLOCK_LEN];\n"\
"  uint i;\n"\
"  for (i = 0; i < BLOCK_LEN; ++i) block[i] = 0;\n"\
"  uint rem = len - off;\n"\
"  for (i = 0; i < rem; ++i) block[i] = message[off + i];\n"\
"  block[rem] = 0x80;\n"\
"  rem++;\n"\
"  if (BLOCK_LEN - rem < LENGTH_SIZE) {\n"\
"    prsha256_compress(hash, block);\n"\
"    for (i = 0; i < BLOCK_LEN; ++i) block[i] = 0;\n"\
"  }\n"\
"  uint bitlen = len;\n"\
"  block[BLOCK_LEN - 1] = (uchar)((bitlen & 0x1FU) << 3);\n"\
"  bitlen >>= 5;\n"\
"  for (i = 1; i < LENGTH_SIZE; i++, bitlen >>= 8)\n"\
"    block[BLOCK_LEN - 1 - i] = (uchar)(bitlen & 0xFFU);\n"\
"  prsha256_compress(hash, block);\n"\
"}\n"\
"\n"\
"\n"\
"static int prsha256_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uint hash[STATE_LEN];\n"\
"  prsha256_hash(password, (uint)length, hash);\n"\
"  int result = 1;\n"\
"  for (int i = 0; i < STATE_LEN && result; ++i) {\n"\
"    result &= hash[i] == ((uint)k_hash[3 + i * 4] | (uint)k_hash[2 + i * 4] << 8 | (uint)k_hash[1 + i * 4] << 16 | (uint)k_hash[0 + i * 4] << 24);\n"\
"  }\n"\
"  return result;\n"\
"}\n"\
"\n"\
"\n"\
"__kernel void prsha256_kernel(__global uchar* result,\n"\
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
"      if (prsha256_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prsha256_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
"\n"\
