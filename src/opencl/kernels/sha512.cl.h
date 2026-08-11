"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 64\n"\
"#define BLOCK_LEN 128\n"\
"#define STATE_LEN 8\n"\
"#define LENGTH_SIZE 16\n"\
"\n"\
"static void prsha512_compress(ulong state[], const uchar block[]) {\n"\
"#define ROTR64(x, n)  (((0UL + (x)) << (64 - (n))) | ((x) >> (n)))\n"\
"#define LOADSCHEDULE(i)  \\\n"\
"                schedule[i] = (ulong)block[i * 8 + 0] << 56  \\\n"\
"                            | (ulong)block[i * 8 + 1] << 48  \\\n"\
"                            | (ulong)block[i * 8 + 2] << 40  \\\n"\
"                            | (ulong)block[i * 8 + 3] << 32  \\\n"\
"                            | (ulong)block[i * 8 + 4] << 24  \\\n"\
"                            | (ulong)block[i * 8 + 5] << 16  \\\n"\
"                            | (ulong)block[i * 8 + 6] <<  8  \\\n"\
"                            | (ulong)block[i * 8 + 7] <<  0;\n"\
"#define SCHEDULE(i)  \\\n"\
"                schedule[i] = 0UL + schedule[i - 16] + schedule[i - 7]  \\\n"\
"                        + (ROTR64(schedule[i - 15], 1) ^ ROTR64(schedule[i - 15], 8) ^ (schedule[i - 15] >> 7))  \\\n"\
"                        + (ROTR64(schedule[i - 2], 19) ^ ROTR64(schedule[i - 2], 61) ^ (schedule[i - 2] >> 6));\n"\
"#define ROUND(a, b, c, d, e, f, g, h, i, k) \\\n"\
"                h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + (ulong)(k) + schedule[i];  \\\n"\
"                d = 0UL + d + h;  \\\n"\
"                h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));\n"\
"  ulong schedule[80];\n"\
"    LOADSCHEDULE(0)\n"\
"    LOADSCHEDULE(1)\n"\
"    LOADSCHEDULE(2)\n"\
"    LOADSCHEDULE(3)\n"\
"    LOADSCHEDULE(4)\n"\
"    LOADSCHEDULE(5)\n"\
"    LOADSCHEDULE(6)\n"\
"    LOADSCHEDULE(7)\n"\
"    LOADSCHEDULE(8)\n"\
"    LOADSCHEDULE(9)\n"\
"    LOADSCHEDULE(10)\n"\
"    LOADSCHEDULE(11)\n"\
"    LOADSCHEDULE(12)\n"\
"    LOADSCHEDULE(13)\n"\
"    LOADSCHEDULE(14)\n"\
"    LOADSCHEDULE(15)\n"\
"    SCHEDULE(16)\n"\
"    SCHEDULE(17)\n"\
"    SCHEDULE(18)\n"\
"    SCHEDULE(19)\n"\
"    SCHEDULE(20)\n"\
"    SCHEDULE(21)\n"\
"    SCHEDULE(22)\n"\
"    SCHEDULE(23)\n"\
"    SCHEDULE(24)\n"\
"    SCHEDULE(25)\n"\
"    SCHEDULE(26)\n"\
"    SCHEDULE(27)\n"\
"    SCHEDULE(28)\n"\
"    SCHEDULE(29)\n"\
"    SCHEDULE(30)\n"\
"    SCHEDULE(31)\n"\
"    SCHEDULE(32)\n"\
"    SCHEDULE(33)\n"\
"    SCHEDULE(34)\n"\
"    SCHEDULE(35)\n"\
"    SCHEDULE(36)\n"\
"    SCHEDULE(37)\n"\
"    SCHEDULE(38)\n"\
"    SCHEDULE(39)\n"\
"    SCHEDULE(40)\n"\
"    SCHEDULE(41)\n"\
"    SCHEDULE(42)\n"\
"    SCHEDULE(43)\n"\
"    SCHEDULE(44)\n"\
"    SCHEDULE(45)\n"\
"    SCHEDULE(46)\n"\
"    SCHEDULE(47)\n"\
"    SCHEDULE(48)\n"\
"    SCHEDULE(49)\n"\
"    SCHEDULE(50)\n"\
"    SCHEDULE(51)\n"\
"    SCHEDULE(52)\n"\
"    SCHEDULE(53)\n"\
"    SCHEDULE(54)\n"\
"    SCHEDULE(55)\n"\
"    SCHEDULE(56)\n"\
"    SCHEDULE(57)\n"\
"    SCHEDULE(58)\n"\
"    SCHEDULE(59)\n"\
"    SCHEDULE(60)\n"\
"    SCHEDULE(61)\n"\
"    SCHEDULE(62)\n"\
"    SCHEDULE(63)\n"\
"    SCHEDULE(64)\n"\
"    SCHEDULE(65)\n"\
"    SCHEDULE(66)\n"\
"    SCHEDULE(67)\n"\
"    SCHEDULE(68)\n"\
"    SCHEDULE(69)\n"\
"    SCHEDULE(70)\n"\
"    SCHEDULE(71)\n"\
"    SCHEDULE(72)\n"\
"    SCHEDULE(73)\n"\
"    SCHEDULE(74)\n"\
"    SCHEDULE(75)\n"\
"    SCHEDULE(76)\n"\
"    SCHEDULE(77)\n"\
"    SCHEDULE(78)\n"\
"    SCHEDULE(79)\n"\
"\n"\
"    ulong a = state[0];\n"\
"    ulong b = state[1];\n"\
"    ulong c = state[2];\n"\
"    ulong d = state[3];\n"\
"    ulong e = state[4];\n"\
"    ulong f = state[5];\n"\
"    ulong g = state[6];\n"\
"    ulong h = state[7];\n"\
"    ROUND(a, b, c, d, e, f, g, h, 0, 0x428A2F98D728AE22)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 1, 0x7137449123EF65CD)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 2, 0xB5C0FBCFEC4D3B2F)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 3, 0xE9B5DBA58189DBBC)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 4, 0x3956C25BF348B538)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 5, 0x59F111F1B605D019)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 6, 0x923F82A4AF194F9B)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 7, 0xAB1C5ED5DA6D8118)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 8, 0xD807AA98A3030242)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 9, 0x12835B0145706FBE)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 10, 0x243185BE4EE4B28C)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 11, 0x550C7DC3D5FFB4E2)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 12, 0x72BE5D74F27B896F)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 13, 0x80DEB1FE3B1696B1)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 14, 0x9BDC06A725C71235)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 15, 0xC19BF174CF692694)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 16, 0xE49B69C19EF14AD2)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 17, 0xEFBE4786384F25E3)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 18, 0x0FC19DC68B8CD5B5)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 19, 0x240CA1CC77AC9C65)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 20, 0x2DE92C6F592B0275)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 21, 0x4A7484AA6EA6E483)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 22, 0x5CB0A9DCBD41FBD4)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 23, 0x76F988DA831153B5)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 24, 0x983E5152EE66DFAB)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 25, 0xA831C66D2DB43210)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 26, 0xB00327C898FB213F)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 27, 0xBF597FC7BEEF0EE4)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 28, 0xC6E00BF33DA88FC2)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 29, 0xD5A79147930AA725)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 30, 0x06CA6351E003826F)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 31, 0x142929670A0E6E70)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 32, 0x27B70A8546D22FFC)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 33, 0x2E1B21385C26C926)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 34, 0x4D2C6DFC5AC42AED)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 35, 0x53380D139D95B3DF)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 36, 0x650A73548BAF63DE)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 37, 0x766A0ABB3C77B2A8)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 38, 0x81C2C92E47EDAEE6)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 39, 0x92722C851482353B)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 40, 0xA2BFE8A14CF10364)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 41, 0xA81A664BBC423001)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 42, 0xC24B8B70D0F89791)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 43, 0xC76C51A30654BE30)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 44, 0xD192E819D6EF5218)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 45, 0xD69906245565A910)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 46, 0xF40E35855771202A)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 47, 0x106AA07032BBD1B8)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 48, 0x19A4C116B8D2D0C8)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 49, 0x1E376C085141AB53)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 50, 0x2748774CDF8EEB99)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 51, 0x34B0BCB5E19B48A8)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 52, 0x391C0CB3C5C95A63)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 53, 0x4ED8AA4AE3418ACB)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 54, 0x5B9CCA4F7763E373)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 55, 0x682E6FF3D6B2B8A3)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 56, 0x748F82EE5DEFB2FC)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 57, 0x78A5636F43172F60)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 58, 0x84C87814A1F0AB72)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 59, 0x8CC702081A6439EC)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 60, 0x90BEFFFA23631E28)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 61, 0xA4506CEBDE82BDE9)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 62, 0xBEF9A3F7B2C67915)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 63, 0xC67178F2E372532B)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 64, 0xCA273ECEEA26619C)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 65, 0xD186B8C721C0C207)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 66, 0xEADA7DD6CDE0EB1E)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 67, 0xF57D4F7FEE6ED178)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 68, 0x06F067AA72176FBA)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 69, 0x0A637DC5A2C898A6)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 70, 0x113F9804BEF90DAE)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 71, 0x1B710B35131C471B)\n"\
"    ROUND(a, b, c, d, e, f, g, h, 72, 0x28DB77F523047D84)\n"\
"    ROUND(h, a, b, c, d, e, f, g, 73, 0x32CAAB7B40C72493)\n"\
"    ROUND(g, h, a, b, c, d, e, f, 74, 0x3C9EBE0A15C9BEBC)\n"\
"    ROUND(f, g, h, a, b, c, d, e, 75, 0x431D67C49C100D4C)\n"\
"    ROUND(e, f, g, h, a, b, c, d, 76, 0x4CC5D4BECB3E42B6)\n"\
"    ROUND(d, e, f, g, h, a, b, c, 77, 0x597F299CFC657E2A)\n"\
"    ROUND(c, d, e, f, g, h, a, b, 78, 0x5FCB6FAB3AD6FAEC)\n"\
"    ROUND(b, c, d, e, f, g, h, a, 79, 0x6C44198C4A475817)\n"\
"    state[0] = 0UL + state[0] + a;\n"\
"    state[1] = 0UL + state[1] + b;\n"\
"    state[2] = 0UL + state[2] + c;\n"\
"    state[3] = 0UL + state[3] + d;\n"\
"    state[4] = 0UL + state[4] + e;\n"\
"    state[5] = 0UL + state[5] + f;\n"\
"    state[6] = 0UL + state[6] + g;\n"\
"    state[7] = 0UL + state[7] + h;\n"\
"#undef ROTR64\n"\
"#undef LOADSCHEDULE\n"\
"#undef SCHEDULE\n"\
"#undef ROUND\n"\
"}\n"\
"\n"\
"static void prsha512_hash(const uchar* message, uint len, ulong* hash) {\n"\
"  hash[0] = 0x6A09E667F3BCC908UL;\n"\
"  hash[1] = 0xBB67AE8584CAA73BUL;\n"\
"  hash[2] = 0x3C6EF372FE94F82BUL;\n"\
"  hash[3] = 0xA54FF53A5F1D36F1UL;\n"\
"  hash[4] = 0x510E527FADE682D1UL;\n"\
"  hash[5] = 0x9B05688C2B3E6C1FUL;\n"\
"  hash[6] = 0x1F83D9ABFB41BD6BUL;\n"\
"  hash[7] = 0x5BE0CD19137E2179UL;\n"\
"  uint off;\n"\
"  for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)\n"\
"    prsha512_compress(hash, &message[off]);\n"\
"  uchar block[BLOCK_LEN];\n"\
"  for (int i = 0; i < BLOCK_LEN; ++i) block[i] = 0;\n"\
"  uint rem = len - off;\n"\
"  for (uint i = 0; i < rem; ++i) block[i] = message[off + i];\n"\
"  block[rem] = 0x80;\n"\
"  rem++;\n"\
"  if (BLOCK_LEN - rem < LENGTH_SIZE) {\n"\
"    prsha512_compress(hash, block);\n"\
"    for (int i = 0; i < BLOCK_LEN; ++i) block[i] = 0;\n"\
"  }\n"\
"  ulong bitlen = (ulong)len;\n"\
"  block[BLOCK_LEN - 1] = (uchar)((bitlen & 0x1FU) << 3);\n"\
"  bitlen >>= 5;\n"\
"  for (int i = 1; i < LENGTH_SIZE; i++, bitlen >>= 8)\n"\
"    block[BLOCK_LEN - 1 - i] = (uchar)(bitlen & 0xFFU);\n"\
"  prsha512_compress(hash, block);\n"\
"}\n"\
"\n"\
"static int prsha512_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  ulong hash[STATE_LEN];\n"\
"  prsha512_hash(password, (uint)length, hash);\n"\
"  for (int i = 0; i < STATE_LEN; ++i) {\n"\
"    const ulong w = hash[i];\n"\
"    const int off = i * 8;\n"\
"    if (k_hash[off + 0] != (uchar)(w >> 56) ||\n"\
"        k_hash[off + 1] != (uchar)(w >> 48) ||\n"\
"        k_hash[off + 2] != (uchar)(w >> 40) ||\n"\
"        k_hash[off + 3] != (uchar)(w >> 32) ||\n"\
"        k_hash[off + 4] != (uchar)(w >> 24) ||\n"\
"        k_hash[off + 5] != (uchar)(w >> 16) ||\n"\
"        k_hash[off + 6] != (uchar)(w >> 8) ||\n"\
"        k_hash[off + 7] != (uchar)(w)) {\n"\
"      return 0;\n"\
"    }\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prsha512_kernel(__global uchar* result,\n"\
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
"      if (prsha512_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prsha512_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
