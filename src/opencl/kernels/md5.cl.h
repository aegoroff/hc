"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 16\n"\
"#define F(x, y, z) (((x) & (y)) | ((~(x)) & (z)))\n"\
"#define G(x, y, z) (((x) & (z)) | ((y) & (~(z))))\n"\
"#define H(x, y, z) ((x) ^ (y) ^ (z))\n"\
"#define I(x, y, z) ((y) ^ ((x) | (~(z))))\n"\
"#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))\n"\
"#define FF(a, b, c, d, x, s, ac) { (a) += F((b),(c),(d)) + (x) + (uint)(ac); (a) = ROTATE_LEFT((a),(s)); (a) += (b); }\n"\
"#define GG(a, b, c, d, x, s, ac) { (a) += G((b),(c),(d)) + (x) + (uint)(ac); (a) = ROTATE_LEFT((a),(s)); (a) += (b); }\n"\
"#define HH(a, b, c, d, x, s, ac) { (a) += H((b),(c),(d)) + (x) + (uint)(ac); (a) = ROTATE_LEFT((a),(s)); (a) += (b); }\n"\
"#define II(a, b, c, d, x, s, ac) { (a) += I((b),(c),(d)) + (x) + (uint)(ac); (a) = ROTATE_LEFT((a),(s)); (a) += (b); }\n"\
"\n"\
"static int prmd5_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  const uint ar = (uint)k_hash[0] | (uint)k_hash[1] << 8 | (uint)k_hash[2] << 16 | (uint)k_hash[3] << 24;\n"\
"  const uint br = (uint)k_hash[4] | (uint)k_hash[5] << 8 | (uint)k_hash[6] << 16 | (uint)k_hash[7] << 24;\n"\
"  const uint cr = (uint)k_hash[8] | (uint)k_hash[9] << 8 | (uint)k_hash[10] << 16 | (uint)k_hash[11] << 24;\n"\
"  const uint dr = (uint)k_hash[12] | (uint)k_hash[13] << 8 | (uint)k_hash[14] << 16 | (uint)k_hash[15] << 24;\n"\
"  const uint a0 = 0x67452301u;\n"\
"  const uint b0 = 0xEFCDAB89u;\n"\
"  const uint c0 = 0x98BADCFEu;\n"\
"  const uint d0 = 0x10325476u;\n"\
"  uint vals[14];\n"\
"  int i;\n"\
"  for (i = 0; i < 14; ++i) vals[i] = 0;\n"\
"  for (i = 0; i < length; ++i) vals[i / 4] |= ((uint)password[i]) << ((i % 4) * 8);\n"\
"  vals[i / 4] |= 0x80u << ((i % 4) * 8);\n"\
"  const uint bitlen = (uint)length * 8u;\n"\
"  uint a = a0, b = b0, c = c0, d = d0;\n"\
"  FF(a,b,c,d, vals[0], 7, 3614090360u);\n"\
"  FF(d,a,b,c, vals[1], 12, 3905402710u);\n"\
"  FF(c,d,a,b, vals[2], 17, 606105819u);\n"\
"  FF(b,c,d,a, vals[3], 22, 3250441966u);\n"\
"  FF(a,b,c,d, vals[4], 7, 4118548399u);\n"\
"  FF(d,a,b,c, vals[5], 12, 1200080426u);\n"\
"  FF(c,d,a,b, vals[6], 17, 2821735955u);\n"\
"  FF(b,c,d,a, vals[7], 22, 4249261313u);\n"\
"  FF(a,b,c,d, vals[8], 7, 1770035416u);\n"\
"  FF(d,a,b,c, vals[9], 12, 2336552879u);\n"\
"  FF(c,d,a,b, vals[10], 17, 4294925233u);\n"\
"  FF(b,c,d,a, vals[11], 22, 2304563134u);\n"\
"  FF(a,b,c,d, vals[12], 7, 1804603682u);\n"\
"  FF(d,a,b,c, vals[13], 12, 4254626195u);\n"\
"  FF(c,d,a,b, bitlen, 17, 2792965006u);\n"\
"  FF(b,c,d,a, 0u, 22, 1236535329u);\n"\
"  GG(a,b,c,d, vals[1], 5, 4129170786u);\n"\
"  GG(d,a,b,c, vals[6], 9, 3225465664u);\n"\
"  GG(c,d,a,b, vals[11], 14, 643717713u);\n"\
"  GG(b,c,d,a, vals[0], 20, 3921069994u);\n"\
"  GG(a,b,c,d, vals[5], 5, 3593408605u);\n"\
"  GG(d,a,b,c, vals[10], 9, 38016083u);\n"\
"  GG(c,d,a,b, 0u, 14, 3634488961u);\n"\
"  GG(b,c,d,a, vals[4], 20, 3889429448u);\n"\
"  GG(a,b,c,d, vals[9], 5, 568446438u);\n"\
"  GG(d,a,b,c, bitlen, 9, 3275163606u);\n"\
"  GG(c,d,a,b, vals[3], 14, 4107603335u);\n"\
"  GG(b,c,d,a, vals[8], 20, 1163531501u);\n"\
"  GG(a,b,c,d, vals[13], 5, 2850285829u);\n"\
"  GG(d,a,b,c, vals[2], 9, 4243563512u);\n"\
"  GG(c,d,a,b, vals[7], 14, 1735328473u);\n"\
"  GG(b,c,d,a, vals[12], 20, 2368359562u);\n"\
"  HH(a,b,c,d, vals[5], 4, 4294588738u);\n"\
"  HH(d,a,b,c, vals[8], 11, 2272392833u);\n"\
"  HH(c,d,a,b, vals[11], 16, 1839030562u);\n"\
"  HH(b,c,d,a, bitlen, 23, 4259657740u);\n"\
"  HH(a,b,c,d, vals[1], 4, 2763975236u);\n"\
"  HH(d,a,b,c, vals[4], 11, 1272893353u);\n"\
"  HH(c,d,a,b, vals[7], 16, 4139469664u);\n"\
"  HH(b,c,d,a, vals[10], 23, 3200236656u);\n"\
"  HH(a,b,c,d, vals[13], 4, 681279174u);\n"\
"  HH(d,a,b,c, vals[0], 11, 3936430074u);\n"\
"  HH(c,d,a,b, vals[3], 16, 3572445317u);\n"\
"  HH(b,c,d,a, vals[6], 23, 76029189u);\n"\
"  HH(a,b,c,d, vals[9], 4, 3654602809u);\n"\
"  HH(d,a,b,c, vals[12], 11, 3873151461u);\n"\
"  HH(c,d,a,b, 0u, 16, 530742520u);\n"\
"  HH(b,c,d,a, vals[2], 23, 3299628645u);\n"\
"  II(a,b,c,d, vals[0], 6, 4096336452u);\n"\
"  II(d,a,b,c, vals[7], 10, 1126891415u);\n"\
"  II(c,d,a,b, bitlen, 15, 2878612391u);\n"\
"  II(b,c,d,a, vals[5], 21, 4237533241u);\n"\
"  II(a,b,c,d, vals[12], 6, 1700485571u);\n"\
"  II(d,a,b,c, vals[3], 10, 2399980690u);\n"\
"  II(c,d,a,b, vals[10], 15, 4293915773u);\n"\
"  II(b,c,d,a, vals[1], 21, 2240044497u);\n"\
"  II(a,b,c,d, vals[8], 6, 1873313359u);\n"\
"  II(d,a,b,c, 0u, 10, 4264355552u);\n"\
"  II(c,d,a,b, vals[6], 15, 2734768916u);\n"\
"  II(b,c,d,a, vals[13], 21, 1309151649u);\n"\
"  II(a,b,c,d, vals[4], 6, 4149444226u);\n"\
"  II(d,a,b,c, vals[11], 10, 3174756917u);\n"\
"  II(c,d,a,b, vals[2], 15, 718787259u);\n"\
"  II(b,c,d,a, vals[9], 21, 3951481745u);\n"\
"  a += a0; b += b0; c += c0; d += d0;\n"\
"  return a == ar && b == br && c == cr && d == dr;\n"\
"}\n"\
"\n"\
"__kernel void prmd5_kernel(__global uchar* result,\n"\
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
"      if (prmd5_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prmd5_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
