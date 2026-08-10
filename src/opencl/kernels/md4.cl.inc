"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 16\n"\
"#define F(B, C, D) ((((C) ^ (D)) & (B)) ^ (D))\n"\
"#define G(B, C, D) (((D) & (C)) | (((D) | (C)) & (B)))\n"\
"#define H(B, C, D) ((B) ^ (C) ^ (D))\n"\
"#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))\n"\
"#define T32(x) ((x) & 0xFFFFFFFFu)\n"\
"\n"\
"static int prmd4_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  const uint ar = (uint)k_hash[0] | (uint)k_hash[1] << 8 | (uint)k_hash[2] << 16 | (uint)k_hash[3] << 24;\n"\
"  const uint br = (uint)k_hash[4] | (uint)k_hash[5] << 8 | (uint)k_hash[6] << 16 | (uint)k_hash[7] << 24;\n"\
"  const uint cr = (uint)k_hash[8] | (uint)k_hash[9] << 8 | (uint)k_hash[10] << 16 | (uint)k_hash[11] << 24;\n"\
"  const uint dr = (uint)k_hash[12] | (uint)k_hash[13] << 8 | (uint)k_hash[14] << 16 | (uint)k_hash[15] << 24;\n"\
"  uint X[16];\n"\
"  int i;\n"\
"  for (i = 0; i < 16; ++i) X[i] = 0;\n"\
"  for (i = 0; i < length; ++i) X[i / 4] |= ((uint)password[i]) << ((i % 4) * 8);\n"\
"  X[i / 4] |= 0x80u << ((i % 4) * 8);\n"\
"  X[14] = (uint)length * 8u;\n"\
"  X[15] = 0;\n"\
"  uint A = 0x67452301u, B = 0xEFCDAB89u, C = 0x98BADCFEu, D = 0x10325476u;\n"\
"  uint AA = A, BB = B, CC = C, DD = D;\n"\
"  A = ROTL(T32(A + F(B, C, D) + X[ 0]), 3);\n"\
"  D = ROTL(T32(D + F(A, B, C) + X[ 1]), 7);\n"\
"  C = ROTL(T32(C + F(D, A, B) + X[ 2]), 11);\n"\
"  B = ROTL(T32(B + F(C, D, A) + X[ 3]), 19);\n"\
"  A = ROTL(T32(A + F(B, C, D) + X[ 4]), 3);\n"\
"  D = ROTL(T32(D + F(A, B, C) + X[ 5]), 7);\n"\
"  C = ROTL(T32(C + F(D, A, B) + X[ 6]), 11);\n"\
"  B = ROTL(T32(B + F(C, D, A) + X[ 7]), 19);\n"\
"  A = ROTL(T32(A + F(B, C, D) + X[ 8]), 3);\n"\
"  D = ROTL(T32(D + F(A, B, C) + X[ 9]), 7);\n"\
"  C = ROTL(T32(C + F(D, A, B) + X[10]), 11);\n"\
"  B = ROTL(T32(B + F(C, D, A) + X[11]), 19);\n"\
"  A = ROTL(T32(A + F(B, C, D) + X[12]), 3);\n"\
"  D = ROTL(T32(D + F(A, B, C) + X[13]), 7);\n"\
"  C = ROTL(T32(C + F(D, A, B) + X[14]), 11);\n"\
"  B = ROTL(T32(B + F(C, D, A) + X[15]), 19);\n"\
"  A = ROTL(T32(A + G(B, C, D) + X[ 0] + 0x5A827999u), 3);\n"\
"  D = ROTL(T32(D + G(A, B, C) + X[ 4] + 0x5A827999u), 5);\n"\
"  C = ROTL(T32(C + G(D, A, B) + X[ 8] + 0x5A827999u), 9);\n"\
"  B = ROTL(T32(B + G(C, D, A) + X[12] + 0x5A827999u), 13);\n"\
"  A = ROTL(T32(A + G(B, C, D) + X[ 1] + 0x5A827999u), 3);\n"\
"  D = ROTL(T32(D + G(A, B, C) + X[ 5] + 0x5A827999u), 5);\n"\
"  C = ROTL(T32(C + G(D, A, B) + X[ 9] + 0x5A827999u), 9);\n"\
"  B = ROTL(T32(B + G(C, D, A) + X[13] + 0x5A827999u), 13);\n"\
"  A = ROTL(T32(A + G(B, C, D) + X[ 2] + 0x5A827999u), 3);\n"\
"  D = ROTL(T32(D + G(A, B, C) + X[ 6] + 0x5A827999u), 5);\n"\
"  C = ROTL(T32(C + G(D, A, B) + X[10] + 0x5A827999u), 9);\n"\
"  B = ROTL(T32(B + G(C, D, A) + X[14] + 0x5A827999u), 13);\n"\
"  A = ROTL(T32(A + G(B, C, D) + X[ 3] + 0x5A827999u), 3);\n"\
"  D = ROTL(T32(D + G(A, B, C) + X[ 7] + 0x5A827999u), 5);\n"\
"  C = ROTL(T32(C + G(D, A, B) + X[11] + 0x5A827999u), 9);\n"\
"  B = ROTL(T32(B + G(C, D, A) + X[15] + 0x5A827999u), 13);\n"\
"  A = ROTL(T32(A + H(B, C, D) + X[ 0] + 0x6ED9EBA1u), 3);\n"\
"  D = ROTL(T32(D + H(A, B, C) + X[ 8] + 0x6ED9EBA1u), 9);\n"\
"  C = ROTL(T32(C + H(D, A, B) + X[ 4] + 0x6ED9EBA1u), 11);\n"\
"  B = ROTL(T32(B + H(C, D, A) + X[12] + 0x6ED9EBA1u), 15);\n"\
"  A = ROTL(T32(A + H(B, C, D) + X[ 2] + 0x6ED9EBA1u), 3);\n"\
"  D = ROTL(T32(D + H(A, B, C) + X[10] + 0x6ED9EBA1u), 9);\n"\
"  C = ROTL(T32(C + H(D, A, B) + X[ 6] + 0x6ED9EBA1u), 11);\n"\
"  B = ROTL(T32(B + H(C, D, A) + X[14] + 0x6ED9EBA1u), 15);\n"\
"  A = ROTL(T32(A + H(B, C, D) + X[ 1] + 0x6ED9EBA1u), 3);\n"\
"  D = ROTL(T32(D + H(A, B, C) + X[ 9] + 0x6ED9EBA1u), 9);\n"\
"  C = ROTL(T32(C + H(D, A, B) + X[ 5] + 0x6ED9EBA1u), 11);\n"\
"  B = ROTL(T32(B + H(C, D, A) + X[13] + 0x6ED9EBA1u), 15);\n"\
"  A = ROTL(T32(A + H(B, C, D) + X[ 3] + 0x6ED9EBA1u), 3);\n"\
"  D = ROTL(T32(D + H(A, B, C) + X[11] + 0x6ED9EBA1u), 9);\n"\
"  C = ROTL(T32(C + H(D, A, B) + X[ 7] + 0x6ED9EBA1u), 11);\n"\
"  B = ROTL(T32(B + H(C, D, A) + X[15] + 0x6ED9EBA1u), 15);\n"\
"  A = T32(A + AA); B = T32(B + BB); C = T32(C + CC); D = T32(D + DD);\n"\
"  return A == ar && B == br && C == cr && D == dr;\n"\
"}\n"\
"\n"\
"__kernel void prmd4_kernel(__global uchar* result,\n"\
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
"      if (prmd4_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prmd4_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
