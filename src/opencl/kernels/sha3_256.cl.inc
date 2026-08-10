"#define GPU_ATTEMPT_SIZE 16\n"\
"#define MAX_RATE 144\n"\
"#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))\n"\
"\n"\
"__constant ulong k_rc[24] = {\n"\
"    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,\n"\
"    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,\n"\
"    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,\n"\
"    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,\n"\
"    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,\n"\
"    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,\n"\
"};\n"\
"\n"\
"static void prkeccak_f1600(ulong* A) {\n"\
"    for (int round = 0; round < 24; round++) {\n"\
"        ulong C[5], D[5];\n"\
"        for (int x = 0; x < 5; x++) {\n"\
"            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];\n"\
"        }\n"\
"        D[0] = ROTL64(C[1], 1) ^ C[4];\n"\
"        D[1] = ROTL64(C[2], 1) ^ C[0];\n"\
"        D[2] = ROTL64(C[3], 1) ^ C[1];\n"\
"        D[3] = ROTL64(C[4], 1) ^ C[2];\n"\
"        D[4] = ROTL64(C[0], 1) ^ C[3];\n"\
"        for (int x = 0; x < 5; x++) {\n"\
"            A[x] ^= D[x];\n"\
"            A[x + 5] ^= D[x];\n"\
"            A[x + 10] ^= D[x];\n"\
"            A[x + 15] ^= D[x];\n"\
"            A[x + 20] ^= D[x];\n"\
"        }\n"\
"\n"\
"        A[1] = ROTL64(A[1], 1);\n"\
"        A[2] = ROTL64(A[2], 62);\n"\
"        A[3] = ROTL64(A[3], 28);\n"\
"        A[4] = ROTL64(A[4], 27);\n"\
"        A[5] = ROTL64(A[5], 36);\n"\
"        A[6] = ROTL64(A[6], 44);\n"\
"        A[7] = ROTL64(A[7], 6);\n"\
"        A[8] = ROTL64(A[8], 55);\n"\
"        A[9] = ROTL64(A[9], 20);\n"\
"        A[10] = ROTL64(A[10], 3);\n"\
"        A[11] = ROTL64(A[11], 10);\n"\
"        A[12] = ROTL64(A[12], 43);\n"\
"        A[13] = ROTL64(A[13], 25);\n"\
"        A[14] = ROTL64(A[14], 39);\n"\
"        A[15] = ROTL64(A[15], 41);\n"\
"        A[16] = ROTL64(A[16], 45);\n"\
"        A[17] = ROTL64(A[17], 15);\n"\
"        A[18] = ROTL64(A[18], 21);\n"\
"        A[19] = ROTL64(A[19], 8);\n"\
"        A[20] = ROTL64(A[20], 18);\n"\
"        A[21] = ROTL64(A[21], 2);\n"\
"        A[22] = ROTL64(A[22], 61);\n"\
"        A[23] = ROTL64(A[23], 56);\n"\
"        A[24] = ROTL64(A[24], 14);\n"\
"\n"\
"        {\n"\
"            ulong A1 = A[1];\n"\
"            A[1] = A[6];\n"\
"            A[6] = A[9];\n"\
"            A[9] = A[22];\n"\
"            A[22] = A[14];\n"\
"            A[14] = A[20];\n"\
"            A[20] = A[2];\n"\
"            A[2] = A[12];\n"\
"            A[12] = A[13];\n"\
"            A[13] = A[19];\n"\
"            A[19] = A[23];\n"\
"            A[23] = A[15];\n"\
"            A[15] = A[4];\n"\
"            A[4] = A[24];\n"\
"            A[24] = A[21];\n"\
"            A[21] = A[8];\n"\
"            A[8] = A[16];\n"\
"            A[16] = A[5];\n"\
"            A[5] = A[3];\n"\
"            A[3] = A[18];\n"\
"            A[18] = A[17];\n"\
"            A[17] = A[11];\n"\
"            A[11] = A[7];\n"\
"            A[7] = A[10];\n"\
"            A[10] = A1;\n"\
"        }\n"\
"\n"\
"        for (int i = 0; i < 25; i += 5) {\n"\
"            ulong A0 = A[0 + i], A1 = A[1 + i];\n"\
"            A[0 + i] ^= ~A1 & A[2 + i];\n"\
"            A[1 + i] ^= ~A[2 + i] & A[3 + i];\n"\
"            A[2 + i] ^= ~A[3 + i] & A[4 + i];\n"\
"            A[3 + i] ^= ~A[4 + i] & A0;\n"\
"            A[4 + i] ^= ~A0 & A1;\n"\
"        }\n"\
"\n"\
"        A[0] ^= k_rc[round];\n"\
"    }\n"\
"}\n"\
"\n"\
"static void prsha3_hash(const uchar* message, uint len, uchar* out,\n"\
"                                             uint rate, uint out_len, uchar pad) {\n"\
"    ulong state[25]; for (int __z = 0; __z < (int)(25); ++__z) state[__z] = 0;\n"\
"    uchar block[MAX_RATE]; for (int __z = 0; __z < (int)(MAX_RATE); ++__z) block[__z] = 0;\n"\
"    { const uint __n = (uint)(len); for (uint __i = 0; __i < __n; ++__i) (block)[__i] = (message)[__i]; }\n"\
"    block[len] |= pad;\n"\
"    block[rate - 1] |= 0x80;\n"\
"\n"\
"    const uint nq = rate / 8;\n"\
"    for (uint i = 0; i < nq; i++) {\n"\
"        const uint o = i * 8;\n"\
"        state[i] ^= (ulong)(block[o + 0])\n"\
"            | ((ulong)(block[o + 1]) << 8)\n"\
"            | ((ulong)(block[o + 2]) << 16)\n"\
"            | ((ulong)(block[o + 3]) << 24)\n"\
"            | ((ulong)(block[o + 4]) << 32)\n"\
"            | ((ulong)(block[o + 5]) << 40)\n"\
"            | ((ulong)(block[o + 6]) << 48)\n"\
"            | ((ulong)(block[o + 7]) << 56);\n"\
"    }\n"\
"    prkeccak_f1600(state);\n"\
"\n"\
"    for (uint i = 0; i < out_len; i++) {\n"\
"        out[i] = (uchar)(state[i >> 3] >> ((i & 7) << 3));\n"\
"    }\n"\
"}\n"\
"\n"\
"static int prsha3_256_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  uchar hash[32];\n"\
"  prsha3_hash(password, (uint)length, hash, 136, 32, (uchar)6);\n"\
"  for (int i = 0; i < 32; ++i) {\n"\
"    if (hash[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prsha3_256_kernel(__global uchar* result,\n"\
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
"    if (prsha3_256_compare(k_hash, attempt, (int)pass_len)) {\n"\
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
"    if (prsha3_256_compare(k_hash, attempt, (int)attempt_len)) {\n"\
"      for (uint k = 0; k < attempt_len; ++k) result[k] = attempt[k];\n"\
"      result[attempt_len] = 0;\n"\
"      *g_found = 1;\n"\
"      return;\n"\
"    }\n"\
"  }\n"\
"}\n"\
