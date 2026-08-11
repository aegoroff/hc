"#define GPU_ATTEMPT_SIZE 16\n"\
"#define DIGESTSIZE 16\n"\
"\n"\
"__constant uchar k_s_md2[256] = {\n"\
"    41,  46,  67, 201, 162, 216, 124,   1,  61,  54,  84, 161,\n"\
"    236, 240,   6,  19,  98, 167,   5,  243, 192, 199, 115, 140,\n"\
"    152, 147,  43, 217, 188,  76, 130, 202,  30, 155,  87,  60,\n"\
"    253, 212, 224,  22, 103,  66, 111,  24, 138,  23, 229,  18,\n"\
"    190,  78, 196, 214, 218, 158, 222,  73, 160, 251, 245, 142,\n"\
"    187,  47, 238, 122, 169, 104, 121, 145,  21, 178,   7,  63,\n"\
"    148, 194,  16, 137,  11,  34,  95,  33, 128, 127,  93, 154,\n"\
"    90, 144,  50,  39,  53,  62, 204, 231, 191, 247, 151,   3,\n"\
"    255,  25,  48, 179,  72, 165, 181, 209, 215,  94, 146,  42,\n"\
"    172,  86, 170, 198,  79, 184,  56, 210, 150, 164, 125, 182,\n"\
"    118, 252, 107, 226, 156, 116,   4, 241,  69, 157, 112,  89,\n"\
"    100, 113, 135,  32, 134,  91, 207, 101, 230,  45, 168,   2,\n"\
"    27,  96,  37, 173, 174, 176, 185, 246,  28,  70,  97, 105,\n"\
"    52,  64, 126,  15,  85,  71, 163,  35, 221,  81, 175,  58,\n"\
"    195,  92, 249, 206, 186, 197, 234,  38,  44,  83,  13, 110,\n"\
"    133,  40, 132,   9, 211, 223, 205, 244,  65, 129,  77,  82,\n"\
"    106, 220,  55, 200, 108, 193, 171, 250,  36, 225, 123,   8,\n"\
"    12, 189, 177,  74, 120, 136, 149, 139, 227,  99, 232, 109,\n"\
"    233, 203, 213, 254,  59,   0,  29,  57, 242, 239, 183,  14,\n"\
"    102,  88, 208, 228, 166, 119, 114, 248, 235, 117,  75,  10,\n"\
"    49,  68,  80, 180, 143, 237,  31,  26, 219, 153, 141,  51,\n"\
"    159,  17, 131, 20\n"\
"};\n"\
"\n"\
"static void prmd2_permute(uchar X[48]) {\n"\
"  int t = 0;\n"\
"  for (int j = 0; j < 18; ++j) {\n"\
"    for (int k = 0; k < 48; k += 8) {\n"\
"      t = (X[k + 0] ^= k_s_md2[t]);\n"\
"      t = (X[k + 1] ^= k_s_md2[t]);\n"\
"      t = (X[k + 2] ^= k_s_md2[t]);\n"\
"      t = (X[k + 3] ^= k_s_md2[t]);\n"\
"      t = (X[k + 4] ^= k_s_md2[t]);\n"\
"      t = (X[k + 5] ^= k_s_md2[t]);\n"\
"      t = (X[k + 6] ^= k_s_md2[t]);\n"\
"      t = (X[k + 7] ^= k_s_md2[t]);\n"\
"    }\n"\
"    t = (t + j) & 0xFF;\n"\
"  }\n"\
"}\n"\
"\n"\
"static int prmd2_compare(__global const uchar* k_hash, uchar* password, const int length) {\n"\
"  if (length >= DIGESTSIZE) return 0;\n"\
"  uchar X[48];\n"\
"  uchar C[16];\n"\
"  uchar block[16];\n"\
"  const uchar pad = (uchar)(DIGESTSIZE - length);\n"\
"  for (int i = 0; i < DIGESTSIZE; ++i) {\n"\
"    block[i] = (i < length) ? password[i] : pad;\n"\
"    X[i] = 0;\n"\
"    C[i] = 0;\n"\
"  }\n"\
"  for (int j = 0; j < 16; ++j) {\n"\
"    X[j + 16] = block[j];\n"\
"    X[j + 32] = (uchar)(block[j] ^ X[j]);\n"\
"  }\n"\
"  prmd2_permute(X);\n"\
"  int t = C[15];\n"\
"  for (int j = 0; j < 16; ++j) {\n"\
"    C[j] ^= k_s_md2[block[j] ^ t];\n"\
"    t = C[j];\n"\
"  }\n"\
"  for (int j = 0; j < 16; ++j) {\n"\
"    X[j + 16] = C[j];\n"\
"    X[j + 32] = (uchar)(C[j] ^ X[j]);\n"\
"  }\n"\
"  prmd2_permute(X);\n"\
"  for (int i = 0; i < DIGESTSIZE; ++i) {\n"\
"    if (X[i] != k_hash[i]) return 0;\n"\
"  }\n"\
"  return 1;\n"\
"}\n"\
"\n"\
"__kernel void prmd2_kernel(__global uchar* result,\n"\
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
"      if (prmd2_compare(k_hash, attempt, (int)(pass_len + 1u))) {\n"\
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
"      if (prmd2_compare(k_hash, attempt, (int)(pass_len + 2u))) {\n"\
"        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];\n"\
"        result[pass_len + 2u] = 0;\n"\
"        *g_found = 1;\n"\
"        return;\n"\
"      }\n"\
"    }\n"\
"  }\n"\
"}\n"\
