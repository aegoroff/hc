#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 16

__constant uchar k_s_md2[256] = {
    41,  46,  67, 201, 162, 216, 124,   1,  61,  54,  84, 161,
    236, 240,   6,  19,  98, 167,   5,  243, 192, 199, 115, 140,
    152, 147,  43, 217, 188,  76, 130, 202,  30, 155,  87,  60,
    253, 212, 224,  22, 103,  66, 111,  24, 138,  23, 229,  18,
    190,  78, 196, 214, 218, 158, 222,  73, 160, 251, 245, 142,
    187,  47, 238, 122, 169, 104, 121, 145,  21, 178,   7,  63,
    148, 194,  16, 137,  11,  34,  95,  33, 128, 127,  93, 154,
    90, 144,  50,  39,  53,  62, 204, 231, 191, 247, 151,   3,
    255,  25,  48, 179,  72, 165, 181, 209, 215,  94, 146,  42,
    172,  86, 170, 198,  79, 184,  56, 210, 150, 164, 125, 182,
    118, 252, 107, 226, 156, 116,   4, 241,  69, 157, 112,  89,
    100, 113, 135,  32, 134,  91, 207, 101, 230,  45, 168,   2,
    27,  96,  37, 173, 174, 176, 185, 246,  28,  70,  97, 105,
    52,  64, 126,  15,  85,  71, 163,  35, 221,  81, 175,  58,
    195,  92, 249, 206, 186, 197, 234,  38,  44,  83,  13, 110,
    133,  40, 132,   9, 211, 223, 205, 244,  65, 129,  77,  82,
    106, 220,  55, 200, 108, 193, 171, 250,  36, 225, 123,   8,
    12, 189, 177,  74, 120, 136, 149, 139, 227,  99, 232, 109,
    233, 203, 213, 254,  59,   0,  29,  57, 242, 239, 183,  14,
    102,  88, 208, 228, 166, 119, 114, 248, 235, 117,  75,  10,
    49,  68,  80, 180, 143, 237,  31,  26, 219, 153, 141,  51,
    159,  17, 131, 20
};

static void prmd2_permute(uchar X[48]) {
  int t = 0;
  for (int j = 0; j < 18; ++j) {
    for (int k = 0; k < 48; k += 8) {
      t = (X[k + 0] ^= k_s_md2[t]);
      t = (X[k + 1] ^= k_s_md2[t]);
      t = (X[k + 2] ^= k_s_md2[t]);
      t = (X[k + 3] ^= k_s_md2[t]);
      t = (X[k + 4] ^= k_s_md2[t]);
      t = (X[k + 5] ^= k_s_md2[t]);
      t = (X[k + 6] ^= k_s_md2[t]);
      t = (X[k + 7] ^= k_s_md2[t]);
    }
    t = (t + j) & 0xFF;
  }
}

static int prmd2_compare(__global const uchar* k_hash, uchar* password, const int length) {
  if (length >= DIGESTSIZE) return 0;
  uchar X[48];
  uchar C[16];
  uchar block[16];
  const uchar pad = (uchar)(DIGESTSIZE - length);
  for (int i = 0; i < DIGESTSIZE; ++i) {
    block[i] = (i < length) ? password[i] : pad;
    X[i] = 0;
    C[i] = 0;
  }
  for (int j = 0; j < 16; ++j) {
    X[j + 16] = block[j];
    X[j + 32] = (uchar)(block[j] ^ X[j]);
  }
  prmd2_permute(X);
  int t = C[15];
  for (int j = 0; j < 16; ++j) {
    C[j] ^= k_s_md2[block[j] ^ t];
    t = C[j];
  }
  for (int j = 0; j < 16; ++j) {
    X[j + 16] = C[j];
    X[j + 32] = (uchar)(C[j] ^ X[j]);
  }
  prmd2_permute(X);
  for (int i = 0; i < DIGESTSIZE; ++i) {
    if (X[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prmd2_kernel(__global uchar* result,
                          __global const uchar* k_dict,
                          __global const uchar* k_hash,
                          __global int* g_found,
                          const ulong start,
                          const uint count,
                          const uint pass_len,
                          const uint dict_length,
                          const uint min_len) {
  const uint ix = get_global_id(0);
  if (ix >= count || *g_found) return;
  ulong idx = start + (ulong)ix;
  uchar attempt[GPU_ATTEMPT_SIZE];
  for (int pos = (int)pass_len - 1; pos >= 0; --pos) {
    attempt[pos] = k_dict[idx % dict_length];
    idx /= dict_length;
  }
  for (uint i = 0; i < dict_length; ++i) {
    attempt[pass_len] = k_dict[i];
    if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {
      if (*g_found) return;
      if (prmd2_compare(k_hash, attempt, (int)(pass_len + 1u))) {
        for (uint k = 0; k < pass_len + 1u; ++k) result[k] = attempt[k];
        result[pass_len + 1u] = 0;
        *g_found = 1;
        return;
      }
    }
    if (pass_len + 2u < min_len) continue;
    for (uint j = 0; j < dict_length; ++j) {
      attempt[pass_len + 1u] = k_dict[j];
      if (*g_found) return;
      if (prmd2_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
