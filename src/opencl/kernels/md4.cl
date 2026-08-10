#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 16
#define F(B, C, D) ((((C) ^ (D)) & (B)) ^ (D))
#define G(B, C, D) (((D) & (C)) | (((D) | (C)) & (B)))
#define H(B, C, D) ((B) ^ (C) ^ (D))
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define T32(x) ((x) & 0xFFFFFFFFu)

static int prmd4_compare(__global const uchar* k_hash, uchar* password, const int length) {
  const uint ar = (uint)k_hash[0] | (uint)k_hash[1] << 8 | (uint)k_hash[2] << 16 | (uint)k_hash[3] << 24;
  const uint br = (uint)k_hash[4] | (uint)k_hash[5] << 8 | (uint)k_hash[6] << 16 | (uint)k_hash[7] << 24;
  const uint cr = (uint)k_hash[8] | (uint)k_hash[9] << 8 | (uint)k_hash[10] << 16 | (uint)k_hash[11] << 24;
  const uint dr = (uint)k_hash[12] | (uint)k_hash[13] << 8 | (uint)k_hash[14] << 16 | (uint)k_hash[15] << 24;
  uint X[16];
  int i;
  for (i = 0; i < 16; ++i) X[i] = 0;
  for (i = 0; i < length; ++i) X[i / 4] |= ((uint)password[i]) << ((i % 4) * 8);
  X[i / 4] |= 0x80u << ((i % 4) * 8);
  X[14] = (uint)length * 8u;
  X[15] = 0;
  uint A = 0x67452301u, B = 0xEFCDAB89u, C = 0x98BADCFEu, D = 0x10325476u;
  uint AA = A, BB = B, CC = C, DD = D;
  A = ROTL(T32(A + F(B, C, D) + X[ 0]), 3);
  D = ROTL(T32(D + F(A, B, C) + X[ 1]), 7);
  C = ROTL(T32(C + F(D, A, B) + X[ 2]), 11);
  B = ROTL(T32(B + F(C, D, A) + X[ 3]), 19);
  A = ROTL(T32(A + F(B, C, D) + X[ 4]), 3);
  D = ROTL(T32(D + F(A, B, C) + X[ 5]), 7);
  C = ROTL(T32(C + F(D, A, B) + X[ 6]), 11);
  B = ROTL(T32(B + F(C, D, A) + X[ 7]), 19);
  A = ROTL(T32(A + F(B, C, D) + X[ 8]), 3);
  D = ROTL(T32(D + F(A, B, C) + X[ 9]), 7);
  C = ROTL(T32(C + F(D, A, B) + X[10]), 11);
  B = ROTL(T32(B + F(C, D, A) + X[11]), 19);
  A = ROTL(T32(A + F(B, C, D) + X[12]), 3);
  D = ROTL(T32(D + F(A, B, C) + X[13]), 7);
  C = ROTL(T32(C + F(D, A, B) + X[14]), 11);
  B = ROTL(T32(B + F(C, D, A) + X[15]), 19);
  A = ROTL(T32(A + G(B, C, D) + X[ 0] + 0x5A827999u), 3);
  D = ROTL(T32(D + G(A, B, C) + X[ 4] + 0x5A827999u), 5);
  C = ROTL(T32(C + G(D, A, B) + X[ 8] + 0x5A827999u), 9);
  B = ROTL(T32(B + G(C, D, A) + X[12] + 0x5A827999u), 13);
  A = ROTL(T32(A + G(B, C, D) + X[ 1] + 0x5A827999u), 3);
  D = ROTL(T32(D + G(A, B, C) + X[ 5] + 0x5A827999u), 5);
  C = ROTL(T32(C + G(D, A, B) + X[ 9] + 0x5A827999u), 9);
  B = ROTL(T32(B + G(C, D, A) + X[13] + 0x5A827999u), 13);
  A = ROTL(T32(A + G(B, C, D) + X[ 2] + 0x5A827999u), 3);
  D = ROTL(T32(D + G(A, B, C) + X[ 6] + 0x5A827999u), 5);
  C = ROTL(T32(C + G(D, A, B) + X[10] + 0x5A827999u), 9);
  B = ROTL(T32(B + G(C, D, A) + X[14] + 0x5A827999u), 13);
  A = ROTL(T32(A + G(B, C, D) + X[ 3] + 0x5A827999u), 3);
  D = ROTL(T32(D + G(A, B, C) + X[ 7] + 0x5A827999u), 5);
  C = ROTL(T32(C + G(D, A, B) + X[11] + 0x5A827999u), 9);
  B = ROTL(T32(B + G(C, D, A) + X[15] + 0x5A827999u), 13);
  A = ROTL(T32(A + H(B, C, D) + X[ 0] + 0x6ED9EBA1u), 3);
  D = ROTL(T32(D + H(A, B, C) + X[ 8] + 0x6ED9EBA1u), 9);
  C = ROTL(T32(C + H(D, A, B) + X[ 4] + 0x6ED9EBA1u), 11);
  B = ROTL(T32(B + H(C, D, A) + X[12] + 0x6ED9EBA1u), 15);
  A = ROTL(T32(A + H(B, C, D) + X[ 2] + 0x6ED9EBA1u), 3);
  D = ROTL(T32(D + H(A, B, C) + X[10] + 0x6ED9EBA1u), 9);
  C = ROTL(T32(C + H(D, A, B) + X[ 6] + 0x6ED9EBA1u), 11);
  B = ROTL(T32(B + H(C, D, A) + X[14] + 0x6ED9EBA1u), 15);
  A = ROTL(T32(A + H(B, C, D) + X[ 1] + 0x6ED9EBA1u), 3);
  D = ROTL(T32(D + H(A, B, C) + X[ 9] + 0x6ED9EBA1u), 9);
  C = ROTL(T32(C + H(D, A, B) + X[ 5] + 0x6ED9EBA1u), 11);
  B = ROTL(T32(B + H(C, D, A) + X[13] + 0x6ED9EBA1u), 15);
  A = ROTL(T32(A + H(B, C, D) + X[ 3] + 0x6ED9EBA1u), 3);
  D = ROTL(T32(D + H(A, B, C) + X[11] + 0x6ED9EBA1u), 9);
  C = ROTL(T32(C + H(D, A, B) + X[ 7] + 0x6ED9EBA1u), 11);
  B = ROTL(T32(B + H(C, D, A) + X[15] + 0x6ED9EBA1u), 15);
  A = T32(A + AA); B = T32(B + BB); C = T32(C + CC); D = T32(D + DD);
  return A == ar && B == br && C == cr && D == dr;
}

__kernel void prmd4_kernel(__global uchar* result,
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
      if (prmd4_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prmd4_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
