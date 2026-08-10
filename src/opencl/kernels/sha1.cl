#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 20

/* Short-password SHA-1: single block, big-endian words (len <= GPU_ATTEMPT_SIZE). */
static int prsha1_compare(__global const uchar* k_hash, uchar* password, const int length) {
  const uint h0 = (uint)k_hash[3] | (uint)k_hash[2] << 8 | (uint)k_hash[1] << 16 | (uint)k_hash[0] << 24;
  const uint h1 = (uint)k_hash[7] | (uint)k_hash[6] << 8 | (uint)k_hash[5] << 16 | (uint)k_hash[4] << 24;
  const uint h2 = (uint)k_hash[11] | (uint)k_hash[10] << 8 | (uint)k_hash[9] << 16 | (uint)k_hash[8] << 24;
  const uint h3 = (uint)k_hash[15] | (uint)k_hash[14] << 8 | (uint)k_hash[13] << 16 | (uint)k_hash[12] << 24;
  const uint h4 = (uint)k_hash[19] | (uint)k_hash[18] << 8 | (uint)k_hash[17] << 16 | (uint)k_hash[16] << 24;

  const uint a0 = 0x67452301u;
  const uint b0 = 0xEFCDAB89u;
  const uint c0 = 0x98BADCFEu;
  const uint d0 = 0x10325476u;
  const uint e0 = 0xC3D2E1F0u;

  uint schedule[16];
  int i;
  for (i = 0; i < 16; ++i) schedule[i] = 0;
  for (i = 0; i < length; ++i)
    schedule[i / 4] |= ((uint)password[i]) << (24 - (i % 4) * 8);
  schedule[i / 4] |= 0x80u << (24 - (i % 4) * 8);
  schedule[15] = (uint)length * 8u;

#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))
#define SCHEDULE(i)  \
                temp = schedule[(i - 3) & 0xF] ^ schedule[(i - 8) & 0xF] ^ schedule[(i - 14) & 0xF] ^ schedule[(i - 16) & 0xF];  \
                schedule[i & 0xF] = ROTL32(temp, 1);
#define ROUND0(a, b, c, d, e, i)  ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999u)
#define ROUND0b(a, b, c, d, e, i) SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999u)
#define ROUND1(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0x6ED9EBA1u)
#define ROUND2(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDCu)
#define ROUND3(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0xCA62C1D6u)
#define ROUNDTAIL(a, b, e, f, i, k)  \
                e = 0U + e + ROTL32(a, 5) + f + (uint)(k) + schedule[i & 0xF];  \
                b = ROTL32(b, 30);

  uint a = a0, b = b0, c = c0, d = d0, e = e0;
  uint temp;
  ROUND0(a, b, c, d, e, 0)
  ROUND0(e, a, b, c, d, 1)
  ROUND0(d, e, a, b, c, 2)
  ROUND0(c, d, e, a, b, 3)
  ROUND0(b, c, d, e, a, 4)
  ROUND0(a, b, c, d, e, 5)
  ROUND0(e, a, b, c, d, 6)
  ROUND0(d, e, a, b, c, 7)
  ROUND0(c, d, e, a, b, 8)
  ROUND0(b, c, d, e, a, 9)
  ROUND0(a, b, c, d, e, 10)
  ROUND0(e, a, b, c, d, 11)
  ROUND0(d, e, a, b, c, 12)
  ROUND0(c, d, e, a, b, 13)
  ROUND0(b, c, d, e, a, 14)
  ROUND0(a, b, c, d, e, 15)
  ROUND0b(e, a, b, c, d, 16)
  ROUND0b(d, e, a, b, c, 17)
  ROUND0b(c, d, e, a, b, 18)
  ROUND0b(b, c, d, e, a, 19)
  ROUND1(a, b, c, d, e, 20)
  ROUND1(e, a, b, c, d, 21)
  ROUND1(d, e, a, b, c, 22)
  ROUND1(c, d, e, a, b, 23)
  ROUND1(b, c, d, e, a, 24)
  ROUND1(a, b, c, d, e, 25)
  ROUND1(e, a, b, c, d, 26)
  ROUND1(d, e, a, b, c, 27)
  ROUND1(c, d, e, a, b, 28)
  ROUND1(b, c, d, e, a, 29)
  ROUND1(a, b, c, d, e, 30)
  ROUND1(e, a, b, c, d, 31)
  ROUND1(d, e, a, b, c, 32)
  ROUND1(c, d, e, a, b, 33)
  ROUND1(b, c, d, e, a, 34)
  ROUND1(a, b, c, d, e, 35)
  ROUND1(e, a, b, c, d, 36)
  ROUND1(d, e, a, b, c, 37)
  ROUND1(c, d, e, a, b, 38)
  ROUND1(b, c, d, e, a, 39)
  ROUND2(a, b, c, d, e, 40)
  ROUND2(e, a, b, c, d, 41)
  ROUND2(d, e, a, b, c, 42)
  ROUND2(c, d, e, a, b, 43)
  ROUND2(b, c, d, e, a, 44)
  ROUND2(a, b, c, d, e, 45)
  ROUND2(e, a, b, c, d, 46)
  ROUND2(d, e, a, b, c, 47)
  ROUND2(c, d, e, a, b, 48)
  ROUND2(b, c, d, e, a, 49)
  ROUND2(a, b, c, d, e, 50)
  ROUND2(e, a, b, c, d, 51)
  ROUND2(d, e, a, b, c, 52)
  ROUND2(c, d, e, a, b, 53)
  ROUND2(b, c, d, e, a, 54)
  ROUND2(a, b, c, d, e, 55)
  ROUND2(e, a, b, c, d, 56)
  ROUND2(d, e, a, b, c, 57)
  ROUND2(c, d, e, a, b, 58)
  ROUND2(b, c, d, e, a, 59)
  ROUND3(a, b, c, d, e, 60)
  ROUND3(e, a, b, c, d, 61)
  ROUND3(d, e, a, b, c, 62)
  ROUND3(c, d, e, a, b, 63)
  ROUND3(b, c, d, e, a, 64)
  ROUND3(a, b, c, d, e, 65)
  ROUND3(e, a, b, c, d, 66)
  ROUND3(d, e, a, b, c, 67)
  ROUND3(c, d, e, a, b, 68)
  ROUND3(b, c, d, e, a, 69)
  ROUND3(a, b, c, d, e, 70)
  ROUND3(e, a, b, c, d, 71)
  ROUND3(d, e, a, b, c, 72)
  ROUND3(c, d, e, a, b, 73)
  ROUND3(b, c, d, e, a, 74)
  ROUND3(a, b, c, d, e, 75)
  ROUND3(e, a, b, c, d, 76)
  ROUND3(d, e, a, b, c, 77)
  ROUND3(c, d, e, a, b, 78)
  ROUND3(b, c, d, e, a, 79)

  a = 0U + a0 + a;
  b = 0U + b0 + b;
  c = 0U + c0 + c;
  d = 0U + d0 + d;
  e = 0U + e0 + e;

#undef ROUNDTAIL
#undef ROUND3
#undef ROUND2
#undef ROUND1
#undef ROUND0b
#undef ROUND0
#undef SCHEDULE
#undef ROTL32

  return a == h0 && b == h1 && c == h2 && d == h3 && e == h4;
}


__kernel void prsha1_kernel(__global uchar* result,
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
      if (prsha1_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prsha1_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
