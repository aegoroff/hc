#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 28
#define BLOCK_LEN 64
#define STATE_LEN 8
#define HASH_LEN (STATE_LEN-1)
#define LENGTH_SIZE 8

static void prsha256_compress(uint state[], const uchar block[]) {
#define ROTR32(x, n)  (((0U + (x)) << (32 - (n))) | ((x) >> (n)))
#define LOADSCHEDULE(i)  \
                schedule[i] = (uint)block[i * 4 + 0] << 24  \
                            | (uint)block[i * 4 + 1] << 16  \
                            | (uint)block[i * 4 + 2] <<  8  \
                            | (uint)block[i * 4 + 3] <<  0;
#define SCHEDULE(i)  \
                schedule[i] = 0U + schedule[i - 16] + schedule[i - 7]  \
                        + (ROTR32(schedule[i - 15], 7) ^ ROTR32(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3))  \
                        + (ROTR32(schedule[i - 2], 17) ^ ROTR32(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10));
#define ROUND(a, b, c, d, e, f, g, h, i, k) \
                h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + (uint)(k) + schedule[i];  \
                d = 0U + d + h;  \
                h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  uint schedule[64];
  LOADSCHEDULE(0)
  LOADSCHEDULE(1)
  LOADSCHEDULE(2)
  LOADSCHEDULE(3)
  LOADSCHEDULE(4)
  LOADSCHEDULE(5)
  LOADSCHEDULE(6)
  LOADSCHEDULE(7)
  LOADSCHEDULE(8)
  LOADSCHEDULE(9)
  LOADSCHEDULE(10)
  LOADSCHEDULE(11)
  LOADSCHEDULE(12)
  LOADSCHEDULE(13)
  LOADSCHEDULE(14)
  LOADSCHEDULE(15)
  SCHEDULE(16)
  SCHEDULE(17)
  SCHEDULE(18)
  SCHEDULE(19)
  SCHEDULE(20)
  SCHEDULE(21)
  SCHEDULE(22)
  SCHEDULE(23)
  SCHEDULE(24)
  SCHEDULE(25)
  SCHEDULE(26)
  SCHEDULE(27)
  SCHEDULE(28)
  SCHEDULE(29)
  SCHEDULE(30)
  SCHEDULE(31)
  SCHEDULE(32)
  SCHEDULE(33)
  SCHEDULE(34)
  SCHEDULE(35)
  SCHEDULE(36)
  SCHEDULE(37)
  SCHEDULE(38)
  SCHEDULE(39)
  SCHEDULE(40)
  SCHEDULE(41)
  SCHEDULE(42)
  SCHEDULE(43)
  SCHEDULE(44)
  SCHEDULE(45)
  SCHEDULE(46)
  SCHEDULE(47)
  SCHEDULE(48)
  SCHEDULE(49)
  SCHEDULE(50)
  SCHEDULE(51)
  SCHEDULE(52)
  SCHEDULE(53)
  SCHEDULE(54)
  SCHEDULE(55)
  SCHEDULE(56)
  SCHEDULE(57)
  SCHEDULE(58)
  SCHEDULE(59)
  SCHEDULE(60)
  SCHEDULE(61)
  SCHEDULE(62)
  SCHEDULE(63)
  uint a = state[0];
  uint b = state[1];
  uint c = state[2];
  uint d = state[3];
  uint e = state[4];
  uint f = state[5];
  uint g = state[6];
  uint h = state[7];
  ROUND(a, b, c, d, e, f, g, h, 0, 0x428A2F98u)
  ROUND(h, a, b, c, d, e, f, g, 1, 0x71374491u)
  ROUND(g, h, a, b, c, d, e, f, 2, 0xB5C0FBCFu)
  ROUND(f, g, h, a, b, c, d, e, 3, 0xE9B5DBA5u)
  ROUND(e, f, g, h, a, b, c, d, 4, 0x3956C25Bu)
  ROUND(d, e, f, g, h, a, b, c, 5, 0x59F111F1u)
  ROUND(c, d, e, f, g, h, a, b, 6, 0x923F82A4u)
  ROUND(b, c, d, e, f, g, h, a, 7, 0xAB1C5ED5u)
  ROUND(a, b, c, d, e, f, g, h, 8, 0xD807AA98u)
  ROUND(h, a, b, c, d, e, f, g, 9, 0x12835B01u)
  ROUND(g, h, a, b, c, d, e, f, 10, 0x243185BEu)
  ROUND(f, g, h, a, b, c, d, e, 11, 0x550C7DC3u)
  ROUND(e, f, g, h, a, b, c, d, 12, 0x72BE5D74u)
  ROUND(d, e, f, g, h, a, b, c, 13, 0x80DEB1FEu)
  ROUND(c, d, e, f, g, h, a, b, 14, 0x9BDC06A7u)
  ROUND(b, c, d, e, f, g, h, a, 15, 0xC19BF174u)
  ROUND(a, b, c, d, e, f, g, h, 16, 0xE49B69C1u)
  ROUND(h, a, b, c, d, e, f, g, 17, 0xEFBE4786u)
  ROUND(g, h, a, b, c, d, e, f, 18, 0x0FC19DC6u)
  ROUND(f, g, h, a, b, c, d, e, 19, 0x240CA1CCu)
  ROUND(e, f, g, h, a, b, c, d, 20, 0x2DE92C6Fu)
  ROUND(d, e, f, g, h, a, b, c, 21, 0x4A7484AAu)
  ROUND(c, d, e, f, g, h, a, b, 22, 0x5CB0A9DCu)
  ROUND(b, c, d, e, f, g, h, a, 23, 0x76F988DAu)
  ROUND(a, b, c, d, e, f, g, h, 24, 0x983E5152u)
  ROUND(h, a, b, c, d, e, f, g, 25, 0xA831C66Du)
  ROUND(g, h, a, b, c, d, e, f, 26, 0xB00327C8u)
  ROUND(f, g, h, a, b, c, d, e, 27, 0xBF597FC7u)
  ROUND(e, f, g, h, a, b, c, d, 28, 0xC6E00BF3u)
  ROUND(d, e, f, g, h, a, b, c, 29, 0xD5A79147u)
  ROUND(c, d, e, f, g, h, a, b, 30, 0x06CA6351u)
  ROUND(b, c, d, e, f, g, h, a, 31, 0x14292967u)
  ROUND(a, b, c, d, e, f, g, h, 32, 0x27B70A85u)
  ROUND(h, a, b, c, d, e, f, g, 33, 0x2E1B2138u)
  ROUND(g, h, a, b, c, d, e, f, 34, 0x4D2C6DFCu)
  ROUND(f, g, h, a, b, c, d, e, 35, 0x53380D13u)
  ROUND(e, f, g, h, a, b, c, d, 36, 0x650A7354u)
  ROUND(d, e, f, g, h, a, b, c, 37, 0x766A0ABBu)
  ROUND(c, d, e, f, g, h, a, b, 38, 0x81C2C92Eu)
  ROUND(b, c, d, e, f, g, h, a, 39, 0x92722C85u)
  ROUND(a, b, c, d, e, f, g, h, 40, 0xA2BFE8A1u)
  ROUND(h, a, b, c, d, e, f, g, 41, 0xA81A664Bu)
  ROUND(g, h, a, b, c, d, e, f, 42, 0xC24B8B70u)
  ROUND(f, g, h, a, b, c, d, e, 43, 0xC76C51A3u)
  ROUND(e, f, g, h, a, b, c, d, 44, 0xD192E819u)
  ROUND(d, e, f, g, h, a, b, c, 45, 0xD6990624u)
  ROUND(c, d, e, f, g, h, a, b, 46, 0xF40E3585u)
  ROUND(b, c, d, e, f, g, h, a, 47, 0x106AA070u)
  ROUND(a, b, c, d, e, f, g, h, 48, 0x19A4C116u)
  ROUND(h, a, b, c, d, e, f, g, 49, 0x1E376C08u)
  ROUND(g, h, a, b, c, d, e, f, 50, 0x2748774Cu)
  ROUND(f, g, h, a, b, c, d, e, 51, 0x34B0BCB5u)
  ROUND(e, f, g, h, a, b, c, d, 52, 0x391C0CB3u)
  ROUND(d, e, f, g, h, a, b, c, 53, 0x4ED8AA4Au)
  ROUND(c, d, e, f, g, h, a, b, 54, 0x5B9CCA4Fu)
  ROUND(b, c, d, e, f, g, h, a, 55, 0x682E6FF3u)
  ROUND(a, b, c, d, e, f, g, h, 56, 0x748F82EEu)
  ROUND(h, a, b, c, d, e, f, g, 57, 0x78A5636Fu)
  ROUND(g, h, a, b, c, d, e, f, 58, 0x84C87814u)
  ROUND(f, g, h, a, b, c, d, e, 59, 0x8CC70208u)
  ROUND(e, f, g, h, a, b, c, d, 60, 0x90BEFFFAu)
  ROUND(d, e, f, g, h, a, b, c, 61, 0xA4506CEBu)
  ROUND(c, d, e, f, g, h, a, b, 62, 0xBEF9A3F7u)
  ROUND(b, c, d, e, f, g, h, a, 63, 0xC67178F2u)
  state[0] = 0U + state[0] + a;
  state[1] = 0U + state[1] + b;
  state[2] = 0U + state[2] + c;
  state[3] = 0U + state[3] + d;
  state[4] = 0U + state[4] + e;
  state[5] = 0U + state[5] + f;
  state[6] = 0U + state[6] + g;
  state[7] = 0U + state[7] + h;
#undef ROUND
#undef SCHEDULE
#undef LOADSCHEDULE
#undef ROTR32
}


static void prsha224_hash(const uchar* message, uint len, uint* hash) {
  uint state[STATE_LEN];
  state[0] = 0xC1059ED8u;
  state[1] = 0x367CD507u;
  state[2] = 0x3070DD17u;
  state[3] = 0xF70E5939u;
  state[4] = 0xFFC00B31u;
  state[5] = 0x68581511u;
  state[6] = 0x64F98FA7u;
  state[7] = 0xBEFA4FA4u;
  uint off;
  for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)
    prsha256_compress(state, &message[off]);
  uchar block[BLOCK_LEN];
  uint i;
  for (i = 0; i < BLOCK_LEN; ++i) block[i] = 0;
  uint rem = len - off;
  for (i = 0; i < rem; ++i) block[i] = message[off + i];
  block[rem] = 0x80;
  rem++;
  if (BLOCK_LEN - rem < LENGTH_SIZE) {
    prsha256_compress(state, block);
    for (i = 0; i < BLOCK_LEN; ++i) block[i] = 0;
  }
  uint bitlen = len;
  block[BLOCK_LEN - 1] = (uchar)((bitlen & 0x1FU) << 3);
  bitlen >>= 5;
  for (i = 1; i < LENGTH_SIZE; i++, bitlen >>= 8)
    block[BLOCK_LEN - 1 - i] = (uchar)(bitlen & 0xFFU);
  prsha256_compress(state, block);
  for (i = 0; i < HASH_LEN; ++i) hash[i] = state[i];
}


static int prsha224_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uint hash[HASH_LEN];
  prsha224_hash(password, (uint)length, hash);
  int result = 1;
  for (int i = 0; i < HASH_LEN && result; ++i) {
    result &= hash[i] == ((uint)k_hash[3 + i * 4] | (uint)k_hash[2 + i * 4] << 8 | (uint)k_hash[1 + i * 4] << 16 | (uint)k_hash[0 + i * 4] << 24);
  }
  return result;
}


__kernel void prsha224_kernel(__global uchar* result,
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
      if (prsha224_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prsha224_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}

