#define GPU_ATTEMPT_SIZE 16
#define BLOCK_LEN 64
#define HASH_LEN 16

#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define ROTL32(x, n) (((0U + (x)) << (n)) | ((x) >> (32 - (n))))
#define FF(a, b, c, d, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }
#define GG(a, b, c, d, x, s) \
    { (a) += G((b), (c), (d)) + (x) + 0x5a827999u; (a) = ROTL32((a), (s)); }
#define HH(a, b, c, d, x, s) \
    { (a) += H((b), (c), (d)) + (x) + 0x6ed9eba1u; (a) = ROTL32((a), (s)); }
#define II(a, b, c, d, x, s) \
    { (a) += I((b), (c), (d)) + (x) + 0x8f1bbcdcu; (a) = ROTL32((a), (s)); }
#define FFF(a, b, c, d, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }
#define GGG(a, b, c, d, x, s) \
    { (a) += G((b), (c), (d)) + (x) + 0x6d703ef3u; (a) = ROTL32((a), (s)); }
#define HHH(a, b, c, d, x, s) \
    { (a) += H((b), (c), (d)) + (x) + 0x5c4dd124u; (a) = ROTL32((a), (s)); }
#define III(a, b, c, d, x, s) \
    { (a) += I((b), (c), (d)) + (x) + 0x50a28be6u; (a) = ROTL32((a), (s)); }



static void prrmd128_compress(uint* state, const uchar* block) {
    uint X[16];
    for (int j = 0; j < 16; j++) {
        const int i = j * 4;
        X[j] = (uint)(block[i + 0])
            | ((uint)(block[i + 1]) << 8)
            | ((uint)(block[i + 2]) << 16)
            | ((uint)(block[i + 3]) << 24);
    }

    uint aa = state[0], bb = state[1], cc = state[2], dd = state[3];
    uint aaa = state[0], bbb = state[1], ccc = state[2], ddd = state[3];

    FF(aa, bb, cc, dd, X[0], 11);
    FF(dd, aa, bb, cc, X[1], 14);
    FF(cc, dd, aa, bb, X[2], 15);
    FF(bb, cc, dd, aa, X[3], 12);
    FF(aa, bb, cc, dd, X[4], 5);
    FF(dd, aa, bb, cc, X[5], 8);
    FF(cc, dd, aa, bb, X[6], 7);
    FF(bb, cc, dd, aa, X[7], 9);
    FF(aa, bb, cc, dd, X[8], 11);
    FF(dd, aa, bb, cc, X[9], 13);
    FF(cc, dd, aa, bb, X[10], 14);
    FF(bb, cc, dd, aa, X[11], 15);
    FF(aa, bb, cc, dd, X[12], 6);
    FF(dd, aa, bb, cc, X[13], 7);
    FF(cc, dd, aa, bb, X[14], 9);
    FF(bb, cc, dd, aa, X[15], 8);

    GG(aa, bb, cc, dd, X[7], 7);
    GG(dd, aa, bb, cc, X[4], 6);
    GG(cc, dd, aa, bb, X[13], 8);
    GG(bb, cc, dd, aa, X[1], 13);
    GG(aa, bb, cc, dd, X[10], 11);
    GG(dd, aa, bb, cc, X[6], 9);
    GG(cc, dd, aa, bb, X[15], 7);
    GG(bb, cc, dd, aa, X[3], 15);
    GG(aa, bb, cc, dd, X[12], 7);
    GG(dd, aa, bb, cc, X[0], 12);
    GG(cc, dd, aa, bb, X[9], 15);
    GG(bb, cc, dd, aa, X[5], 9);
    GG(aa, bb, cc, dd, X[2], 11);
    GG(dd, aa, bb, cc, X[14], 7);
    GG(cc, dd, aa, bb, X[11], 13);
    GG(bb, cc, dd, aa, X[8], 12);

    HH(aa, bb, cc, dd, X[3], 11);
    HH(dd, aa, bb, cc, X[10], 13);
    HH(cc, dd, aa, bb, X[14], 6);
    HH(bb, cc, dd, aa, X[4], 7);
    HH(aa, bb, cc, dd, X[9], 14);
    HH(dd, aa, bb, cc, X[15], 9);
    HH(cc, dd, aa, bb, X[8], 13);
    HH(bb, cc, dd, aa, X[1], 15);
    HH(aa, bb, cc, dd, X[2], 14);
    HH(dd, aa, bb, cc, X[7], 8);
    HH(cc, dd, aa, bb, X[0], 13);
    HH(bb, cc, dd, aa, X[6], 6);
    HH(aa, bb, cc, dd, X[13], 5);
    HH(dd, aa, bb, cc, X[11], 12);
    HH(cc, dd, aa, bb, X[5], 7);
    HH(bb, cc, dd, aa, X[12], 5);

    II(aa, bb, cc, dd, X[1], 11);
    II(dd, aa, bb, cc, X[9], 12);
    II(cc, dd, aa, bb, X[11], 14);
    II(bb, cc, dd, aa, X[10], 15);
    II(aa, bb, cc, dd, X[0], 14);
    II(dd, aa, bb, cc, X[8], 15);
    II(cc, dd, aa, bb, X[12], 9);
    II(bb, cc, dd, aa, X[4], 8);
    II(aa, bb, cc, dd, X[13], 9);
    II(dd, aa, bb, cc, X[3], 14);
    II(cc, dd, aa, bb, X[7], 5);
    II(bb, cc, dd, aa, X[15], 6);
    II(aa, bb, cc, dd, X[14], 8);
    II(dd, aa, bb, cc, X[5], 6);
    II(cc, dd, aa, bb, X[6], 5);
    II(bb, cc, dd, aa, X[2], 12);

    III(aaa, bbb, ccc, ddd, X[5], 8);
    III(ddd, aaa, bbb, ccc, X[14], 9);
    III(ccc, ddd, aaa, bbb, X[7], 9);
    III(bbb, ccc, ddd, aaa, X[0], 11);
    III(aaa, bbb, ccc, ddd, X[9], 13);
    III(ddd, aaa, bbb, ccc, X[2], 15);
    III(ccc, ddd, aaa, bbb, X[11], 15);
    III(bbb, ccc, ddd, aaa, X[4], 5);
    III(aaa, bbb, ccc, ddd, X[13], 7);
    III(ddd, aaa, bbb, ccc, X[6], 7);
    III(ccc, ddd, aaa, bbb, X[15], 8);
    III(bbb, ccc, ddd, aaa, X[8], 11);
    III(aaa, bbb, ccc, ddd, X[1], 14);
    III(ddd, aaa, bbb, ccc, X[10], 14);
    III(ccc, ddd, aaa, bbb, X[3], 12);
    III(bbb, ccc, ddd, aaa, X[12], 6);

    HHH(aaa, bbb, ccc, ddd, X[6], 9);
    HHH(ddd, aaa, bbb, ccc, X[11], 13);
    HHH(ccc, ddd, aaa, bbb, X[3], 15);
    HHH(bbb, ccc, ddd, aaa, X[7], 7);
    HHH(aaa, bbb, ccc, ddd, X[0], 12);
    HHH(ddd, aaa, bbb, ccc, X[13], 8);
    HHH(ccc, ddd, aaa, bbb, X[5], 9);
    HHH(bbb, ccc, ddd, aaa, X[10], 11);
    HHH(aaa, bbb, ccc, ddd, X[14], 7);
    HHH(ddd, aaa, bbb, ccc, X[15], 7);
    HHH(ccc, ddd, aaa, bbb, X[8], 12);
    HHH(bbb, ccc, ddd, aaa, X[12], 7);
    HHH(aaa, bbb, ccc, ddd, X[4], 6);
    HHH(ddd, aaa, bbb, ccc, X[9], 15);
    HHH(ccc, ddd, aaa, bbb, X[1], 13);
    HHH(bbb, ccc, ddd, aaa, X[2], 11);

    GGG(aaa, bbb, ccc, ddd, X[15], 9);
    GGG(ddd, aaa, bbb, ccc, X[5], 7);
    GGG(ccc, ddd, aaa, bbb, X[1], 15);
    GGG(bbb, ccc, ddd, aaa, X[3], 11);
    GGG(aaa, bbb, ccc, ddd, X[7], 8);
    GGG(ddd, aaa, bbb, ccc, X[14], 6);
    GGG(ccc, ddd, aaa, bbb, X[6], 6);
    GGG(bbb, ccc, ddd, aaa, X[9], 14);
    GGG(aaa, bbb, ccc, ddd, X[11], 12);
    GGG(ddd, aaa, bbb, ccc, X[8], 13);
    GGG(ccc, ddd, aaa, bbb, X[12], 5);
    GGG(bbb, ccc, ddd, aaa, X[2], 14);
    GGG(aaa, bbb, ccc, ddd, X[10], 13);
    GGG(ddd, aaa, bbb, ccc, X[0], 13);
    GGG(ccc, ddd, aaa, bbb, X[4], 7);
    GGG(bbb, ccc, ddd, aaa, X[13], 5);

    FFF(aaa, bbb, ccc, ddd, X[8], 15);
    FFF(ddd, aaa, bbb, ccc, X[6], 5);
    FFF(ccc, ddd, aaa, bbb, X[4], 8);
    FFF(bbb, ccc, ddd, aaa, X[1], 11);
    FFF(aaa, bbb, ccc, ddd, X[3], 14);
    FFF(ddd, aaa, bbb, ccc, X[11], 14);
    FFF(ccc, ddd, aaa, bbb, X[15], 6);
    FFF(bbb, ccc, ddd, aaa, X[0], 14);
    FFF(aaa, bbb, ccc, ddd, X[5], 6);
    FFF(ddd, aaa, bbb, ccc, X[12], 9);
    FFF(ccc, ddd, aaa, bbb, X[2], 12);
    FFF(bbb, ccc, ddd, aaa, X[13], 9);
    FFF(aaa, bbb, ccc, ddd, X[9], 12);
    FFF(ddd, aaa, bbb, ccc, X[7], 5);
    FFF(ccc, ddd, aaa, bbb, X[10], 15);
    FFF(bbb, ccc, ddd, aaa, X[14], 8);

    ddd += cc + state[1];
    state[1] = state[2] + dd + aaa;
    state[2] = state[3] + aa + bbb;
    state[3] = state[0] + bb + ccc;
    state[0] = ddd;
}

static void prrmd128_hash(const uchar* message, uint len, uchar* hash) {
    uint state[4] = {
        0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u
    };

    uchar block[BLOCK_LEN]; for (int __z = 0; __z < (int)(BLOCK_LEN); ++__z) block[__z] = 0;
    for (uint __i = 0; __i < (len); ++__i) (block)[__i] = (message)[__i];
    block[len] = 0x80;
    const ulong bitlen = (ulong)(len) << 3;
    block[56] = (uchar)(bitlen);
    block[57] = (uchar)(bitlen >> 8);
    block[58] = (uchar)(bitlen >> 16);
    block[59] = (uchar)(bitlen >> 24);
    block[60] = (uchar)(bitlen >> 32);
    block[61] = (uchar)(bitlen >> 40);
    block[62] = (uchar)(bitlen >> 48);
    block[63] = (uchar)(bitlen >> 56);

    prrmd128_compress(state, block);

    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = (uchar)(state[i >> 2] >> ((i & 3) << 3));
}

static int prrmd128_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[16];
  prrmd128_hash(password, (uint)length, hash);
  for (int i = 0; i < 16; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}


__kernel void prrmd128_kernel(__global uchar* result,
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
  if (pass_len >= min_len) {
    if (*g_found) return;
    if (prrmd128_compare(k_hash, attempt, (int)pass_len)) {
      for (uint k = 0; k < pass_len; ++k) result[k] = attempt[k];
      result[pass_len] = 0;
      *g_found = 1;
      return;
    }
  }
  const uint attempt_len = pass_len + 1u;
  if (attempt_len < min_len) return;
  for (uint i = 0; i < dict_length; ++i) {
    attempt[pass_len] = k_dict[i];
    if (*g_found) return;
    if (prrmd128_compare(k_hash, attempt, (int)attempt_len)) {
      for (uint k = 0; k < attempt_len; ++k) result[k] = attempt[k];
      result[attempt_len] = 0;
      *g_found = 1;
      return;
    }
  }
}
