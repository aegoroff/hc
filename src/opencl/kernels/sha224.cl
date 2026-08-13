#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 28
#define BLOCK_LEN 64
#define STATE_LEN 8
#define HASH_LEN (STATE_LEN-1)
#define LENGTH_SIZE 8

static void prsha256_compress(uint state[], uint w[16]) {
#define ROTR32(x, n)  (((0U + (x)) << (32 - (n))) | ((x) >> (n)))
  /* Rolling w[16] instead of schedule[64]: cuts private mem on Intel Arc.
   * BF passwords fit one block, so callers pack BE words + pad into w. */
  uint a = state[0];
  uint b = state[1];
  uint c = state[2];
  uint d = state[3];
  uint e = state[4];
  uint f = state[5];
  uint g = state[6];
  uint h = state[7];
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x428A2F98u + w[0];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x71374491u + w[1];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xB5C0FBCFu + w[2];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xE9B5DBA5u + w[3];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x3956C25Bu + w[4];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x59F111F1u + w[5];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x923F82A4u + w[6];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xAB1C5ED5u + w[7];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xD807AA98u + w[8];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x12835B01u + w[9];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x243185BEu + w[10];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x550C7DC3u + w[11];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x72BE5D74u + w[12];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x80DEB1FEu + w[13];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x9BDC06A7u + w[14];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xC19BF174u + w[15];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[0] = 0U + w[0] + w[9]
            + (ROTR32(w[1], 7) ^ ROTR32(w[1], 18) ^ (w[1] >> 3))
            + (ROTR32(w[14], 17) ^ ROTR32(w[14], 19) ^ (w[14] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xE49B69C1u + w[0];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[1] = 0U + w[1] + w[10]
            + (ROTR32(w[2], 7) ^ ROTR32(w[2], 18) ^ (w[2] >> 3))
            + (ROTR32(w[15], 17) ^ ROTR32(w[15], 19) ^ (w[15] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xEFBE4786u + w[1];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[2] = 0U + w[2] + w[11]
            + (ROTR32(w[3], 7) ^ ROTR32(w[3], 18) ^ (w[3] >> 3))
            + (ROTR32(w[0], 17) ^ ROTR32(w[0], 19) ^ (w[0] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x0FC19DC6u + w[2];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[3] = 0U + w[3] + w[12]
            + (ROTR32(w[4], 7) ^ ROTR32(w[4], 18) ^ (w[4] >> 3))
            + (ROTR32(w[1], 17) ^ ROTR32(w[1], 19) ^ (w[1] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x240CA1CCu + w[3];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[4] = 0U + w[4] + w[13]
            + (ROTR32(w[5], 7) ^ ROTR32(w[5], 18) ^ (w[5] >> 3))
            + (ROTR32(w[2], 17) ^ ROTR32(w[2], 19) ^ (w[2] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x2DE92C6Fu + w[4];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[5] = 0U + w[5] + w[14]
            + (ROTR32(w[6], 7) ^ ROTR32(w[6], 18) ^ (w[6] >> 3))
            + (ROTR32(w[3], 17) ^ ROTR32(w[3], 19) ^ (w[3] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4A7484AAu + w[5];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[6] = 0U + w[6] + w[15]
            + (ROTR32(w[7], 7) ^ ROTR32(w[7], 18) ^ (w[7] >> 3))
            + (ROTR32(w[4], 17) ^ ROTR32(w[4], 19) ^ (w[4] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5CB0A9DCu + w[6];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[7] = 0U + w[7] + w[0]
            + (ROTR32(w[8], 7) ^ ROTR32(w[8], 18) ^ (w[8] >> 3))
            + (ROTR32(w[5], 17) ^ ROTR32(w[5], 19) ^ (w[5] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x76F988DAu + w[7];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[8] = 0U + w[8] + w[1]
            + (ROTR32(w[9], 7) ^ ROTR32(w[9], 18) ^ (w[9] >> 3))
            + (ROTR32(w[6], 17) ^ ROTR32(w[6], 19) ^ (w[6] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x983E5152u + w[8];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[9] = 0U + w[9] + w[2]
            + (ROTR32(w[10], 7) ^ ROTR32(w[10], 18) ^ (w[10] >> 3))
            + (ROTR32(w[7], 17) ^ ROTR32(w[7], 19) ^ (w[7] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xA831C66Du + w[9];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[10] = 0U + w[10] + w[3]
            + (ROTR32(w[11], 7) ^ ROTR32(w[11], 18) ^ (w[11] >> 3))
            + (ROTR32(w[8], 17) ^ ROTR32(w[8], 19) ^ (w[8] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xB00327C8u + w[10];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[11] = 0U + w[11] + w[4]
            + (ROTR32(w[12], 7) ^ ROTR32(w[12], 18) ^ (w[12] >> 3))
            + (ROTR32(w[9], 17) ^ ROTR32(w[9], 19) ^ (w[9] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xBF597FC7u + w[11];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[12] = 0U + w[12] + w[5]
            + (ROTR32(w[13], 7) ^ ROTR32(w[13], 18) ^ (w[13] >> 3))
            + (ROTR32(w[10], 17) ^ ROTR32(w[10], 19) ^ (w[10] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xC6E00BF3u + w[12];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[13] = 0U + w[13] + w[6]
            + (ROTR32(w[14], 7) ^ ROTR32(w[14], 18) ^ (w[14] >> 3))
            + (ROTR32(w[11], 17) ^ ROTR32(w[11], 19) ^ (w[11] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xD5A79147u + w[13];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[14] = 0U + w[14] + w[7]
            + (ROTR32(w[15], 7) ^ ROTR32(w[15], 18) ^ (w[15] >> 3))
            + (ROTR32(w[12], 17) ^ ROTR32(w[12], 19) ^ (w[12] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x06CA6351u + w[14];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[15] = 0U + w[15] + w[8]
            + (ROTR32(w[0], 7) ^ ROTR32(w[0], 18) ^ (w[0] >> 3))
            + (ROTR32(w[13], 17) ^ ROTR32(w[13], 19) ^ (w[13] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x14292967u + w[15];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[0] = 0U + w[0] + w[9]
            + (ROTR32(w[1], 7) ^ ROTR32(w[1], 18) ^ (w[1] >> 3))
            + (ROTR32(w[14], 17) ^ ROTR32(w[14], 19) ^ (w[14] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x27B70A85u + w[0];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[1] = 0U + w[1] + w[10]
            + (ROTR32(w[2], 7) ^ ROTR32(w[2], 18) ^ (w[2] >> 3))
            + (ROTR32(w[15], 17) ^ ROTR32(w[15], 19) ^ (w[15] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x2E1B2138u + w[1];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[2] = 0U + w[2] + w[11]
            + (ROTR32(w[3], 7) ^ ROTR32(w[3], 18) ^ (w[3] >> 3))
            + (ROTR32(w[0], 17) ^ ROTR32(w[0], 19) ^ (w[0] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x4D2C6DFCu + w[2];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[3] = 0U + w[3] + w[12]
            + (ROTR32(w[4], 7) ^ ROTR32(w[4], 18) ^ (w[4] >> 3))
            + (ROTR32(w[1], 17) ^ ROTR32(w[1], 19) ^ (w[1] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x53380D13u + w[3];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[4] = 0U + w[4] + w[13]
            + (ROTR32(w[5], 7) ^ ROTR32(w[5], 18) ^ (w[5] >> 3))
            + (ROTR32(w[2], 17) ^ ROTR32(w[2], 19) ^ (w[2] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x650A7354u + w[4];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[5] = 0U + w[5] + w[14]
            + (ROTR32(w[6], 7) ^ ROTR32(w[6], 18) ^ (w[6] >> 3))
            + (ROTR32(w[3], 17) ^ ROTR32(w[3], 19) ^ (w[3] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x766A0ABBu + w[5];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[6] = 0U + w[6] + w[15]
            + (ROTR32(w[7], 7) ^ ROTR32(w[7], 18) ^ (w[7] >> 3))
            + (ROTR32(w[4], 17) ^ ROTR32(w[4], 19) ^ (w[4] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x81C2C92Eu + w[6];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[7] = 0U + w[7] + w[0]
            + (ROTR32(w[8], 7) ^ ROTR32(w[8], 18) ^ (w[8] >> 3))
            + (ROTR32(w[5], 17) ^ ROTR32(w[5], 19) ^ (w[5] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x92722C85u + w[7];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[8] = 0U + w[8] + w[1]
            + (ROTR32(w[9], 7) ^ ROTR32(w[9], 18) ^ (w[9] >> 3))
            + (ROTR32(w[6], 17) ^ ROTR32(w[6], 19) ^ (w[6] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xA2BFE8A1u + w[8];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[9] = 0U + w[9] + w[2]
            + (ROTR32(w[10], 7) ^ ROTR32(w[10], 18) ^ (w[10] >> 3))
            + (ROTR32(w[7], 17) ^ ROTR32(w[7], 19) ^ (w[7] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xA81A664Bu + w[9];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[10] = 0U + w[10] + w[3]
            + (ROTR32(w[11], 7) ^ ROTR32(w[11], 18) ^ (w[11] >> 3))
            + (ROTR32(w[8], 17) ^ ROTR32(w[8], 19) ^ (w[8] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xC24B8B70u + w[10];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[11] = 0U + w[11] + w[4]
            + (ROTR32(w[12], 7) ^ ROTR32(w[12], 18) ^ (w[12] >> 3))
            + (ROTR32(w[9], 17) ^ ROTR32(w[9], 19) ^ (w[9] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xC76C51A3u + w[11];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[12] = 0U + w[12] + w[5]
            + (ROTR32(w[13], 7) ^ ROTR32(w[13], 18) ^ (w[13] >> 3))
            + (ROTR32(w[10], 17) ^ ROTR32(w[10], 19) ^ (w[10] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xD192E819u + w[12];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[13] = 0U + w[13] + w[6]
            + (ROTR32(w[14], 7) ^ ROTR32(w[14], 18) ^ (w[14] >> 3))
            + (ROTR32(w[11], 17) ^ ROTR32(w[11], 19) ^ (w[11] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xD6990624u + w[13];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[14] = 0U + w[14] + w[7]
            + (ROTR32(w[15], 7) ^ ROTR32(w[15], 18) ^ (w[15] >> 3))
            + (ROTR32(w[12], 17) ^ ROTR32(w[12], 19) ^ (w[12] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xF40E3585u + w[14];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[15] = 0U + w[15] + w[8]
            + (ROTR32(w[0], 7) ^ ROTR32(w[0], 18) ^ (w[0] >> 3))
            + (ROTR32(w[13], 17) ^ ROTR32(w[13], 19) ^ (w[13] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x106AA070u + w[15];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[0] = 0U + w[0] + w[9]
            + (ROTR32(w[1], 7) ^ ROTR32(w[1], 18) ^ (w[1] >> 3))
            + (ROTR32(w[14], 17) ^ ROTR32(w[14], 19) ^ (w[14] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x19A4C116u + w[0];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[1] = 0U + w[1] + w[10]
            + (ROTR32(w[2], 7) ^ ROTR32(w[2], 18) ^ (w[2] >> 3))
            + (ROTR32(w[15], 17) ^ ROTR32(w[15], 19) ^ (w[15] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x1E376C08u + w[1];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[2] = 0U + w[2] + w[11]
            + (ROTR32(w[3], 7) ^ ROTR32(w[3], 18) ^ (w[3] >> 3))
            + (ROTR32(w[0], 17) ^ ROTR32(w[0], 19) ^ (w[0] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x2748774Cu + w[2];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[3] = 0U + w[3] + w[12]
            + (ROTR32(w[4], 7) ^ ROTR32(w[4], 18) ^ (w[4] >> 3))
            + (ROTR32(w[1], 17) ^ ROTR32(w[1], 19) ^ (w[1] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x34B0BCB5u + w[3];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[4] = 0U + w[4] + w[13]
            + (ROTR32(w[5], 7) ^ ROTR32(w[5], 18) ^ (w[5] >> 3))
            + (ROTR32(w[2], 17) ^ ROTR32(w[2], 19) ^ (w[2] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x391C0CB3u + w[4];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[5] = 0U + w[5] + w[14]
            + (ROTR32(w[6], 7) ^ ROTR32(w[6], 18) ^ (w[6] >> 3))
            + (ROTR32(w[3], 17) ^ ROTR32(w[3], 19) ^ (w[3] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4ED8AA4Au + w[5];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[6] = 0U + w[6] + w[15]
            + (ROTR32(w[7], 7) ^ ROTR32(w[7], 18) ^ (w[7] >> 3))
            + (ROTR32(w[4], 17) ^ ROTR32(w[4], 19) ^ (w[4] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5B9CCA4Fu + w[6];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[7] = 0U + w[7] + w[0]
            + (ROTR32(w[8], 7) ^ ROTR32(w[8], 18) ^ (w[8] >> 3))
            + (ROTR32(w[5], 17) ^ ROTR32(w[5], 19) ^ (w[5] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x682E6FF3u + w[7];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w[8] = 0U + w[8] + w[1]
            + (ROTR32(w[9], 7) ^ ROTR32(w[9], 18) ^ (w[9] >> 3))
            + (ROTR32(w[6], 17) ^ ROTR32(w[6], 19) ^ (w[6] >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x748F82EEu + w[8];
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w[9] = 0U + w[9] + w[2]
            + (ROTR32(w[10], 7) ^ ROTR32(w[10], 18) ^ (w[10] >> 3))
            + (ROTR32(w[7], 17) ^ ROTR32(w[7], 19) ^ (w[7] >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x78A5636Fu + w[9];
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w[10] = 0U + w[10] + w[3]
            + (ROTR32(w[11], 7) ^ ROTR32(w[11], 18) ^ (w[11] >> 3))
            + (ROTR32(w[8], 17) ^ ROTR32(w[8], 19) ^ (w[8] >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x84C87814u + w[10];
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w[11] = 0U + w[11] + w[4]
            + (ROTR32(w[12], 7) ^ ROTR32(w[12], 18) ^ (w[12] >> 3))
            + (ROTR32(w[9], 17) ^ ROTR32(w[9], 19) ^ (w[9] >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x8CC70208u + w[11];
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w[12] = 0U + w[12] + w[5]
            + (ROTR32(w[13], 7) ^ ROTR32(w[13], 18) ^ (w[13] >> 3))
            + (ROTR32(w[10], 17) ^ ROTR32(w[10], 19) ^ (w[10] >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x90BEFFFAu + w[12];
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w[13] = 0U + w[13] + w[6]
            + (ROTR32(w[14], 7) ^ ROTR32(w[14], 18) ^ (w[14] >> 3))
            + (ROTR32(w[11], 17) ^ ROTR32(w[11], 19) ^ (w[11] >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xA4506CEBu + w[13];
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w[14] = 0U + w[14] + w[7]
            + (ROTR32(w[15], 7) ^ ROTR32(w[15], 18) ^ (w[15] >> 3))
            + (ROTR32(w[12], 17) ^ ROTR32(w[12], 19) ^ (w[12] >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xBEF9A3F7u + w[14];
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w[15] = 0U + w[15] + w[8]
            + (ROTR32(w[0], 7) ^ ROTR32(w[0], 18) ^ (w[0] >> 3))
            + (ROTR32(w[13], 17) ^ ROTR32(w[13], 19) ^ (w[13] >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xC67178F2u + w[15];
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  state[0] = 0U + state[0] + a;
  state[1] = 0U + state[1] + b;
  state[2] = 0U + state[2] + c;
  state[3] = 0U + state[3] + d;
  state[4] = 0U + state[4] + e;
  state[5] = 0U + state[5] + f;
  state[6] = 0U + state[6] + g;
  state[7] = 0U + state[7] + h;
#undef ROTR32
}

static void prsha256_pack_w(const uchar* message, uint len, uint w[16]) {
  for (int i = 0; i < 16; ++i) w[i] = 0;
  for (uint i = 0; i < len; ++i)
    w[i / 4u] |= ((uint)message[i]) << (24 - (i % 4u) * 8);
  w[len / 4u] |= 0x80u << (24 - (len % 4u) * 8);
  w[15] = len * 8u;
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
  uint w[16];
  prsha256_pack_w(message, len, w);
  prsha256_compress(state, w);
  for (int i = 0; i < HASH_LEN; ++i) hash[i] = state[i];
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

