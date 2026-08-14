#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 32
#define BLOCK_LEN 64
#define STATE_LEN 8
#define LENGTH_SIZE 8

/* One-block BF path: pack + compress with named w0..w15 (no w[16] array / no 16-arg call). */
static void prsha256_process(uint state[], const uchar* message, uint len) {
#define ROTR32(x, n)  (((0U + (x)) << (32 - (n))) | ((x) >> (n)))
  uint w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
  uint w8 = 0, w9 = 0, w10 = 0, w11 = 0, w12 = 0, w13 = 0, w14 = 0, w15 = 0;
  for (uint i = 0; i < len; ++i) {
    uint b = ((uint)message[i]) << (24 - (i % 4u) * 8);
    switch (i / 4u) {
      case 0u: w0 |= b; break;
      case 1u: w1 |= b; break;
      case 2u: w2 |= b; break;
      case 3u: w3 |= b; break;
    }
  }
  {
    uint b = 0x80u << (24 - (len % 4u) * 8);
    switch (len / 4u) {
      case 0u: w0 |= b; break;
      case 1u: w1 |= b; break;
      case 2u: w2 |= b; break;
      case 3u: w3 |= b; break;
      case 4u: w4 |= b; break;
    }
  }
  w15 = len * 8u;
  uint a = state[0];
  uint b = state[1];
  uint c = state[2];
  uint d = state[3];
  uint e = state[4];
  uint f = state[5];
  uint g = state[6];
  uint h = state[7];
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x428A2F98u + w0;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x71374491u + w1;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xB5C0FBCFu + w2;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xE9B5DBA5u + w3;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x3956C25Bu + w4;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x59F111F1u + w5;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x923F82A4u + w6;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xAB1C5ED5u + w7;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xD807AA98u + w8;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x12835B01u + w9;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x243185BEu + w10;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x550C7DC3u + w11;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x72BE5D74u + w12;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x80DEB1FEu + w13;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x9BDC06A7u + w14;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xC19BF174u + w15;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w0 = 0U + w0 + w9
            + (ROTR32(w1, 7) ^ ROTR32(w1, 18) ^ (w1 >> 3))
            + (ROTR32(w14, 17) ^ ROTR32(w14, 19) ^ (w14 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xE49B69C1u + w0;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w1 = 0U + w1 + w10
            + (ROTR32(w2, 7) ^ ROTR32(w2, 18) ^ (w2 >> 3))
            + (ROTR32(w15, 17) ^ ROTR32(w15, 19) ^ (w15 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xEFBE4786u + w1;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w2 = 0U + w2 + w11
            + (ROTR32(w3, 7) ^ ROTR32(w3, 18) ^ (w3 >> 3))
            + (ROTR32(w0, 17) ^ ROTR32(w0, 19) ^ (w0 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x0FC19DC6u + w2;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w3 = 0U + w3 + w12
            + (ROTR32(w4, 7) ^ ROTR32(w4, 18) ^ (w4 >> 3))
            + (ROTR32(w1, 17) ^ ROTR32(w1, 19) ^ (w1 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x240CA1CCu + w3;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w4 = 0U + w4 + w13
            + (ROTR32(w5, 7) ^ ROTR32(w5, 18) ^ (w5 >> 3))
            + (ROTR32(w2, 17) ^ ROTR32(w2, 19) ^ (w2 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x2DE92C6Fu + w4;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w5 = 0U + w5 + w14
            + (ROTR32(w6, 7) ^ ROTR32(w6, 18) ^ (w6 >> 3))
            + (ROTR32(w3, 17) ^ ROTR32(w3, 19) ^ (w3 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4A7484AAu + w5;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w6 = 0U + w6 + w15
            + (ROTR32(w7, 7) ^ ROTR32(w7, 18) ^ (w7 >> 3))
            + (ROTR32(w4, 17) ^ ROTR32(w4, 19) ^ (w4 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5CB0A9DCu + w6;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w7 = 0U + w7 + w0
            + (ROTR32(w8, 7) ^ ROTR32(w8, 18) ^ (w8 >> 3))
            + (ROTR32(w5, 17) ^ ROTR32(w5, 19) ^ (w5 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x76F988DAu + w7;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w8 = 0U + w8 + w1
            + (ROTR32(w9, 7) ^ ROTR32(w9, 18) ^ (w9 >> 3))
            + (ROTR32(w6, 17) ^ ROTR32(w6, 19) ^ (w6 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x983E5152u + w8;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w9 = 0U + w9 + w2
            + (ROTR32(w10, 7) ^ ROTR32(w10, 18) ^ (w10 >> 3))
            + (ROTR32(w7, 17) ^ ROTR32(w7, 19) ^ (w7 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xA831C66Du + w9;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w10 = 0U + w10 + w3
            + (ROTR32(w11, 7) ^ ROTR32(w11, 18) ^ (w11 >> 3))
            + (ROTR32(w8, 17) ^ ROTR32(w8, 19) ^ (w8 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xB00327C8u + w10;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w11 = 0U + w11 + w4
            + (ROTR32(w12, 7) ^ ROTR32(w12, 18) ^ (w12 >> 3))
            + (ROTR32(w9, 17) ^ ROTR32(w9, 19) ^ (w9 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xBF597FC7u + w11;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w12 = 0U + w12 + w5
            + (ROTR32(w13, 7) ^ ROTR32(w13, 18) ^ (w13 >> 3))
            + (ROTR32(w10, 17) ^ ROTR32(w10, 19) ^ (w10 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xC6E00BF3u + w12;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w13 = 0U + w13 + w6
            + (ROTR32(w14, 7) ^ ROTR32(w14, 18) ^ (w14 >> 3))
            + (ROTR32(w11, 17) ^ ROTR32(w11, 19) ^ (w11 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xD5A79147u + w13;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w14 = 0U + w14 + w7
            + (ROTR32(w15, 7) ^ ROTR32(w15, 18) ^ (w15 >> 3))
            + (ROTR32(w12, 17) ^ ROTR32(w12, 19) ^ (w12 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x06CA6351u + w14;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w15 = 0U + w15 + w8
            + (ROTR32(w0, 7) ^ ROTR32(w0, 18) ^ (w0 >> 3))
            + (ROTR32(w13, 17) ^ ROTR32(w13, 19) ^ (w13 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x14292967u + w15;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w0 = 0U + w0 + w9
            + (ROTR32(w1, 7) ^ ROTR32(w1, 18) ^ (w1 >> 3))
            + (ROTR32(w14, 17) ^ ROTR32(w14, 19) ^ (w14 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x27B70A85u + w0;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w1 = 0U + w1 + w10
            + (ROTR32(w2, 7) ^ ROTR32(w2, 18) ^ (w2 >> 3))
            + (ROTR32(w15, 17) ^ ROTR32(w15, 19) ^ (w15 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x2E1B2138u + w1;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w2 = 0U + w2 + w11
            + (ROTR32(w3, 7) ^ ROTR32(w3, 18) ^ (w3 >> 3))
            + (ROTR32(w0, 17) ^ ROTR32(w0, 19) ^ (w0 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x4D2C6DFCu + w2;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w3 = 0U + w3 + w12
            + (ROTR32(w4, 7) ^ ROTR32(w4, 18) ^ (w4 >> 3))
            + (ROTR32(w1, 17) ^ ROTR32(w1, 19) ^ (w1 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x53380D13u + w3;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w4 = 0U + w4 + w13
            + (ROTR32(w5, 7) ^ ROTR32(w5, 18) ^ (w5 >> 3))
            + (ROTR32(w2, 17) ^ ROTR32(w2, 19) ^ (w2 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x650A7354u + w4;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w5 = 0U + w5 + w14
            + (ROTR32(w6, 7) ^ ROTR32(w6, 18) ^ (w6 >> 3))
            + (ROTR32(w3, 17) ^ ROTR32(w3, 19) ^ (w3 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x766A0ABBu + w5;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w6 = 0U + w6 + w15
            + (ROTR32(w7, 7) ^ ROTR32(w7, 18) ^ (w7 >> 3))
            + (ROTR32(w4, 17) ^ ROTR32(w4, 19) ^ (w4 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x81C2C92Eu + w6;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w7 = 0U + w7 + w0
            + (ROTR32(w8, 7) ^ ROTR32(w8, 18) ^ (w8 >> 3))
            + (ROTR32(w5, 17) ^ ROTR32(w5, 19) ^ (w5 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x92722C85u + w7;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w8 = 0U + w8 + w1
            + (ROTR32(w9, 7) ^ ROTR32(w9, 18) ^ (w9 >> 3))
            + (ROTR32(w6, 17) ^ ROTR32(w6, 19) ^ (w6 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0xA2BFE8A1u + w8;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w9 = 0U + w9 + w2
            + (ROTR32(w10, 7) ^ ROTR32(w10, 18) ^ (w10 >> 3))
            + (ROTR32(w7, 17) ^ ROTR32(w7, 19) ^ (w7 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0xA81A664Bu + w9;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w10 = 0U + w10 + w3
            + (ROTR32(w11, 7) ^ ROTR32(w11, 18) ^ (w11 >> 3))
            + (ROTR32(w8, 17) ^ ROTR32(w8, 19) ^ (w8 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0xC24B8B70u + w10;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w11 = 0U + w11 + w4
            + (ROTR32(w12, 7) ^ ROTR32(w12, 18) ^ (w12 >> 3))
            + (ROTR32(w9, 17) ^ ROTR32(w9, 19) ^ (w9 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0xC76C51A3u + w11;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w12 = 0U + w12 + w5
            + (ROTR32(w13, 7) ^ ROTR32(w13, 18) ^ (w13 >> 3))
            + (ROTR32(w10, 17) ^ ROTR32(w10, 19) ^ (w10 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0xD192E819u + w12;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w13 = 0U + w13 + w6
            + (ROTR32(w14, 7) ^ ROTR32(w14, 18) ^ (w14 >> 3))
            + (ROTR32(w11, 17) ^ ROTR32(w11, 19) ^ (w11 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xD6990624u + w13;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w14 = 0U + w14 + w7
            + (ROTR32(w15, 7) ^ ROTR32(w15, 18) ^ (w15 >> 3))
            + (ROTR32(w12, 17) ^ ROTR32(w12, 19) ^ (w12 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xF40E3585u + w14;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w15 = 0U + w15 + w8
            + (ROTR32(w0, 7) ^ ROTR32(w0, 18) ^ (w0 >> 3))
            + (ROTR32(w13, 17) ^ ROTR32(w13, 19) ^ (w13 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x106AA070u + w15;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w0 = 0U + w0 + w9
            + (ROTR32(w1, 7) ^ ROTR32(w1, 18) ^ (w1 >> 3))
            + (ROTR32(w14, 17) ^ ROTR32(w14, 19) ^ (w14 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x19A4C116u + w0;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w1 = 0U + w1 + w10
            + (ROTR32(w2, 7) ^ ROTR32(w2, 18) ^ (w2 >> 3))
            + (ROTR32(w15, 17) ^ ROTR32(w15, 19) ^ (w15 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x1E376C08u + w1;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w2 = 0U + w2 + w11
            + (ROTR32(w3, 7) ^ ROTR32(w3, 18) ^ (w3 >> 3))
            + (ROTR32(w0, 17) ^ ROTR32(w0, 19) ^ (w0 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x2748774Cu + w2;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w3 = 0U + w3 + w12
            + (ROTR32(w4, 7) ^ ROTR32(w4, 18) ^ (w4 >> 3))
            + (ROTR32(w1, 17) ^ ROTR32(w1, 19) ^ (w1 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x34B0BCB5u + w3;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w4 = 0U + w4 + w13
            + (ROTR32(w5, 7) ^ ROTR32(w5, 18) ^ (w5 >> 3))
            + (ROTR32(w2, 17) ^ ROTR32(w2, 19) ^ (w2 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x391C0CB3u + w4;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w5 = 0U + w5 + w14
            + (ROTR32(w6, 7) ^ ROTR32(w6, 18) ^ (w6 >> 3))
            + (ROTR32(w3, 17) ^ ROTR32(w3, 19) ^ (w3 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0x4ED8AA4Au + w5;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w6 = 0U + w6 + w15
            + (ROTR32(w7, 7) ^ ROTR32(w7, 18) ^ (w7 >> 3))
            + (ROTR32(w4, 17) ^ ROTR32(w4, 19) ^ (w4 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0x5B9CCA4Fu + w6;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w7 = 0U + w7 + w0
            + (ROTR32(w8, 7) ^ ROTR32(w8, 18) ^ (w8 >> 3))
            + (ROTR32(w5, 17) ^ ROTR32(w5, 19) ^ (w5 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0x682E6FF3u + w7;
  e = 0U + e + a;
  a = 0U + a + (ROTR32(b, 2) ^ ROTR32(b, 13) ^ ROTR32(b, 22)) + ((b & (c | d)) | (c & d));
  w8 = 0U + w8 + w1
            + (ROTR32(w9, 7) ^ ROTR32(w9, 18) ^ (w9 >> 3))
            + (ROTR32(w6, 17) ^ ROTR32(w6, 19) ^ (w6 >> 10));
  h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + 0x748F82EEu + w8;
  d = 0U + d + h;
  h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));
  w9 = 0U + w9 + w2
            + (ROTR32(w10, 7) ^ ROTR32(w10, 18) ^ (w10 >> 3))
            + (ROTR32(w7, 17) ^ ROTR32(w7, 19) ^ (w7 >> 10));
  g = 0U + g + (ROTR32(d, 6) ^ ROTR32(d, 11) ^ ROTR32(d, 25)) + (f ^ (d & (e ^ f))) + 0x78A5636Fu + w9;
  c = 0U + c + g;
  g = 0U + g + (ROTR32(h, 2) ^ ROTR32(h, 13) ^ ROTR32(h, 22)) + ((h & (a | b)) | (a & b));
  w10 = 0U + w10 + w3
            + (ROTR32(w11, 7) ^ ROTR32(w11, 18) ^ (w11 >> 3))
            + (ROTR32(w8, 17) ^ ROTR32(w8, 19) ^ (w8 >> 10));
  f = 0U + f + (ROTR32(c, 6) ^ ROTR32(c, 11) ^ ROTR32(c, 25)) + (e ^ (c & (d ^ e))) + 0x84C87814u + w10;
  b = 0U + b + f;
  f = 0U + f + (ROTR32(g, 2) ^ ROTR32(g, 13) ^ ROTR32(g, 22)) + ((g & (h | a)) | (h & a));
  w11 = 0U + w11 + w4
            + (ROTR32(w12, 7) ^ ROTR32(w12, 18) ^ (w12 >> 3))
            + (ROTR32(w9, 17) ^ ROTR32(w9, 19) ^ (w9 >> 10));
  e = 0U + e + (ROTR32(b, 6) ^ ROTR32(b, 11) ^ ROTR32(b, 25)) + (d ^ (b & (c ^ d))) + 0x8CC70208u + w11;
  a = 0U + a + e;
  e = 0U + e + (ROTR32(f, 2) ^ ROTR32(f, 13) ^ ROTR32(f, 22)) + ((f & (g | h)) | (g & h));
  w12 = 0U + w12 + w5
            + (ROTR32(w13, 7) ^ ROTR32(w13, 18) ^ (w13 >> 3))
            + (ROTR32(w10, 17) ^ ROTR32(w10, 19) ^ (w10 >> 10));
  d = 0U + d + (ROTR32(a, 6) ^ ROTR32(a, 11) ^ ROTR32(a, 25)) + (c ^ (a & (b ^ c))) + 0x90BEFFFAu + w12;
  h = 0U + h + d;
  d = 0U + d + (ROTR32(e, 2) ^ ROTR32(e, 13) ^ ROTR32(e, 22)) + ((e & (f | g)) | (f & g));
  w13 = 0U + w13 + w6
            + (ROTR32(w14, 7) ^ ROTR32(w14, 18) ^ (w14 >> 3))
            + (ROTR32(w11, 17) ^ ROTR32(w11, 19) ^ (w11 >> 10));
  c = 0U + c + (ROTR32(h, 6) ^ ROTR32(h, 11) ^ ROTR32(h, 25)) + (b ^ (h & (a ^ b))) + 0xA4506CEBu + w13;
  g = 0U + g + c;
  c = 0U + c + (ROTR32(d, 2) ^ ROTR32(d, 13) ^ ROTR32(d, 22)) + ((d & (e | f)) | (e & f));
  w14 = 0U + w14 + w7
            + (ROTR32(w15, 7) ^ ROTR32(w15, 18) ^ (w15 >> 3))
            + (ROTR32(w12, 17) ^ ROTR32(w12, 19) ^ (w12 >> 10));
  b = 0U + b + (ROTR32(g, 6) ^ ROTR32(g, 11) ^ ROTR32(g, 25)) + (a ^ (g & (h ^ a))) + 0xBEF9A3F7u + w14;
  f = 0U + f + b;
  b = 0U + b + (ROTR32(c, 2) ^ ROTR32(c, 13) ^ ROTR32(c, 22)) + ((c & (d | e)) | (d & e));
  w15 = 0U + w15 + w8
            + (ROTR32(w0, 7) ^ ROTR32(w0, 18) ^ (w0 >> 3))
            + (ROTR32(w13, 17) ^ ROTR32(w13, 19) ^ (w13 >> 10));
  a = 0U + a + (ROTR32(f, 6) ^ ROTR32(f, 11) ^ ROTR32(f, 25)) + (h ^ (f & (g ^ h))) + 0xC67178F2u + w15;
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

static void prsha256_hash(const uchar* message, uint len, uint* hash) {
  hash[0] = 0x6A09E667u;
  hash[1] = 0xBB67AE85u;
  hash[2] = 0x3C6EF372u;
  hash[3] = 0xA54FF53Au;
  hash[4] = 0x510E527Fu;
  hash[5] = 0x9B05688Cu;
  hash[6] = 0x1F83D9ABu;
  hash[7] = 0x5BE0CD19u;
  prsha256_process(hash, message, len);
}



static int prsha256_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uint hash[STATE_LEN];
  prsha256_hash(password, (uint)length, hash);
  int result = 1;
  for (int i = 0; i < STATE_LEN && result; ++i) {
    result &= hash[i] == ((uint)k_hash[3 + i * 4] | (uint)k_hash[2 + i * 4] << 8 | (uint)k_hash[1 + i * 4] << 16 | (uint)k_hash[0 + i * 4] << 24);
  }
  return result;
}


__kernel void prsha256_kernel(__global uchar* result,
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
      if (prsha256_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prsha256_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}

