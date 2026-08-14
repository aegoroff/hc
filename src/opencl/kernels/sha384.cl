#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 48
#define BLOCK_LEN 128
#define STATE_LEN 8
#define HASH_WORDS 6
#define LENGTH_SIZE 16

/* One-block BF path: pack + compress with named w0..w15. */
static void prsha512_process(ulong state[], const uchar* message, uint len) {
#define ROTR64(x, n)  (((0UL + (x)) << (64 - (n))) | ((x) >> (n)))
  ulong w0 = 0, w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 = 0, w6 = 0, w7 = 0;
  ulong w8 = 0, w9 = 0, w10 = 0, w11 = 0, w12 = 0, w13 = 0, w14 = 0, w15 = 0;
  for (uint i = 0; i < len; ++i) {
    ulong b = ((ulong)message[i]) << (56 - (i % 8u) * 8);
    switch (i / 8u) {
      case 0u: w0 |= b; break;
      case 1u: w1 |= b; break;
    }
  }
  {
    ulong b = 0x80UL << (56 - (len % 8u) * 8);
    switch (len / 8u) {
      case 0u: w0 |= b; break;
      case 1u: w1 |= b; break;
      case 2u: w2 |= b; break;
    }
  }
  w15 = (ulong)len * 8UL;
  ulong a = state[0];
  ulong b = state[1];
  ulong c = state[2];
  ulong d = state[3];
  ulong e = state[4];
  ulong f = state[5];
  ulong g = state[6];
  ulong h = state[7];
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x428A2F98D728AE22UL + w0;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x7137449123EF65CDUL + w1;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xB5C0FBCFEC4D3B2FUL + w2;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xE9B5DBA58189DBBCUL + w3;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x3956C25BF348B538UL + w4;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x59F111F1B605D019UL + w5;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x923F82A4AF194F9BUL + w6;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xAB1C5ED5DA6D8118UL + w7;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xD807AA98A3030242UL + w8;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x12835B0145706FBEUL + w9;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x243185BE4EE4B28CUL + w10;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x550C7DC3D5FFB4E2UL + w11;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x72BE5D74F27B896FUL + w12;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x80DEB1FE3B1696B1UL + w13;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x9BDC06A725C71235UL + w14;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xC19BF174CF692694UL + w15;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w0 = 0UL + w0 + w9
            + (ROTR64(w1, 1) ^ ROTR64(w1, 8) ^ (w1 >> 7))
            + (ROTR64(w14, 19) ^ ROTR64(w14, 61) ^ (w14 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xE49B69C19EF14AD2UL + w0;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w1 = 0UL + w1 + w10
            + (ROTR64(w2, 1) ^ ROTR64(w2, 8) ^ (w2 >> 7))
            + (ROTR64(w15, 19) ^ ROTR64(w15, 61) ^ (w15 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xEFBE4786384F25E3UL + w1;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w2 = 0UL + w2 + w11
            + (ROTR64(w3, 1) ^ ROTR64(w3, 8) ^ (w3 >> 7))
            + (ROTR64(w0, 19) ^ ROTR64(w0, 61) ^ (w0 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x0FC19DC68B8CD5B5UL + w2;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w3 = 0UL + w3 + w12
            + (ROTR64(w4, 1) ^ ROTR64(w4, 8) ^ (w4 >> 7))
            + (ROTR64(w1, 19) ^ ROTR64(w1, 61) ^ (w1 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x240CA1CC77AC9C65UL + w3;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w4 = 0UL + w4 + w13
            + (ROTR64(w5, 1) ^ ROTR64(w5, 8) ^ (w5 >> 7))
            + (ROTR64(w2, 19) ^ ROTR64(w2, 61) ^ (w2 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x2DE92C6F592B0275UL + w4;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w5 = 0UL + w5 + w14
            + (ROTR64(w6, 1) ^ ROTR64(w6, 8) ^ (w6 >> 7))
            + (ROTR64(w3, 19) ^ ROTR64(w3, 61) ^ (w3 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x4A7484AA6EA6E483UL + w5;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w6 = 0UL + w6 + w15
            + (ROTR64(w7, 1) ^ ROTR64(w7, 8) ^ (w7 >> 7))
            + (ROTR64(w4, 19) ^ ROTR64(w4, 61) ^ (w4 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5CB0A9DCBD41FBD4UL + w6;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w7 = 0UL + w7 + w0
            + (ROTR64(w8, 1) ^ ROTR64(w8, 8) ^ (w8 >> 7))
            + (ROTR64(w5, 19) ^ ROTR64(w5, 61) ^ (w5 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x76F988DA831153B5UL + w7;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w8 = 0UL + w8 + w1
            + (ROTR64(w9, 1) ^ ROTR64(w9, 8) ^ (w9 >> 7))
            + (ROTR64(w6, 19) ^ ROTR64(w6, 61) ^ (w6 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x983E5152EE66DFABUL + w8;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w9 = 0UL + w9 + w2
            + (ROTR64(w10, 1) ^ ROTR64(w10, 8) ^ (w10 >> 7))
            + (ROTR64(w7, 19) ^ ROTR64(w7, 61) ^ (w7 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xA831C66D2DB43210UL + w9;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w10 = 0UL + w10 + w3
            + (ROTR64(w11, 1) ^ ROTR64(w11, 8) ^ (w11 >> 7))
            + (ROTR64(w8, 19) ^ ROTR64(w8, 61) ^ (w8 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xB00327C898FB213FUL + w10;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w11 = 0UL + w11 + w4
            + (ROTR64(w12, 1) ^ ROTR64(w12, 8) ^ (w12 >> 7))
            + (ROTR64(w9, 19) ^ ROTR64(w9, 61) ^ (w9 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xBF597FC7BEEF0EE4UL + w11;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w12 = 0UL + w12 + w5
            + (ROTR64(w13, 1) ^ ROTR64(w13, 8) ^ (w13 >> 7))
            + (ROTR64(w10, 19) ^ ROTR64(w10, 61) ^ (w10 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0xC6E00BF33DA88FC2UL + w12;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w13 = 0UL + w13 + w6
            + (ROTR64(w14, 1) ^ ROTR64(w14, 8) ^ (w14 >> 7))
            + (ROTR64(w11, 19) ^ ROTR64(w11, 61) ^ (w11 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xD5A79147930AA725UL + w13;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w14 = 0UL + w14 + w7
            + (ROTR64(w15, 1) ^ ROTR64(w15, 8) ^ (w15 >> 7))
            + (ROTR64(w12, 19) ^ ROTR64(w12, 61) ^ (w12 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x06CA6351E003826FUL + w14;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w15 = 0UL + w15 + w8
            + (ROTR64(w0, 1) ^ ROTR64(w0, 8) ^ (w0 >> 7))
            + (ROTR64(w13, 19) ^ ROTR64(w13, 61) ^ (w13 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x142929670A0E6E70UL + w15;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w0 = 0UL + w0 + w9
            + (ROTR64(w1, 1) ^ ROTR64(w1, 8) ^ (w1 >> 7))
            + (ROTR64(w14, 19) ^ ROTR64(w14, 61) ^ (w14 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x27B70A8546D22FFCUL + w0;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w1 = 0UL + w1 + w10
            + (ROTR64(w2, 1) ^ ROTR64(w2, 8) ^ (w2 >> 7))
            + (ROTR64(w15, 19) ^ ROTR64(w15, 61) ^ (w15 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x2E1B21385C26C926UL + w1;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w2 = 0UL + w2 + w11
            + (ROTR64(w3, 1) ^ ROTR64(w3, 8) ^ (w3 >> 7))
            + (ROTR64(w0, 19) ^ ROTR64(w0, 61) ^ (w0 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x4D2C6DFC5AC42AEDUL + w2;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w3 = 0UL + w3 + w12
            + (ROTR64(w4, 1) ^ ROTR64(w4, 8) ^ (w4 >> 7))
            + (ROTR64(w1, 19) ^ ROTR64(w1, 61) ^ (w1 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x53380D139D95B3DFUL + w3;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w4 = 0UL + w4 + w13
            + (ROTR64(w5, 1) ^ ROTR64(w5, 8) ^ (w5 >> 7))
            + (ROTR64(w2, 19) ^ ROTR64(w2, 61) ^ (w2 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x650A73548BAF63DEUL + w4;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w5 = 0UL + w5 + w14
            + (ROTR64(w6, 1) ^ ROTR64(w6, 8) ^ (w6 >> 7))
            + (ROTR64(w3, 19) ^ ROTR64(w3, 61) ^ (w3 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x766A0ABB3C77B2A8UL + w5;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w6 = 0UL + w6 + w15
            + (ROTR64(w7, 1) ^ ROTR64(w7, 8) ^ (w7 >> 7))
            + (ROTR64(w4, 19) ^ ROTR64(w4, 61) ^ (w4 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x81C2C92E47EDAEE6UL + w6;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w7 = 0UL + w7 + w0
            + (ROTR64(w8, 1) ^ ROTR64(w8, 8) ^ (w8 >> 7))
            + (ROTR64(w5, 19) ^ ROTR64(w5, 61) ^ (w5 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x92722C851482353BUL + w7;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w8 = 0UL + w8 + w1
            + (ROTR64(w9, 1) ^ ROTR64(w9, 8) ^ (w9 >> 7))
            + (ROTR64(w6, 19) ^ ROTR64(w6, 61) ^ (w6 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xA2BFE8A14CF10364UL + w8;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w9 = 0UL + w9 + w2
            + (ROTR64(w10, 1) ^ ROTR64(w10, 8) ^ (w10 >> 7))
            + (ROTR64(w7, 19) ^ ROTR64(w7, 61) ^ (w7 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xA81A664BBC423001UL + w9;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w10 = 0UL + w10 + w3
            + (ROTR64(w11, 1) ^ ROTR64(w11, 8) ^ (w11 >> 7))
            + (ROTR64(w8, 19) ^ ROTR64(w8, 61) ^ (w8 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xC24B8B70D0F89791UL + w10;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w11 = 0UL + w11 + w4
            + (ROTR64(w12, 1) ^ ROTR64(w12, 8) ^ (w12 >> 7))
            + (ROTR64(w9, 19) ^ ROTR64(w9, 61) ^ (w9 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xC76C51A30654BE30UL + w11;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w12 = 0UL + w12 + w5
            + (ROTR64(w13, 1) ^ ROTR64(w13, 8) ^ (w13 >> 7))
            + (ROTR64(w10, 19) ^ ROTR64(w10, 61) ^ (w10 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0xD192E819D6EF5218UL + w12;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w13 = 0UL + w13 + w6
            + (ROTR64(w14, 1) ^ ROTR64(w14, 8) ^ (w14 >> 7))
            + (ROTR64(w11, 19) ^ ROTR64(w11, 61) ^ (w11 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xD69906245565A910UL + w13;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w14 = 0UL + w14 + w7
            + (ROTR64(w15, 1) ^ ROTR64(w15, 8) ^ (w15 >> 7))
            + (ROTR64(w12, 19) ^ ROTR64(w12, 61) ^ (w12 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0xF40E35855771202AUL + w14;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w15 = 0UL + w15 + w8
            + (ROTR64(w0, 1) ^ ROTR64(w0, 8) ^ (w0 >> 7))
            + (ROTR64(w13, 19) ^ ROTR64(w13, 61) ^ (w13 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x106AA07032BBD1B8UL + w15;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w0 = 0UL + w0 + w9
            + (ROTR64(w1, 1) ^ ROTR64(w1, 8) ^ (w1 >> 7))
            + (ROTR64(w14, 19) ^ ROTR64(w14, 61) ^ (w14 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x19A4C116B8D2D0C8UL + w0;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w1 = 0UL + w1 + w10
            + (ROTR64(w2, 1) ^ ROTR64(w2, 8) ^ (w2 >> 7))
            + (ROTR64(w15, 19) ^ ROTR64(w15, 61) ^ (w15 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x1E376C085141AB53UL + w1;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w2 = 0UL + w2 + w11
            + (ROTR64(w3, 1) ^ ROTR64(w3, 8) ^ (w3 >> 7))
            + (ROTR64(w0, 19) ^ ROTR64(w0, 61) ^ (w0 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x2748774CDF8EEB99UL + w2;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w3 = 0UL + w3 + w12
            + (ROTR64(w4, 1) ^ ROTR64(w4, 8) ^ (w4 >> 7))
            + (ROTR64(w1, 19) ^ ROTR64(w1, 61) ^ (w1 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x34B0BCB5E19B48A8UL + w3;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w4 = 0UL + w4 + w13
            + (ROTR64(w5, 1) ^ ROTR64(w5, 8) ^ (w5 >> 7))
            + (ROTR64(w2, 19) ^ ROTR64(w2, 61) ^ (w2 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x391C0CB3C5C95A63UL + w4;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w5 = 0UL + w5 + w14
            + (ROTR64(w6, 1) ^ ROTR64(w6, 8) ^ (w6 >> 7))
            + (ROTR64(w3, 19) ^ ROTR64(w3, 61) ^ (w3 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x4ED8AA4AE3418ACBUL + w5;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w6 = 0UL + w6 + w15
            + (ROTR64(w7, 1) ^ ROTR64(w7, 8) ^ (w7 >> 7))
            + (ROTR64(w4, 19) ^ ROTR64(w4, 61) ^ (w4 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5B9CCA4F7763E373UL + w6;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w7 = 0UL + w7 + w0
            + (ROTR64(w8, 1) ^ ROTR64(w8, 8) ^ (w8 >> 7))
            + (ROTR64(w5, 19) ^ ROTR64(w5, 61) ^ (w5 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x682E6FF3D6B2B8A3UL + w7;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w8 = 0UL + w8 + w1
            + (ROTR64(w9, 1) ^ ROTR64(w9, 8) ^ (w9 >> 7))
            + (ROTR64(w6, 19) ^ ROTR64(w6, 61) ^ (w6 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x748F82EE5DEFB2FCUL + w8;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w9 = 0UL + w9 + w2
            + (ROTR64(w10, 1) ^ ROTR64(w10, 8) ^ (w10 >> 7))
            + (ROTR64(w7, 19) ^ ROTR64(w7, 61) ^ (w7 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x78A5636F43172F60UL + w9;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w10 = 0UL + w10 + w3
            + (ROTR64(w11, 1) ^ ROTR64(w11, 8) ^ (w11 >> 7))
            + (ROTR64(w8, 19) ^ ROTR64(w8, 61) ^ (w8 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x84C87814A1F0AB72UL + w10;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w11 = 0UL + w11 + w4
            + (ROTR64(w12, 1) ^ ROTR64(w12, 8) ^ (w12 >> 7))
            + (ROTR64(w9, 19) ^ ROTR64(w9, 61) ^ (w9 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x8CC702081A6439ECUL + w11;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w12 = 0UL + w12 + w5
            + (ROTR64(w13, 1) ^ ROTR64(w13, 8) ^ (w13 >> 7))
            + (ROTR64(w10, 19) ^ ROTR64(w10, 61) ^ (w10 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x90BEFFFA23631E28UL + w12;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w13 = 0UL + w13 + w6
            + (ROTR64(w14, 1) ^ ROTR64(w14, 8) ^ (w14 >> 7))
            + (ROTR64(w11, 19) ^ ROTR64(w11, 61) ^ (w11 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xA4506CEBDE82BDE9UL + w13;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w14 = 0UL + w14 + w7
            + (ROTR64(w15, 1) ^ ROTR64(w15, 8) ^ (w15 >> 7))
            + (ROTR64(w12, 19) ^ ROTR64(w12, 61) ^ (w12 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0xBEF9A3F7B2C67915UL + w14;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w15 = 0UL + w15 + w8
            + (ROTR64(w0, 1) ^ ROTR64(w0, 8) ^ (w0 >> 7))
            + (ROTR64(w13, 19) ^ ROTR64(w13, 61) ^ (w13 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xC67178F2E372532BUL + w15;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w0 = 0UL + w0 + w9
            + (ROTR64(w1, 1) ^ ROTR64(w1, 8) ^ (w1 >> 7))
            + (ROTR64(w14, 19) ^ ROTR64(w14, 61) ^ (w14 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xCA273ECEEA26619CUL + w0;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w1 = 0UL + w1 + w10
            + (ROTR64(w2, 1) ^ ROTR64(w2, 8) ^ (w2 >> 7))
            + (ROTR64(w15, 19) ^ ROTR64(w15, 61) ^ (w15 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xD186B8C721C0C207UL + w1;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w2 = 0UL + w2 + w11
            + (ROTR64(w3, 1) ^ ROTR64(w3, 8) ^ (w3 >> 7))
            + (ROTR64(w0, 19) ^ ROTR64(w0, 61) ^ (w0 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xEADA7DD6CDE0EB1EUL + w2;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w3 = 0UL + w3 + w12
            + (ROTR64(w4, 1) ^ ROTR64(w4, 8) ^ (w4 >> 7))
            + (ROTR64(w1, 19) ^ ROTR64(w1, 61) ^ (w1 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xF57D4F7FEE6ED178UL + w3;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w4 = 0UL + w4 + w13
            + (ROTR64(w5, 1) ^ ROTR64(w5, 8) ^ (w5 >> 7))
            + (ROTR64(w2, 19) ^ ROTR64(w2, 61) ^ (w2 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x06F067AA72176FBAUL + w4;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w5 = 0UL + w5 + w14
            + (ROTR64(w6, 1) ^ ROTR64(w6, 8) ^ (w6 >> 7))
            + (ROTR64(w3, 19) ^ ROTR64(w3, 61) ^ (w3 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x0A637DC5A2C898A6UL + w5;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w6 = 0UL + w6 + w15
            + (ROTR64(w7, 1) ^ ROTR64(w7, 8) ^ (w7 >> 7))
            + (ROTR64(w4, 19) ^ ROTR64(w4, 61) ^ (w4 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x113F9804BEF90DAEUL + w6;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w7 = 0UL + w7 + w0
            + (ROTR64(w8, 1) ^ ROTR64(w8, 8) ^ (w8 >> 7))
            + (ROTR64(w5, 19) ^ ROTR64(w5, 61) ^ (w5 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x1B710B35131C471BUL + w7;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w8 = 0UL + w8 + w1
            + (ROTR64(w9, 1) ^ ROTR64(w9, 8) ^ (w9 >> 7))
            + (ROTR64(w6, 19) ^ ROTR64(w6, 61) ^ (w6 >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x28DB77F523047D84UL + w8;
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w9 = 0UL + w9 + w2
            + (ROTR64(w10, 1) ^ ROTR64(w10, 8) ^ (w10 >> 7))
            + (ROTR64(w7, 19) ^ ROTR64(w7, 61) ^ (w7 >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x32CAAB7B40C72493UL + w9;
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w10 = 0UL + w10 + w3
            + (ROTR64(w11, 1) ^ ROTR64(w11, 8) ^ (w11 >> 7))
            + (ROTR64(w8, 19) ^ ROTR64(w8, 61) ^ (w8 >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x3C9EBE0A15C9BEBCUL + w10;
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w11 = 0UL + w11 + w4
            + (ROTR64(w12, 1) ^ ROTR64(w12, 8) ^ (w12 >> 7))
            + (ROTR64(w9, 19) ^ ROTR64(w9, 61) ^ (w9 >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x431D67C49C100D4CUL + w11;
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w12 = 0UL + w12 + w5
            + (ROTR64(w13, 1) ^ ROTR64(w13, 8) ^ (w13 >> 7))
            + (ROTR64(w10, 19) ^ ROTR64(w10, 61) ^ (w10 >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x4CC5D4BECB3E42B6UL + w12;
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w13 = 0UL + w13 + w6
            + (ROTR64(w14, 1) ^ ROTR64(w14, 8) ^ (w14 >> 7))
            + (ROTR64(w11, 19) ^ ROTR64(w11, 61) ^ (w11 >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x597F299CFC657E2AUL + w13;
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w14 = 0UL + w14 + w7
            + (ROTR64(w15, 1) ^ ROTR64(w15, 8) ^ (w15 >> 7))
            + (ROTR64(w12, 19) ^ ROTR64(w12, 61) ^ (w12 >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5FCB6FAB3AD6FAECUL + w14;
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w15 = 0UL + w15 + w8
            + (ROTR64(w0, 1) ^ ROTR64(w0, 8) ^ (w0 >> 7))
            + (ROTR64(w13, 19) ^ ROTR64(w13, 61) ^ (w13 >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x6C44198C4A475817UL + w15;
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  state[0] = 0UL + state[0] + a;
  state[1] = 0UL + state[1] + b;
  state[2] = 0UL + state[2] + c;
  state[3] = 0UL + state[3] + d;
  state[4] = 0UL + state[4] + e;
  state[5] = 0UL + state[5] + f;
  state[6] = 0UL + state[6] + g;
  state[7] = 0UL + state[7] + h;
#undef ROTR64
}



static void prsha384_hash(const uchar* message, uint len, ulong* hash) {
  ulong state[STATE_LEN];
  state[0] = 0xCBBB9D5DC1059ED8UL;
  state[1] = 0x629A292A367CD507UL;
  state[2] = 0x9159015A3070DD17UL;
  state[3] = 0x152FECD8F70E5939UL;
  state[4] = 0x67332667FFC00B31UL;
  state[5] = 0x8EB44A8768581511UL;
  state[6] = 0xDB0C2E0D64F98FA7UL;
  state[7] = 0x47B5481DBEFA4FA4UL;
  prsha512_process(state, message, len);
  for (int i = 0; i < HASH_WORDS; ++i) hash[i] = state[i];
}

static int prsha384_compare(__global const uchar* k_hash, uchar* password, const int length) {
  ulong hash[HASH_WORDS];
  prsha384_hash(password, (uint)length, hash);
  for (int i = 0; i < HASH_WORDS; ++i) {
    const ulong w = hash[i];
    const int off = i * 8;
    if (k_hash[off + 0] != (uchar)(w >> 56) ||
        k_hash[off + 1] != (uchar)(w >> 48) ||
        k_hash[off + 2] != (uchar)(w >> 40) ||
        k_hash[off + 3] != (uchar)(w >> 32) ||
        k_hash[off + 4] != (uchar)(w >> 24) ||
        k_hash[off + 5] != (uchar)(w >> 16) ||
        k_hash[off + 6] != (uchar)(w >> 8) ||
        k_hash[off + 7] != (uchar)(w)) {
      return 0;
    }
  }
  return 1;
}

__kernel void prsha384_kernel(__global uchar* result,
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
      if (prsha384_compare(k_hash, attempt, (int)(pass_len + 1u))) {
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
      if (prsha384_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
