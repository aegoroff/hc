#define GPU_ATTEMPT_SIZE 16
#define DIGESTSIZE 48
#define BLOCK_LEN 128
#define STATE_LEN 8
#define HASH_WORDS 6
#define LENGTH_SIZE 16

static void prsha512_compress(ulong state[], const uchar block[]) {
#define ROTR64(x, n)  (((0UL + (x)) << (64 - (n))) | ((x) >> (n)))
  /* Rolling w[16] instead of schedule[80]: cuts ~640 B/thread private mem
   * that spilled to slow local memory on Intel Arc. */
  ulong w[16];
  w[0] = (ulong)block[0 * 8 + 0] << 56
            | (ulong)block[0 * 8 + 1] << 48
            | (ulong)block[0 * 8 + 2] << 40
            | (ulong)block[0 * 8 + 3] << 32
            | (ulong)block[0 * 8 + 4] << 24
            | (ulong)block[0 * 8 + 5] << 16
            | (ulong)block[0 * 8 + 6] <<  8
            | (ulong)block[0 * 8 + 7] <<  0;
  w[1] = (ulong)block[1 * 8 + 0] << 56
            | (ulong)block[1 * 8 + 1] << 48
            | (ulong)block[1 * 8 + 2] << 40
            | (ulong)block[1 * 8 + 3] << 32
            | (ulong)block[1 * 8 + 4] << 24
            | (ulong)block[1 * 8 + 5] << 16
            | (ulong)block[1 * 8 + 6] <<  8
            | (ulong)block[1 * 8 + 7] <<  0;
  w[2] = (ulong)block[2 * 8 + 0] << 56
            | (ulong)block[2 * 8 + 1] << 48
            | (ulong)block[2 * 8 + 2] << 40
            | (ulong)block[2 * 8 + 3] << 32
            | (ulong)block[2 * 8 + 4] << 24
            | (ulong)block[2 * 8 + 5] << 16
            | (ulong)block[2 * 8 + 6] <<  8
            | (ulong)block[2 * 8 + 7] <<  0;
  w[3] = (ulong)block[3 * 8 + 0] << 56
            | (ulong)block[3 * 8 + 1] << 48
            | (ulong)block[3 * 8 + 2] << 40
            | (ulong)block[3 * 8 + 3] << 32
            | (ulong)block[3 * 8 + 4] << 24
            | (ulong)block[3 * 8 + 5] << 16
            | (ulong)block[3 * 8 + 6] <<  8
            | (ulong)block[3 * 8 + 7] <<  0;
  w[4] = (ulong)block[4 * 8 + 0] << 56
            | (ulong)block[4 * 8 + 1] << 48
            | (ulong)block[4 * 8 + 2] << 40
            | (ulong)block[4 * 8 + 3] << 32
            | (ulong)block[4 * 8 + 4] << 24
            | (ulong)block[4 * 8 + 5] << 16
            | (ulong)block[4 * 8 + 6] <<  8
            | (ulong)block[4 * 8 + 7] <<  0;
  w[5] = (ulong)block[5 * 8 + 0] << 56
            | (ulong)block[5 * 8 + 1] << 48
            | (ulong)block[5 * 8 + 2] << 40
            | (ulong)block[5 * 8 + 3] << 32
            | (ulong)block[5 * 8 + 4] << 24
            | (ulong)block[5 * 8 + 5] << 16
            | (ulong)block[5 * 8 + 6] <<  8
            | (ulong)block[5 * 8 + 7] <<  0;
  w[6] = (ulong)block[6 * 8 + 0] << 56
            | (ulong)block[6 * 8 + 1] << 48
            | (ulong)block[6 * 8 + 2] << 40
            | (ulong)block[6 * 8 + 3] << 32
            | (ulong)block[6 * 8 + 4] << 24
            | (ulong)block[6 * 8 + 5] << 16
            | (ulong)block[6 * 8 + 6] <<  8
            | (ulong)block[6 * 8 + 7] <<  0;
  w[7] = (ulong)block[7 * 8 + 0] << 56
            | (ulong)block[7 * 8 + 1] << 48
            | (ulong)block[7 * 8 + 2] << 40
            | (ulong)block[7 * 8 + 3] << 32
            | (ulong)block[7 * 8 + 4] << 24
            | (ulong)block[7 * 8 + 5] << 16
            | (ulong)block[7 * 8 + 6] <<  8
            | (ulong)block[7 * 8 + 7] <<  0;
  w[8] = (ulong)block[8 * 8 + 0] << 56
            | (ulong)block[8 * 8 + 1] << 48
            | (ulong)block[8 * 8 + 2] << 40
            | (ulong)block[8 * 8 + 3] << 32
            | (ulong)block[8 * 8 + 4] << 24
            | (ulong)block[8 * 8 + 5] << 16
            | (ulong)block[8 * 8 + 6] <<  8
            | (ulong)block[8 * 8 + 7] <<  0;
  w[9] = (ulong)block[9 * 8 + 0] << 56
            | (ulong)block[9 * 8 + 1] << 48
            | (ulong)block[9 * 8 + 2] << 40
            | (ulong)block[9 * 8 + 3] << 32
            | (ulong)block[9 * 8 + 4] << 24
            | (ulong)block[9 * 8 + 5] << 16
            | (ulong)block[9 * 8 + 6] <<  8
            | (ulong)block[9 * 8 + 7] <<  0;
  w[10] = (ulong)block[10 * 8 + 0] << 56
            | (ulong)block[10 * 8 + 1] << 48
            | (ulong)block[10 * 8 + 2] << 40
            | (ulong)block[10 * 8 + 3] << 32
            | (ulong)block[10 * 8 + 4] << 24
            | (ulong)block[10 * 8 + 5] << 16
            | (ulong)block[10 * 8 + 6] <<  8
            | (ulong)block[10 * 8 + 7] <<  0;
  w[11] = (ulong)block[11 * 8 + 0] << 56
            | (ulong)block[11 * 8 + 1] << 48
            | (ulong)block[11 * 8 + 2] << 40
            | (ulong)block[11 * 8 + 3] << 32
            | (ulong)block[11 * 8 + 4] << 24
            | (ulong)block[11 * 8 + 5] << 16
            | (ulong)block[11 * 8 + 6] <<  8
            | (ulong)block[11 * 8 + 7] <<  0;
  w[12] = (ulong)block[12 * 8 + 0] << 56
            | (ulong)block[12 * 8 + 1] << 48
            | (ulong)block[12 * 8 + 2] << 40
            | (ulong)block[12 * 8 + 3] << 32
            | (ulong)block[12 * 8 + 4] << 24
            | (ulong)block[12 * 8 + 5] << 16
            | (ulong)block[12 * 8 + 6] <<  8
            | (ulong)block[12 * 8 + 7] <<  0;
  w[13] = (ulong)block[13 * 8 + 0] << 56
            | (ulong)block[13 * 8 + 1] << 48
            | (ulong)block[13 * 8 + 2] << 40
            | (ulong)block[13 * 8 + 3] << 32
            | (ulong)block[13 * 8 + 4] << 24
            | (ulong)block[13 * 8 + 5] << 16
            | (ulong)block[13 * 8 + 6] <<  8
            | (ulong)block[13 * 8 + 7] <<  0;
  w[14] = (ulong)block[14 * 8 + 0] << 56
            | (ulong)block[14 * 8 + 1] << 48
            | (ulong)block[14 * 8 + 2] << 40
            | (ulong)block[14 * 8 + 3] << 32
            | (ulong)block[14 * 8 + 4] << 24
            | (ulong)block[14 * 8 + 5] << 16
            | (ulong)block[14 * 8 + 6] <<  8
            | (ulong)block[14 * 8 + 7] <<  0;
  w[15] = (ulong)block[15 * 8 + 0] << 56
            | (ulong)block[15 * 8 + 1] << 48
            | (ulong)block[15 * 8 + 2] << 40
            | (ulong)block[15 * 8 + 3] << 32
            | (ulong)block[15 * 8 + 4] << 24
            | (ulong)block[15 * 8 + 5] << 16
            | (ulong)block[15 * 8 + 6] <<  8
            | (ulong)block[15 * 8 + 7] <<  0;
  ulong a = state[0];
  ulong b = state[1];
  ulong c = state[2];
  ulong d = state[3];
  ulong e = state[4];
  ulong f = state[5];
  ulong g = state[6];
  ulong h = state[7];
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x428A2F98D728AE22UL + w[0];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x7137449123EF65CDUL + w[1];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xB5C0FBCFEC4D3B2FUL + w[2];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xE9B5DBA58189DBBCUL + w[3];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x3956C25BF348B538UL + w[4];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x59F111F1B605D019UL + w[5];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x923F82A4AF194F9BUL + w[6];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xAB1C5ED5DA6D8118UL + w[7];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xD807AA98A3030242UL + w[8];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x12835B0145706FBEUL + w[9];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x243185BE4EE4B28CUL + w[10];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x550C7DC3D5FFB4E2UL + w[11];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x72BE5D74F27B896FUL + w[12];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x80DEB1FE3B1696B1UL + w[13];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x9BDC06A725C71235UL + w[14];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xC19BF174CF692694UL + w[15];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[0] = 0UL + w[0] + w[9]
            + (ROTR64(w[1], 1) ^ ROTR64(w[1], 8) ^ (w[1] >> 7))
            + (ROTR64(w[14], 19) ^ ROTR64(w[14], 61) ^ (w[14] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xE49B69C19EF14AD2UL + w[0];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[1] = 0UL + w[1] + w[10]
            + (ROTR64(w[2], 1) ^ ROTR64(w[2], 8) ^ (w[2] >> 7))
            + (ROTR64(w[15], 19) ^ ROTR64(w[15], 61) ^ (w[15] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xEFBE4786384F25E3UL + w[1];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[2] = 0UL + w[2] + w[11]
            + (ROTR64(w[3], 1) ^ ROTR64(w[3], 8) ^ (w[3] >> 7))
            + (ROTR64(w[0], 19) ^ ROTR64(w[0], 61) ^ (w[0] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x0FC19DC68B8CD5B5UL + w[2];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[3] = 0UL + w[3] + w[12]
            + (ROTR64(w[4], 1) ^ ROTR64(w[4], 8) ^ (w[4] >> 7))
            + (ROTR64(w[1], 19) ^ ROTR64(w[1], 61) ^ (w[1] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x240CA1CC77AC9C65UL + w[3];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[4] = 0UL + w[4] + w[13]
            + (ROTR64(w[5], 1) ^ ROTR64(w[5], 8) ^ (w[5] >> 7))
            + (ROTR64(w[2], 19) ^ ROTR64(w[2], 61) ^ (w[2] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x2DE92C6F592B0275UL + w[4];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[5] = 0UL + w[5] + w[14]
            + (ROTR64(w[6], 1) ^ ROTR64(w[6], 8) ^ (w[6] >> 7))
            + (ROTR64(w[3], 19) ^ ROTR64(w[3], 61) ^ (w[3] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x4A7484AA6EA6E483UL + w[5];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[6] = 0UL + w[6] + w[15]
            + (ROTR64(w[7], 1) ^ ROTR64(w[7], 8) ^ (w[7] >> 7))
            + (ROTR64(w[4], 19) ^ ROTR64(w[4], 61) ^ (w[4] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5CB0A9DCBD41FBD4UL + w[6];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[7] = 0UL + w[7] + w[0]
            + (ROTR64(w[8], 1) ^ ROTR64(w[8], 8) ^ (w[8] >> 7))
            + (ROTR64(w[5], 19) ^ ROTR64(w[5], 61) ^ (w[5] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x76F988DA831153B5UL + w[7];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[8] = 0UL + w[8] + w[1]
            + (ROTR64(w[9], 1) ^ ROTR64(w[9], 8) ^ (w[9] >> 7))
            + (ROTR64(w[6], 19) ^ ROTR64(w[6], 61) ^ (w[6] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x983E5152EE66DFABUL + w[8];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[9] = 0UL + w[9] + w[2]
            + (ROTR64(w[10], 1) ^ ROTR64(w[10], 8) ^ (w[10] >> 7))
            + (ROTR64(w[7], 19) ^ ROTR64(w[7], 61) ^ (w[7] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xA831C66D2DB43210UL + w[9];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[10] = 0UL + w[10] + w[3]
            + (ROTR64(w[11], 1) ^ ROTR64(w[11], 8) ^ (w[11] >> 7))
            + (ROTR64(w[8], 19) ^ ROTR64(w[8], 61) ^ (w[8] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xB00327C898FB213FUL + w[10];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[11] = 0UL + w[11] + w[4]
            + (ROTR64(w[12], 1) ^ ROTR64(w[12], 8) ^ (w[12] >> 7))
            + (ROTR64(w[9], 19) ^ ROTR64(w[9], 61) ^ (w[9] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xBF597FC7BEEF0EE4UL + w[11];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[12] = 0UL + w[12] + w[5]
            + (ROTR64(w[13], 1) ^ ROTR64(w[13], 8) ^ (w[13] >> 7))
            + (ROTR64(w[10], 19) ^ ROTR64(w[10], 61) ^ (w[10] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0xC6E00BF33DA88FC2UL + w[12];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[13] = 0UL + w[13] + w[6]
            + (ROTR64(w[14], 1) ^ ROTR64(w[14], 8) ^ (w[14] >> 7))
            + (ROTR64(w[11], 19) ^ ROTR64(w[11], 61) ^ (w[11] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xD5A79147930AA725UL + w[13];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[14] = 0UL + w[14] + w[7]
            + (ROTR64(w[15], 1) ^ ROTR64(w[15], 8) ^ (w[15] >> 7))
            + (ROTR64(w[12], 19) ^ ROTR64(w[12], 61) ^ (w[12] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x06CA6351E003826FUL + w[14];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[15] = 0UL + w[15] + w[8]
            + (ROTR64(w[0], 1) ^ ROTR64(w[0], 8) ^ (w[0] >> 7))
            + (ROTR64(w[13], 19) ^ ROTR64(w[13], 61) ^ (w[13] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x142929670A0E6E70UL + w[15];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[0] = 0UL + w[0] + w[9]
            + (ROTR64(w[1], 1) ^ ROTR64(w[1], 8) ^ (w[1] >> 7))
            + (ROTR64(w[14], 19) ^ ROTR64(w[14], 61) ^ (w[14] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x27B70A8546D22FFCUL + w[0];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[1] = 0UL + w[1] + w[10]
            + (ROTR64(w[2], 1) ^ ROTR64(w[2], 8) ^ (w[2] >> 7))
            + (ROTR64(w[15], 19) ^ ROTR64(w[15], 61) ^ (w[15] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x2E1B21385C26C926UL + w[1];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[2] = 0UL + w[2] + w[11]
            + (ROTR64(w[3], 1) ^ ROTR64(w[3], 8) ^ (w[3] >> 7))
            + (ROTR64(w[0], 19) ^ ROTR64(w[0], 61) ^ (w[0] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x4D2C6DFC5AC42AEDUL + w[2];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[3] = 0UL + w[3] + w[12]
            + (ROTR64(w[4], 1) ^ ROTR64(w[4], 8) ^ (w[4] >> 7))
            + (ROTR64(w[1], 19) ^ ROTR64(w[1], 61) ^ (w[1] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x53380D139D95B3DFUL + w[3];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[4] = 0UL + w[4] + w[13]
            + (ROTR64(w[5], 1) ^ ROTR64(w[5], 8) ^ (w[5] >> 7))
            + (ROTR64(w[2], 19) ^ ROTR64(w[2], 61) ^ (w[2] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x650A73548BAF63DEUL + w[4];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[5] = 0UL + w[5] + w[14]
            + (ROTR64(w[6], 1) ^ ROTR64(w[6], 8) ^ (w[6] >> 7))
            + (ROTR64(w[3], 19) ^ ROTR64(w[3], 61) ^ (w[3] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x766A0ABB3C77B2A8UL + w[5];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[6] = 0UL + w[6] + w[15]
            + (ROTR64(w[7], 1) ^ ROTR64(w[7], 8) ^ (w[7] >> 7))
            + (ROTR64(w[4], 19) ^ ROTR64(w[4], 61) ^ (w[4] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x81C2C92E47EDAEE6UL + w[6];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[7] = 0UL + w[7] + w[0]
            + (ROTR64(w[8], 1) ^ ROTR64(w[8], 8) ^ (w[8] >> 7))
            + (ROTR64(w[5], 19) ^ ROTR64(w[5], 61) ^ (w[5] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x92722C851482353BUL + w[7];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[8] = 0UL + w[8] + w[1]
            + (ROTR64(w[9], 1) ^ ROTR64(w[9], 8) ^ (w[9] >> 7))
            + (ROTR64(w[6], 19) ^ ROTR64(w[6], 61) ^ (w[6] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xA2BFE8A14CF10364UL + w[8];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[9] = 0UL + w[9] + w[2]
            + (ROTR64(w[10], 1) ^ ROTR64(w[10], 8) ^ (w[10] >> 7))
            + (ROTR64(w[7], 19) ^ ROTR64(w[7], 61) ^ (w[7] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xA81A664BBC423001UL + w[9];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[10] = 0UL + w[10] + w[3]
            + (ROTR64(w[11], 1) ^ ROTR64(w[11], 8) ^ (w[11] >> 7))
            + (ROTR64(w[8], 19) ^ ROTR64(w[8], 61) ^ (w[8] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xC24B8B70D0F89791UL + w[10];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[11] = 0UL + w[11] + w[4]
            + (ROTR64(w[12], 1) ^ ROTR64(w[12], 8) ^ (w[12] >> 7))
            + (ROTR64(w[9], 19) ^ ROTR64(w[9], 61) ^ (w[9] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xC76C51A30654BE30UL + w[11];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[12] = 0UL + w[12] + w[5]
            + (ROTR64(w[13], 1) ^ ROTR64(w[13], 8) ^ (w[13] >> 7))
            + (ROTR64(w[10], 19) ^ ROTR64(w[10], 61) ^ (w[10] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0xD192E819D6EF5218UL + w[12];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[13] = 0UL + w[13] + w[6]
            + (ROTR64(w[14], 1) ^ ROTR64(w[14], 8) ^ (w[14] >> 7))
            + (ROTR64(w[11], 19) ^ ROTR64(w[11], 61) ^ (w[11] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xD69906245565A910UL + w[13];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[14] = 0UL + w[14] + w[7]
            + (ROTR64(w[15], 1) ^ ROTR64(w[15], 8) ^ (w[15] >> 7))
            + (ROTR64(w[12], 19) ^ ROTR64(w[12], 61) ^ (w[12] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0xF40E35855771202AUL + w[14];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[15] = 0UL + w[15] + w[8]
            + (ROTR64(w[0], 1) ^ ROTR64(w[0], 8) ^ (w[0] >> 7))
            + (ROTR64(w[13], 19) ^ ROTR64(w[13], 61) ^ (w[13] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x106AA07032BBD1B8UL + w[15];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[0] = 0UL + w[0] + w[9]
            + (ROTR64(w[1], 1) ^ ROTR64(w[1], 8) ^ (w[1] >> 7))
            + (ROTR64(w[14], 19) ^ ROTR64(w[14], 61) ^ (w[14] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x19A4C116B8D2D0C8UL + w[0];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[1] = 0UL + w[1] + w[10]
            + (ROTR64(w[2], 1) ^ ROTR64(w[2], 8) ^ (w[2] >> 7))
            + (ROTR64(w[15], 19) ^ ROTR64(w[15], 61) ^ (w[15] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x1E376C085141AB53UL + w[1];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[2] = 0UL + w[2] + w[11]
            + (ROTR64(w[3], 1) ^ ROTR64(w[3], 8) ^ (w[3] >> 7))
            + (ROTR64(w[0], 19) ^ ROTR64(w[0], 61) ^ (w[0] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x2748774CDF8EEB99UL + w[2];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[3] = 0UL + w[3] + w[12]
            + (ROTR64(w[4], 1) ^ ROTR64(w[4], 8) ^ (w[4] >> 7))
            + (ROTR64(w[1], 19) ^ ROTR64(w[1], 61) ^ (w[1] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x34B0BCB5E19B48A8UL + w[3];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[4] = 0UL + w[4] + w[13]
            + (ROTR64(w[5], 1) ^ ROTR64(w[5], 8) ^ (w[5] >> 7))
            + (ROTR64(w[2], 19) ^ ROTR64(w[2], 61) ^ (w[2] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x391C0CB3C5C95A63UL + w[4];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[5] = 0UL + w[5] + w[14]
            + (ROTR64(w[6], 1) ^ ROTR64(w[6], 8) ^ (w[6] >> 7))
            + (ROTR64(w[3], 19) ^ ROTR64(w[3], 61) ^ (w[3] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x4ED8AA4AE3418ACBUL + w[5];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[6] = 0UL + w[6] + w[15]
            + (ROTR64(w[7], 1) ^ ROTR64(w[7], 8) ^ (w[7] >> 7))
            + (ROTR64(w[4], 19) ^ ROTR64(w[4], 61) ^ (w[4] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5B9CCA4F7763E373UL + w[6];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[7] = 0UL + w[7] + w[0]
            + (ROTR64(w[8], 1) ^ ROTR64(w[8], 8) ^ (w[8] >> 7))
            + (ROTR64(w[5], 19) ^ ROTR64(w[5], 61) ^ (w[5] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x682E6FF3D6B2B8A3UL + w[7];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[8] = 0UL + w[8] + w[1]
            + (ROTR64(w[9], 1) ^ ROTR64(w[9], 8) ^ (w[9] >> 7))
            + (ROTR64(w[6], 19) ^ ROTR64(w[6], 61) ^ (w[6] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x748F82EE5DEFB2FCUL + w[8];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[9] = 0UL + w[9] + w[2]
            + (ROTR64(w[10], 1) ^ ROTR64(w[10], 8) ^ (w[10] >> 7))
            + (ROTR64(w[7], 19) ^ ROTR64(w[7], 61) ^ (w[7] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x78A5636F43172F60UL + w[9];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[10] = 0UL + w[10] + w[3]
            + (ROTR64(w[11], 1) ^ ROTR64(w[11], 8) ^ (w[11] >> 7))
            + (ROTR64(w[8], 19) ^ ROTR64(w[8], 61) ^ (w[8] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x84C87814A1F0AB72UL + w[10];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[11] = 0UL + w[11] + w[4]
            + (ROTR64(w[12], 1) ^ ROTR64(w[12], 8) ^ (w[12] >> 7))
            + (ROTR64(w[9], 19) ^ ROTR64(w[9], 61) ^ (w[9] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x8CC702081A6439ECUL + w[11];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[12] = 0UL + w[12] + w[5]
            + (ROTR64(w[13], 1) ^ ROTR64(w[13], 8) ^ (w[13] >> 7))
            + (ROTR64(w[10], 19) ^ ROTR64(w[10], 61) ^ (w[10] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x90BEFFFA23631E28UL + w[12];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[13] = 0UL + w[13] + w[6]
            + (ROTR64(w[14], 1) ^ ROTR64(w[14], 8) ^ (w[14] >> 7))
            + (ROTR64(w[11], 19) ^ ROTR64(w[11], 61) ^ (w[11] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0xA4506CEBDE82BDE9UL + w[13];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[14] = 0UL + w[14] + w[7]
            + (ROTR64(w[15], 1) ^ ROTR64(w[15], 8) ^ (w[15] >> 7))
            + (ROTR64(w[12], 19) ^ ROTR64(w[12], 61) ^ (w[12] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0xBEF9A3F7B2C67915UL + w[14];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[15] = 0UL + w[15] + w[8]
            + (ROTR64(w[0], 1) ^ ROTR64(w[0], 8) ^ (w[0] >> 7))
            + (ROTR64(w[13], 19) ^ ROTR64(w[13], 61) ^ (w[13] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0xC67178F2E372532BUL + w[15];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[0] = 0UL + w[0] + w[9]
            + (ROTR64(w[1], 1) ^ ROTR64(w[1], 8) ^ (w[1] >> 7))
            + (ROTR64(w[14], 19) ^ ROTR64(w[14], 61) ^ (w[14] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0xCA273ECEEA26619CUL + w[0];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[1] = 0UL + w[1] + w[10]
            + (ROTR64(w[2], 1) ^ ROTR64(w[2], 8) ^ (w[2] >> 7))
            + (ROTR64(w[15], 19) ^ ROTR64(w[15], 61) ^ (w[15] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0xD186B8C721C0C207UL + w[1];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[2] = 0UL + w[2] + w[11]
            + (ROTR64(w[3], 1) ^ ROTR64(w[3], 8) ^ (w[3] >> 7))
            + (ROTR64(w[0], 19) ^ ROTR64(w[0], 61) ^ (w[0] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0xEADA7DD6CDE0EB1EUL + w[2];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[3] = 0UL + w[3] + w[12]
            + (ROTR64(w[4], 1) ^ ROTR64(w[4], 8) ^ (w[4] >> 7))
            + (ROTR64(w[1], 19) ^ ROTR64(w[1], 61) ^ (w[1] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0xF57D4F7FEE6ED178UL + w[3];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[4] = 0UL + w[4] + w[13]
            + (ROTR64(w[5], 1) ^ ROTR64(w[5], 8) ^ (w[5] >> 7))
            + (ROTR64(w[2], 19) ^ ROTR64(w[2], 61) ^ (w[2] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x06F067AA72176FBAUL + w[4];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[5] = 0UL + w[5] + w[14]
            + (ROTR64(w[6], 1) ^ ROTR64(w[6], 8) ^ (w[6] >> 7))
            + (ROTR64(w[3], 19) ^ ROTR64(w[3], 61) ^ (w[3] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x0A637DC5A2C898A6UL + w[5];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[6] = 0UL + w[6] + w[15]
            + (ROTR64(w[7], 1) ^ ROTR64(w[7], 8) ^ (w[7] >> 7))
            + (ROTR64(w[4], 19) ^ ROTR64(w[4], 61) ^ (w[4] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x113F9804BEF90DAEUL + w[6];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[7] = 0UL + w[7] + w[0]
            + (ROTR64(w[8], 1) ^ ROTR64(w[8], 8) ^ (w[8] >> 7))
            + (ROTR64(w[5], 19) ^ ROTR64(w[5], 61) ^ (w[5] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x1B710B35131C471BUL + w[7];
  e = 0UL + e + a;
  a = 0UL + a + (ROTR64(b, 28) ^ ROTR64(b, 34) ^ ROTR64(b, 39)) + ((b & (c | d)) | (c & d));
  w[8] = 0UL + w[8] + w[1]
            + (ROTR64(w[9], 1) ^ ROTR64(w[9], 8) ^ (w[9] >> 7))
            + (ROTR64(w[6], 19) ^ ROTR64(w[6], 61) ^ (w[6] >> 6));
  h = 0UL + h + (ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41)) + (g ^ (e & (f ^ g))) + 0x28DB77F523047D84UL + w[8];
  d = 0UL + d + h;
  h = 0UL + h + (ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39)) + ((a & (b | c)) | (b & c));
  w[9] = 0UL + w[9] + w[2]
            + (ROTR64(w[10], 1) ^ ROTR64(w[10], 8) ^ (w[10] >> 7))
            + (ROTR64(w[7], 19) ^ ROTR64(w[7], 61) ^ (w[7] >> 6));
  g = 0UL + g + (ROTR64(d, 14) ^ ROTR64(d, 18) ^ ROTR64(d, 41)) + (f ^ (d & (e ^ f))) + 0x32CAAB7B40C72493UL + w[9];
  c = 0UL + c + g;
  g = 0UL + g + (ROTR64(h, 28) ^ ROTR64(h, 34) ^ ROTR64(h, 39)) + ((h & (a | b)) | (a & b));
  w[10] = 0UL + w[10] + w[3]
            + (ROTR64(w[11], 1) ^ ROTR64(w[11], 8) ^ (w[11] >> 7))
            + (ROTR64(w[8], 19) ^ ROTR64(w[8], 61) ^ (w[8] >> 6));
  f = 0UL + f + (ROTR64(c, 14) ^ ROTR64(c, 18) ^ ROTR64(c, 41)) + (e ^ (c & (d ^ e))) + 0x3C9EBE0A15C9BEBCUL + w[10];
  b = 0UL + b + f;
  f = 0UL + f + (ROTR64(g, 28) ^ ROTR64(g, 34) ^ ROTR64(g, 39)) + ((g & (h | a)) | (h & a));
  w[11] = 0UL + w[11] + w[4]
            + (ROTR64(w[12], 1) ^ ROTR64(w[12], 8) ^ (w[12] >> 7))
            + (ROTR64(w[9], 19) ^ ROTR64(w[9], 61) ^ (w[9] >> 6));
  e = 0UL + e + (ROTR64(b, 14) ^ ROTR64(b, 18) ^ ROTR64(b, 41)) + (d ^ (b & (c ^ d))) + 0x431D67C49C100D4CUL + w[11];
  a = 0UL + a + e;
  e = 0UL + e + (ROTR64(f, 28) ^ ROTR64(f, 34) ^ ROTR64(f, 39)) + ((f & (g | h)) | (g & h));
  w[12] = 0UL + w[12] + w[5]
            + (ROTR64(w[13], 1) ^ ROTR64(w[13], 8) ^ (w[13] >> 7))
            + (ROTR64(w[10], 19) ^ ROTR64(w[10], 61) ^ (w[10] >> 6));
  d = 0UL + d + (ROTR64(a, 14) ^ ROTR64(a, 18) ^ ROTR64(a, 41)) + (c ^ (a & (b ^ c))) + 0x4CC5D4BECB3E42B6UL + w[12];
  h = 0UL + h + d;
  d = 0UL + d + (ROTR64(e, 28) ^ ROTR64(e, 34) ^ ROTR64(e, 39)) + ((e & (f | g)) | (f & g));
  w[13] = 0UL + w[13] + w[6]
            + (ROTR64(w[14], 1) ^ ROTR64(w[14], 8) ^ (w[14] >> 7))
            + (ROTR64(w[11], 19) ^ ROTR64(w[11], 61) ^ (w[11] >> 6));
  c = 0UL + c + (ROTR64(h, 14) ^ ROTR64(h, 18) ^ ROTR64(h, 41)) + (b ^ (h & (a ^ b))) + 0x597F299CFC657E2AUL + w[13];
  g = 0UL + g + c;
  c = 0UL + c + (ROTR64(d, 28) ^ ROTR64(d, 34) ^ ROTR64(d, 39)) + ((d & (e | f)) | (e & f));
  w[14] = 0UL + w[14] + w[7]
            + (ROTR64(w[15], 1) ^ ROTR64(w[15], 8) ^ (w[15] >> 7))
            + (ROTR64(w[12], 19) ^ ROTR64(w[12], 61) ^ (w[12] >> 6));
  b = 0UL + b + (ROTR64(g, 14) ^ ROTR64(g, 18) ^ ROTR64(g, 41)) + (a ^ (g & (h ^ a))) + 0x5FCB6FAB3AD6FAECUL + w[14];
  f = 0UL + f + b;
  b = 0UL + b + (ROTR64(c, 28) ^ ROTR64(c, 34) ^ ROTR64(c, 39)) + ((c & (d | e)) | (d & e));
  w[15] = 0UL + w[15] + w[8]
            + (ROTR64(w[0], 1) ^ ROTR64(w[0], 8) ^ (w[0] >> 7))
            + (ROTR64(w[13], 19) ^ ROTR64(w[13], 61) ^ (w[13] >> 6));
  a = 0UL + a + (ROTR64(f, 14) ^ ROTR64(f, 18) ^ ROTR64(f, 41)) + (h ^ (f & (g ^ h))) + 0x6C44198C4A475817UL + w[15];
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
  uint off;
  for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)
    prsha512_compress(state, &message[off]);
  uchar block[BLOCK_LEN];
  for (int i = 0; i < BLOCK_LEN; ++i) block[i] = 0;
  uint rem = len - off;
  for (uint i = 0; i < rem; ++i) block[i] = message[off + i];
  block[rem] = 0x80;
  rem++;
  if (BLOCK_LEN - rem < LENGTH_SIZE) {
    prsha512_compress(state, block);
    for (int i = 0; i < BLOCK_LEN; ++i) block[i] = 0;
  }
  ulong bitlen = (ulong)len;
  block[BLOCK_LEN - 1] = (uchar)((bitlen & 0x1FU) << 3);
  bitlen >>= 5;
  for (int i = 1; i < LENGTH_SIZE; i++, bitlen >>= 8)
    block[BLOCK_LEN - 1 - i] = (uchar)(bitlen & 0xFFU);
  prsha512_compress(state, block);
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
