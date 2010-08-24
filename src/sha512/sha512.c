/*-
 * Copyright (c) 2001-2003 Allan Saddi <allan@saddi.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY ALLAN SADDI AND HIS CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL ALLAN SADDI OR HIS CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: sha512.c 680 2003-07-25 21:58:07Z asaddi $
 */

/*
 * Define WORDS_BIGENDIAN if compiling on a big-endian architecture.
 *
 * Define SHA512_TEST to test the implementation using the NIST's
 * sample messages. The output should be:
 *
 *   ddaf35a193617aba cc417349ae204131 12e6fa4e89a97ea2 0a9eeee64b55d39a
 *   2192992a274fc1a8 36ba3c23a3feebbd 454d4423643ce80e 2a9ac94fa54ca49f
 *   8e959b75dae313da 8cf4f72814fc143f 8f7779c6eb9f7fa1 7299aeadb6889018
 *   501d289e4900f7e4 331b99dec4b5433a c7d329eeb6dd2654 5e96e55b874be909
 *   e718483d0ce76964 4e2e42c7bc15b463 8e1f98b13b204428 5632a803afa973eb
 *   de0ff244877ea60a 4cb0432ce577c31b eb009c5c2c49aa2e 4eadb217ad8cc09b
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#include <string.h>

#include "sha512.h"

#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#define Ch(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define Maj(x, y, z) (((x) & ((y) | (z))) | ((y) & (z)))
#define SIGMA0(x) (ROTR64((x), 28) ^ ROTR64((x), 34) ^ ROTR64((x), 39))
#define SIGMA1(x) (ROTR64((x), 14) ^ ROTR64((x), 18) ^ ROTR64((x), 41))
#define sigma0(x) (ROTR64((x), 1) ^ ROTR64((x), 8) ^ ((x) >> 7))
#define sigma1(x) (ROTR64((x), 19) ^ ROTR64((x), 61) ^ ((x) >> 6))

#define DO_ROUND() { \
  t1 = h + SIGMA1(e) + Ch(e, f, g) + *(Kp++) + *(W++); \
  t2 = SIGMA0(a) + Maj(a, b, c); \
  h = g; \
  g = f; \
  f = e; \
  e = d + t1; \
  d = c; \
  c = b; \
  b = a; \
  a = t1 + t2; \
}

static const unsigned long long K[80] = {
  0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
  0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
  0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
  0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
  0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
  0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
  0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
  0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
  0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
  0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
  0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
  0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
  0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
  0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
  0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
  0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
  0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
  0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
  0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
  0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
  0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
  0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
  0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
  0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
  0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
  0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
  0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
  0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
  0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
  0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
  0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
  0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
  0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
  0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
  0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
  0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
  0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
  0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
  0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
  0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

#ifndef RUNTIME_ENDIAN

#ifdef WORDS_BIGENDIAN

#define BYTESWAP(x) (x)
#define BYTESWAP64(x) (x)

#else /* WORDS_BIGENDIAN */

#define BYTESWAP(x) ((ROTR((x), 8) & 0xff00ff00L) | \
		     (ROTL((x), 8) & 0x00ff00ffL))
#define BYTESWAP64(x) _byteswap64(x)

static unsigned long long _byteswap64(unsigned long long x)
{
  unsigned long a = x >> 32;
  unsigned long b = (unsigned long) x;
  return ((unsigned long long) BYTESWAP(b) << 32) | (unsigned long long) BYTESWAP(a);
}

#endif /* WORDS_BIGENDIAN */

#else /* !RUNTIME_ENDIAN */

#define BYTESWAP(x) _byteswap(sc->littleEndian, x)
#define BYTESWAP64(x) _byteswap64(sc->littleEndian, x)

#define _BYTESWAP(x) ((ROTR((x), 8) & 0xff00ff00L) | \
		      (ROTL((x), 8) & 0x00ff00ffL))
#define _BYTESWAP64(x) __byteswap64(x)

static inline unsigned long long __byteswap64(unsigned long long x)
{
  unsigned long a = x >> 32;
  unsigned long b = (unsigned long) x;
  return ((unsigned long long) _BYTESWAP(b) << 32) | (unsigned long long) _BYTESWAP(a);
}

static inline unsigned long _byteswap(int littleEndian, unsigned long x)
{
  if (!littleEndian)
    return x;
  else
    return _BYTESWAP(x);
}

static inline unsigned long long _byteswap64(int littleEndian, unsigned long long x)
{
  if (!littleEndian)
    return x;
  else
    return _BYTESWAP64(x);
}

static inline void setEndian(int *littleEndianp)
{
  union {
    unsigned long w;
    unsigned char b[4];
  } endian;

  endian.w = 1L;
  *littleEndianp = endian.b[0] != 0;
}

#endif /* !RUNTIME_ENDIAN */

static const unsigned char padding[128] = {
  0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

void
SHA512Init (SHA512Context *sc)
{
#ifdef RUNTIME_ENDIAN
  setEndian (&sc->littleEndian);
#endif /* RUNTIME_ENDIAN */

  sc->totalLength[0] = 0LL;
  sc->totalLength[1] = 0LL;
  sc->hash[0] = 0x6a09e667f3bcc908ULL;
  sc->hash[1] = 0xbb67ae8584caa73bULL;
  sc->hash[2] = 0x3c6ef372fe94f82bULL;
  sc->hash[3] = 0xa54ff53a5f1d36f1ULL;
  sc->hash[4] = 0x510e527fade682d1ULL;
  sc->hash[5] = 0x9b05688c2b3e6c1fULL;
  sc->hash[6] = 0x1f83d9abfb41bd6bULL;
  sc->hash[7] = 0x5be0cd19137e2179ULL;
  sc->bufferLength = 0L;
}

static void
burnStack (int size)
{
  char buf[128];

  memset (buf, 0, sizeof (buf));
  size -= sizeof (buf);
  if (size > 0)
    burnStack (size);
}

static void
SHA512Guts (SHA512Context *sc, const unsigned long long *cbuf)
{
  unsigned long long buf[80];
  unsigned long long *W, *W2, *W7, *W15, *W16;
  unsigned long long a, b, c, d, e, f, g, h;
  unsigned long long t1, t2;
  const unsigned long long *Kp;
  int i;

  W = buf;

  for (i = 15; i >= 0; i--) {
    *(W++) = BYTESWAP64(*cbuf);
    cbuf++;
  }

  W16 = &buf[0];
  W15 = &buf[1];
  W7 = &buf[9];
  W2 = &buf[14];

  for (i = 63; i >= 0; i--) {
    *(W++) = sigma1(*W2) + *(W7++) + sigma0(*W15) + *(W16++);
    W2++;
    W15++;
  }

  a = sc->hash[0];
  b = sc->hash[1];
  c = sc->hash[2];
  d = sc->hash[3];
  e = sc->hash[4];
  f = sc->hash[5];
  g = sc->hash[6];
  h = sc->hash[7];

  Kp = K;
  W = buf;

  for (i = 79; i >= 0; i--)
    DO_ROUND();

  sc->hash[0] += a;
  sc->hash[1] += b;
  sc->hash[2] += c;
  sc->hash[3] += d;
  sc->hash[4] += e;
  sc->hash[5] += f;
  sc->hash[6] += g;
  sc->hash[7] += h;
}

void
SHA512Update (SHA512Context *sc, const void *vdata, unsigned long len)
{
  const unsigned char *data = vdata;
  unsigned long bufferBytesLeft;
  unsigned long bytesToCopy;
  unsigned long long carryCheck;
  int needBurn = 0;

#ifdef SHA512_FAST_COPY
  if (sc->bufferLength) {
    bufferBytesLeft = 128L - sc->bufferLength;

    bytesToCopy = bufferBytesLeft;
    if (bytesToCopy > len)
      bytesToCopy = len;

    memcpy (&sc->buffer.bytes[sc->bufferLength], data, bytesToCopy);

    carryCheck = sc->totalLength[1];
    sc->totalLength[1] += bytesToCopy * 8L;
    if (sc->totalLength[1] < carryCheck)
      sc->totalLength[0]++;

    sc->bufferLength += bytesToCopy;
    data += bytesToCopy;
    len -= bytesToCopy;

    if (sc->bufferLength == 128L) {
      SHA512Guts (sc, sc->buffer.words);
      needBurn = 1;
      sc->bufferLength = 0L;
    }
  }

  while (len > 127) {
    carryCheck = sc->totalLength[1];
    sc->totalLength[1] += 1024L;
    if (sc->totalLength[1] < carryCheck)
      sc->totalLength[0]++;

    SHA512Guts (sc, data);
    needBurn = 1;

    data += 128L;
    len -= 128L;
  }

  if (len) {
    memcpy (&sc->buffer.bytes[sc->bufferLength], data, len);

    carryCheck = sc->totalLength[1];
    sc->totalLength[1] += len * 8L;
    if (sc->totalLength[1] < carryCheck)
      sc->totalLength[0]++;

    sc->bufferLength += len;
  }
#else /* SHA512_FAST_COPY */
  while (len) {
    bufferBytesLeft = 128L - sc->bufferLength;

    bytesToCopy = bufferBytesLeft;
    if (bytesToCopy > len)
      bytesToCopy = len;

    memcpy (&sc->buffer.bytes[sc->bufferLength], data, bytesToCopy);

    carryCheck = sc->totalLength[1];
    sc->totalLength[1] += bytesToCopy * 8L;
    if (sc->totalLength[1] < carryCheck)
      sc->totalLength[0]++;

    sc->bufferLength += bytesToCopy;
    data += bytesToCopy;
    len -= bytesToCopy;

    if (sc->bufferLength == 128L) {
      SHA512Guts (sc, sc->buffer.words);
      needBurn = 1;
      sc->bufferLength = 0L;
    }
  }
#endif /* SHA512_FAST_COPY */

  if (needBurn)
    burnStack (sizeof (unsigned long long[90]) + sizeof (unsigned long long *[6]) + sizeof (int));
}

void SHA512Final (unsigned char* hash, SHA512Context *sc)
{
  unsigned long bytesToPad;
  unsigned long long lengthPad[2];
  int i;

  bytesToPad = 240L - sc->bufferLength;
  if (bytesToPad > 128L)
    bytesToPad -= 128L;

  lengthPad[0] = BYTESWAP64(sc->totalLength[0]);
  lengthPad[1] = BYTESWAP64(sc->totalLength[1]);

  SHA512Update (sc, padding, bytesToPad);
  SHA512Update (sc, lengthPad, 16L);

  if (hash) {
    for (i = 0; i < SHA512_HASH_WORDS; i++) {
#ifdef SHA384_FAST_COPY
      *((unsigned long long *) hash) = BYTESWAP64(sc->hash[i]);
#else /* SHA384_FAST_COPY */
      hash[0] = (unsigned char) (sc->hash[i] >> 56);
      hash[1] = (unsigned char) (sc->hash[i] >> 48);
      hash[2] = (unsigned char) (sc->hash[i] >> 40);
      hash[3] = (unsigned char) (sc->hash[i] >> 32);
      hash[4] = (unsigned char) (sc->hash[i] >> 24);
      hash[5] = (unsigned char) (sc->hash[i] >> 16);
      hash[6] = (unsigned char) (sc->hash[i] >> 8);
      hash[7] = (unsigned char) sc->hash[i];
#endif /* SHA384_FAST_COPY */
      hash += 8;
    }
  }
}
