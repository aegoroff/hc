/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains MD4 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-09-28
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include <stdint.h>
#include "md4.h"
#include "cuda_runtime.h"
#include "gpu.h"

#define DIGESTSIZE 16
__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];

#define F(B, C, D)     ((((C) ^ (D)) & (B)) ^ (D))
#define G(B, C, D)     (((D) & (C)) | (((D) | (C)) & (B)))
#define H(B, C, D)     ((B) ^ (C) ^ (D))

typedef unsigned int sph_u32;
typedef unsigned long long sph_u64;

#define ROTL   SPH_ROTL32

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#define SPH_ROTL32(x, n)   SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))

#define SPH_BLEN     64U
#define SPH_WLEN      4U
#define SPH_MAXPAD   (SPH_BLEN - (SPH_WLEN << 1))
#define SPH_C64(x)    ((sph_u64)(x ## ULL))
#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))

__constant__ static const sph_u32 IV[4] = {
    SPH_C32(0x67452301), SPH_C32(0xEFCDAB89),
    SPH_C32(0x98BADCFE), SPH_C32(0x10325476)
};

typedef struct {
    unsigned char buf[64];    /* first field, for alignment */
    sph_u32 val[4];
    sph_u64 count;
} gpu_md4_context;

#define MD4_ROUND_BODY(in, r)   do { \
		sph_u32 A, B, C, D; \
 \
		A = (r)[0]; \
		B = (r)[1]; \
		C = (r)[2]; \
		D = (r)[3]; \
 \
  A = ROTL(SPH_T32(A + F(B, C, D) + in( 0)), 3); \
  D = ROTL(SPH_T32(D + F(A, B, C) + in( 1)), 7); \
  C = ROTL(SPH_T32(C + F(D, A, B) + in( 2)), 11); \
  B = ROTL(SPH_T32(B + F(C, D, A) + in( 3)), 19); \
  A = ROTL(SPH_T32(A + F(B, C, D) + in( 4)), 3); \
  D = ROTL(SPH_T32(D + F(A, B, C) + in( 5)), 7); \
  C = ROTL(SPH_T32(C + F(D, A, B) + in( 6)), 11); \
  B = ROTL(SPH_T32(B + F(C, D, A) + in( 7)), 19); \
  A = ROTL(SPH_T32(A + F(B, C, D) + in( 8)), 3); \
  D = ROTL(SPH_T32(D + F(A, B, C) + in( 9)), 7); \
  C = ROTL(SPH_T32(C + F(D, A, B) + in(10)), 11); \
  B = ROTL(SPH_T32(B + F(C, D, A) + in(11)), 19); \
  A = ROTL(SPH_T32(A + F(B, C, D) + in(12)), 3); \
  D = ROTL(SPH_T32(D + F(A, B, C) + in(13)), 7); \
  C = ROTL(SPH_T32(C + F(D, A, B) + in(14)), 11); \
  B = ROTL(SPH_T32(B + F(C, D, A) + in(15)), 19); \
 \
  A = ROTL(SPH_T32(A + G(B, C, D) + in( 0) + SPH_C32(0x5A827999)), 3); \
  D = ROTL(SPH_T32(D + G(A, B, C) + in( 4) + SPH_C32(0x5A827999)), 5); \
  C = ROTL(SPH_T32(C + G(D, A, B) + in( 8) + SPH_C32(0x5A827999)), 9); \
  B = ROTL(SPH_T32(B + G(C, D, A) + in(12) + SPH_C32(0x5A827999)), 13); \
  A = ROTL(SPH_T32(A + G(B, C, D) + in( 1) + SPH_C32(0x5A827999)), 3); \
  D = ROTL(SPH_T32(D + G(A, B, C) + in( 5) + SPH_C32(0x5A827999)), 5); \
  C = ROTL(SPH_T32(C + G(D, A, B) + in( 9) + SPH_C32(0x5A827999)), 9); \
  B = ROTL(SPH_T32(B + G(C, D, A) + in(13) + SPH_C32(0x5A827999)), 13); \
  A = ROTL(SPH_T32(A + G(B, C, D) + in( 2) + SPH_C32(0x5A827999)), 3); \
  D = ROTL(SPH_T32(D + G(A, B, C) + in( 6) + SPH_C32(0x5A827999)), 5); \
  C = ROTL(SPH_T32(C + G(D, A, B) + in(10) + SPH_C32(0x5A827999)), 9); \
  B = ROTL(SPH_T32(B + G(C, D, A) + in(14) + SPH_C32(0x5A827999)), 13); \
  A = ROTL(SPH_T32(A + G(B, C, D) + in( 3) + SPH_C32(0x5A827999)), 3); \
  D = ROTL(SPH_T32(D + G(A, B, C) + in( 7) + SPH_C32(0x5A827999)), 5); \
  C = ROTL(SPH_T32(C + G(D, A, B) + in(11) + SPH_C32(0x5A827999)), 9); \
  B = ROTL(SPH_T32(B + G(C, D, A) + in(15) + SPH_C32(0x5A827999)), 13); \
 \
  A = ROTL(SPH_T32(A + H(B, C, D) + in( 0) + SPH_C32(0x6ED9EBA1)), 3); \
  D = ROTL(SPH_T32(D + H(A, B, C) + in( 8) + SPH_C32(0x6ED9EBA1)), 9); \
  C = ROTL(SPH_T32(C + H(D, A, B) + in( 4) + SPH_C32(0x6ED9EBA1)), 11); \
  B = ROTL(SPH_T32(B + H(C, D, A) + in(12) + SPH_C32(0x6ED9EBA1)), 15); \
  A = ROTL(SPH_T32(A + H(B, C, D) + in( 2) + SPH_C32(0x6ED9EBA1)), 3); \
  D = ROTL(SPH_T32(D + H(A, B, C) + in(10) + SPH_C32(0x6ED9EBA1)), 9); \
  C = ROTL(SPH_T32(C + H(D, A, B) + in( 6) + SPH_C32(0x6ED9EBA1)), 11); \
  B = ROTL(SPH_T32(B + H(C, D, A) + in(14) + SPH_C32(0x6ED9EBA1)), 15); \
  A = ROTL(SPH_T32(A + H(B, C, D) + in( 1) + SPH_C32(0x6ED9EBA1)), 3); \
  D = ROTL(SPH_T32(D + H(A, B, C) + in( 9) + SPH_C32(0x6ED9EBA1)), 9); \
  C = ROTL(SPH_T32(C + H(D, A, B) + in( 5) + SPH_C32(0x6ED9EBA1)), 11); \
  B = ROTL(SPH_T32(B + H(C, D, A) + in(13) + SPH_C32(0x6ED9EBA1)), 15); \
  A = ROTL(SPH_T32(A + H(B, C, D) + in( 3) + SPH_C32(0x6ED9EBA1)), 3); \
  D = ROTL(SPH_T32(D + H(A, B, C) + in(11) + SPH_C32(0x6ED9EBA1)), 9); \
  C = ROTL(SPH_T32(C + H(D, A, B) + in( 7) + SPH_C32(0x6ED9EBA1)), 11); \
  B = ROTL(SPH_T32(B + H(C, D, A) + in(15) + SPH_C32(0x6ED9EBA1)), 15); \
 \
		(r)[0] = SPH_T32(r[0] + A); \
		(r)[1] = SPH_T32(r[1] + B); \
		(r)[2] = SPH_T32(r[2] + C); \
		(r)[3] = SPH_T32(r[3] + D); \
	} while (0)

__global__ static void prmd4_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length, BOOL use_wide_pass);
__host__ static void prmd4_run_kernel(gpu_tread_ctx_t * ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len);
__device__ static BOOL prmd4_compare(void* password, const int length);
__device__ static void prmd4_calculate(void* cc, const void* data, size_t len);
__device__ static void prmd4_round(const unsigned char* data, sph_u32 r[4]);
__device__ static sph_u32 prmd4_dec32le_aligned(const void* src);
__device__ static void prmd4_comp(const sph_u32 msg[16], sph_u32 val[4]);
__device__ static void prmd4_short(void* cc, const void* data, size_t len);
__device__ static void prmd4_addbits_and_close(void* cc, unsigned ub, unsigned n, void* dst);
__device__ static void prmd4_enc64le_aligned(void* dst, sph_u64 val);
__device__ static void prmd4_enc32le(void* dst, sph_u32 val);

void md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prmd4_run_kernel);
}

void md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE));
    CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(&ctx->variants_), ctx->variants_size_ * sizeof(unsigned char), cudaHostAllocDefault));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__global__ void prmd4_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length, BOOL use_wide_pass) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;
    wchar_t wide_attempt[GPU_ATTEMPT_SIZE];
    
    size_t len = 0;
    while (attempt[len]) {
        ++len;
    }

    if (use_wide_pass) {
        for (int i = 0; i < len; ++i) {
            wide_attempt[i] = attempt[i];
        }

        if (prmd4_compare(wide_attempt, len * sizeof(wchar_t))) {
            memcpy(result, attempt, len);
            return;
        }
    }
    else {
        if (prmd4_compare(attempt, len)) {
            memcpy(result, attempt, len);
            return;
        }
    }

    const size_t attempt_len = len + 1;
    for (int i = 0; i < dict_length; ++i)
    {
        attempt[len] = k_dict[i];

        if (use_wide_pass) {
            for (int i = 0; i < attempt_len; ++i) {
                wide_attempt[i] = attempt[i];
            }

            if (prmd4_compare(wide_attempt, attempt_len * sizeof(wchar_t))) {
                memcpy(result, attempt, attempt_len);
                return;
            }
        }
        else {
            if (prmd4_compare(attempt, attempt_len)) {
                memcpy(result, attempt, attempt_len);
                return;
            }
        }
    }
}


__host__ void prmd4_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prmd4_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>> (dev_result, dev_variants, static_cast<uint32_t>(dict_len), ctx->use_wide_pass_);
}

__device__ __forceinline__ BOOL prmd4_compare(void* password, const int length) {
    gpu_md4_context ctx = { 0 };
    uint8_t hash[DIGESTSIZE];
    memcpy(ctx.val, IV, sizeof IV);

    prmd4_calculate(&ctx, password, length);

    prmd4_addbits_and_close(&ctx, 0, 0, hash);

    BOOL result = TRUE;

#pragma unroll (DIGESTSIZE)
    for (int i = 0; i < DIGESTSIZE && result; ++i) {
        result &= hash[i] == k_hash[i];
    }

    return result;
}

__device__ __forceinline__ void prmd4_calculate(void* cc, const void* data, size_t len) {
    gpu_md4_context* sc;
    unsigned current;
    size_t orig_len;

    if (len < (2 * SPH_BLEN)) {
        prmd4_short(cc, data, len);
        return;
    }
    sc = (gpu_md4_context*)cc;

    current = (unsigned)sc->count & (SPH_BLEN - 1U);

    if (current > 0) {
        unsigned t;

        t = SPH_BLEN - current;
        prmd4_short(cc, data, t);
        data = (const unsigned char*)data + t;
        len -= t;
    }

    orig_len = len;
    while (len >= SPH_BLEN) {
        prmd4_round((unsigned char*)data, sc->val);
        len -= SPH_BLEN;
        data = (const unsigned char*)data + SPH_BLEN;
    }
    if (len > 0)
        memcpy(sc->buf, data, len);
    sc->count += (sph_u64)orig_len;
}

__device__ __forceinline__ void prmd4_comp(const sph_u32 msg[16], sph_u32 val[4]) {
#define X(i)   msg[i]
    MD4_ROUND_BODY(X, val);
#undef X
}

__device__ __forceinline__ void prmd4_enc64le_aligned(void* dst, sph_u64 val) {
    *(sph_u64*)dst = val;
}

__device__ __forceinline__ void prmd4_enc32le(void* dst, sph_u32 val) {
    *(sph_u32*)dst = val;
}

/*
 * One round of MD4. The data must be aligned for 32-bit access.
 */
__device__ __forceinline__ void prmd4_round(const unsigned char* data, sph_u32 r[4]) {
   #define X(idx)    prmd4_dec32le_aligned(data + 4 * (idx))

    MD4_ROUND_BODY(X, r);

    #undef X
}

__device__ __forceinline__ sph_u32 prmd4_dec32le_aligned(const void* src) {
    return *(const sph_u32*)src;
}

__device__ __forceinline__ void prmd4_short(void* cc, const void* data, size_t len) {
    gpu_md4_context* sc;
    unsigned current;

    sc = (gpu_md4_context*)cc;
    current = (unsigned)sc->count & (SPH_BLEN - 1U);

    while (len > 0) {
        unsigned clen;

        clen = SPH_BLEN - current;
        if (clen > len)
            clen = len;
        memcpy(sc->buf + current, data, clen);
        data = (const unsigned char*)data + clen;
        current += clen;
        len -= clen;
        if (current == SPH_BLEN) {
            prmd4_round(sc->buf, sc->val);
            current = 0;
        }

        sc->count += clen;
    }
}

__device__ __forceinline__ void prmd4_addbits_and_close(void* cc, unsigned ub, unsigned n, void* dst) {
    gpu_md4_context* sc;
    unsigned current;

    sc = (gpu_md4_context*)cc;
    current = (unsigned)sc->count & (SPH_BLEN - 1U);

    {
        unsigned z;

        z = 0x80 >> n;
        sc->buf[current++] = ((ub & -z) | z) & 0xFF;
    }

    if (current > SPH_MAXPAD) {
        memset(sc->buf + current, 0, SPH_BLEN - current);
        prmd4_round(sc->buf, sc->val);
        memset(sc->buf, 0, SPH_MAXPAD);
    }
    else {
        memset(sc->buf + current, 0, SPH_MAXPAD - current);
    }


    prmd4_enc64le_aligned(sc->buf + SPH_MAXPAD,
        SPH_T64(sc->count << 3) + (sph_u64)n);

    prmd4_round(sc->buf, sc->val);

#pragma unroll (4)
    for (unsigned u = 0; u < 4; u++) {
        prmd4_enc32le((unsigned char*)dst + 4 * u, sc->val[u]);
    }
}