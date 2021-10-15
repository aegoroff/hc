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
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#include <stdint.h>
#include "md4.h"
#include "cuda_runtime.h"
#include "gpu.h"

#define DIGESTSIZE 16
__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];
__device__ static BOOL g_found;

#define F(B, C, D)     ((((C) ^ (D)) & (B)) ^ (D))
#define G(B, C, D)     (((D) & (C)) | (((D) | (C)) & (B)))
#define H(B, C, D)     ((B) ^ (C) ^ (D))

#define ROTL   SPH_ROTL32

#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#define SPH_ROTL32(x, n)   SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))

#define SPH_BLEN     64U
#define SPH_WLEN      4U
#define SPH_MAXPAD   (SPH_BLEN - (SPH_WLEN << 1))
#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))

typedef struct {
    uint8_t buf[64];    /* first field, for alignment */
    uint32_t val[4];
    uint64_t count;
} gpu_md4_context;

#define MD4_ROUND_BODY(in, r)   do { \
		uint32_t A, B, C, D; \
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
__device__ static void prmd4_round(const unsigned char* data, uint32_t r[4]);
__device__ static uint32_t prmd4_dec32le_aligned(const void* src);
__device__ static void prmd4_short(void* cc, const void* data, size_t len);
__device__ static void prmd4_addbits_and_close(void* cc, unsigned ub, unsigned n);
__device__ static void prmd4_enc64le_aligned(void* dst, uint64_t val);
__device__ static void prmd4_enc32le(void* dst, uint32_t val);

void md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prmd4_run_kernel);
}

void md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));

    const BOOL f = FALSE;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_found, &f, sizeof(BOOL)));
}

__global__ void prmd4_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length, BOOL use_wide_pass) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;
    wchar_t wide_attempt[GPU_ATTEMPT_SIZE];
    void* attemptBuf = NULL;
    size_t attemptLen = 0;
    size_t curentLen = 0;
    
    size_t len = 0;

    // strlen
    while (attempt[len]) {
        ++len;
    }

    for (int i = 0; i < dict_length; ++i) {
        attempt[len] = k_dict[i];

        curentLen = len + 1;
        // Optimization: it was calculated before
        // Calculate only on first iteration
        if (curentLen == 4) {
            if (g_found) {
                return;
            }

            if (use_wide_pass) {
                for (int i = 0; i < curentLen; ++i) {
                    wide_attempt[i] = attempt[i];
                }

                attemptBuf = wide_attempt;
                attemptLen = curentLen * sizeof(wchar_t);
            } else {
                attemptBuf = attempt;
                attemptLen = curentLen;
            }

            if (prmd4_compare(attemptBuf, attemptLen)) {
                memcpy(result, attempt, curentLen);
                g_found = TRUE;
                return;
            }
        }

        curentLen = len + 2;

        for (int j = 0; j < dict_length; ++j) {
            attempt[len + 1] = k_dict[j];

            if (g_found) {
                return;
            }

            if (use_wide_pass) {
                for (int i = 0; i < curentLen; ++i) {
                    wide_attempt[i] = attempt[i];
                }

                attemptBuf = wide_attempt;
                attemptLen = curentLen * sizeof(wchar_t);
            } else {
                attemptBuf = attempt;
                attemptLen = curentLen;
            }

            if (prmd4_compare(attemptBuf, attemptLen)) {
                memcpy(result, attempt, curentLen);
                g_found = TRUE;
                return;
            }
        }
    }
}


__host__ void prmd4_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prmd4_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>> (dev_result, dev_variants, static_cast<uint32_t>(dict_len), ctx->use_wide_pass_);
}

__device__ __forceinline__ BOOL prmd4_compare(void* password, const int length) {
    // load into register
    const uint32_t ar = (unsigned)k_hash[0] | (unsigned)k_hash[1] << 8 | (unsigned)k_hash[2] << 16 | (unsigned)k_hash[3] << 24;
    const uint32_t br = (unsigned)k_hash[4] | (unsigned)k_hash[5] << 8 | (unsigned)k_hash[6] << 16 | (unsigned)k_hash[7] << 24;
    const uint32_t cr = (unsigned)k_hash[8] | (unsigned)k_hash[9] << 8 | (unsigned)k_hash[10] << 16 | (unsigned)k_hash[11] << 24;
    const uint32_t dr = (unsigned)k_hash[12] | (unsigned)k_hash[13] << 8 | (unsigned)k_hash[14] << 16 | (unsigned)k_hash[15] << 24;

    gpu_md4_context ctx = { 0 };

    ctx.val[0] = 0x67452301;
    ctx.val[1] = 0xEFCDAB89;
    ctx.val[2] = 0x98BADCFE;
    ctx.val[3] = 0x10325476;

    prmd4_calculate(&ctx, password, length);

    prmd4_addbits_and_close(&ctx, 0, 0);

    const uint32_t a = ctx.val[0];
    const uint32_t b = ctx.val[1];
    const uint32_t c = ctx.val[2];
    const uint32_t d = ctx.val[3];

    return a == ar && b == br && c == cr && d == dr;
}

__device__ __forceinline__ void prmd4_calculate(void* cc, const void* data, size_t len) {
    if (len < (2 * SPH_BLEN)) {
        prmd4_short(cc, data, len);
        return;
    }

    gpu_md4_context* sc = (gpu_md4_context*)cc;
    unsigned current = (unsigned)sc->count & (SPH_BLEN - 1U);

    if (current > 0) {
        unsigned t = SPH_BLEN - current;
        prmd4_short(cc, data, t);
        data = (const unsigned char*)data + t;
        len -= t;
    }

    size_t orig_len = len;
    while (len >= SPH_BLEN) {
        prmd4_round((unsigned char*)data, sc->val);
        len -= SPH_BLEN;
        data = (const unsigned char*)data + SPH_BLEN;
    }

    if (len > 0) {
        memcpy(sc->buf, data, len);
    }

    sc->count += (uint64_t)orig_len;
}

__device__ __forceinline__ void prmd4_enc64le_aligned(void* dst, uint64_t val) {
    *(uint64_t*)dst = val;
}

__device__ __forceinline__ void prmd4_enc32le(void* dst, uint32_t val) {
    *(uint32_t*)dst = val;
}

/*
 * One round of MD4. The data must be aligned for 32-bit access.
 */
__device__ __forceinline__ void prmd4_round(const unsigned char* data, uint32_t r[4]) {
   #define X(idx)    prmd4_dec32le_aligned(data + 4 * (idx))

    MD4_ROUND_BODY(X, r);

    #undef X
}

__device__ __forceinline__ uint32_t prmd4_dec32le_aligned(const void* src) {
    return *(const uint32_t*)src;
}

__device__ __forceinline__ void prmd4_short(void* cc, const void* data, size_t len) {
    gpu_md4_context* sc = (gpu_md4_context*)cc;
    unsigned current = (unsigned)sc->count & (SPH_BLEN - 1U);

    while (len > 0) {
        unsigned clen = SPH_BLEN - current;
        if (clen > len) {
            clen = len;
        }
            
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

__device__ __forceinline__ void prmd4_addbits_and_close(void* cc, unsigned ub, unsigned n) {
    gpu_md4_context* sc = (gpu_md4_context*)cc;
    unsigned current = (unsigned)sc->count & (SPH_BLEN - 1U);

    {
        unsigned z = 0x80 >> n;
        sc->buf[current++] = ((ub & -z) | z) & 0xFF;
    }

    if (current > SPH_MAXPAD) {
        memset(sc->buf + current, 0, SPH_BLEN - current);
        prmd4_round(sc->buf, sc->val);
        memset(sc->buf, 0, SPH_MAXPAD);
    } else {
        memset(sc->buf + current, 0, SPH_MAXPAD - current);
    }

    prmd4_enc64le_aligned(sc->buf + SPH_MAXPAD, SPH_T64(sc->count << 3) + (uint64_t)n);

    prmd4_round(sc->buf, sc->val);
}