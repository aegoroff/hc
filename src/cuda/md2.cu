/*!
 * \brief   The file contains MD2 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-11-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "md2.h"
#include "cuda_runtime.h"
#include "gpu_abi.h"

#define DIGESTSIZE 16

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];

/* MD2 PI-substitution table (RFC 1319). */
__constant__ static const uint8_t k_s_md2[256] = {
    41,  46,  67, 201, 162, 216, 124,   1,  61,  54,  84, 161,
    236, 240,   6,  19,  98, 167,   5,  243, 192, 199, 115, 140,
    152, 147,  43, 217, 188,  76, 130, 202,  30, 155,  87,  60,
    253, 212, 224,  22, 103,  66, 111,  24, 138,  23, 229,  18,
    190,  78, 196, 214, 218, 158, 222,  73, 160, 251, 245, 142,
    187,  47, 238, 122, 169, 104, 121, 145,  21, 178,   7,  63,
    148, 194,  16, 137,  11,  34,  95,  33, 128, 127,  93, 154,
    90, 144,  50,  39,  53,  62, 204, 231, 191, 247, 151,   3,
    255,  25,  48, 179,  72, 165, 181, 209, 215,  94, 146,  42,
    172,  86, 170, 198,  79, 184,  56, 210, 150, 164, 125, 182,
    118, 252, 107, 226, 156, 116,   4, 241,  69, 157, 112,  89,
    100, 113, 135,  32, 134,  91, 207, 101, 230,  45, 168,   2,
    27,  96,  37, 173, 174, 176, 185, 246,  28,  70,  97, 105,
    52,  64, 126,  15,  85,  71, 163,  35, 221,  81, 175,  58,
    195,  92, 249, 206, 186, 197, 234,  38,  44,  83,  13, 110,
    133,  40, 132,   9, 211, 223, 205, 244,  65, 129,  77,  82,
    106, 220,  55, 200, 108, 193, 171, 250,  36, 225, 123,   8,
    12, 189, 177,  74, 120, 136, 149, 139, 227,  99, 232, 109,
    233, 203, 213, 254,  59,   0,  29,  57, 242, 239, 183,  14,
    102,  88, 208, 228, 166, 119, 114, 248, 235, 117,  75,  10,
    49,  68,  80, 180, 143, 237,  31,  26, 219, 153, 141,  51,
    159,  17, 131, 20
};

__global__ static void prmd2_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                    const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__host__ static void prmd2_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len);

__device__ static BOOL prmd2_hash_eq(const uint8_t* password, const int length, const uint8_t* sbox);

__host__ void md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash,
                                 gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE, 0, cudaMemcpyHostToDevice));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ void prmd2_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    GPU_LAUNCH_INDEX_KERNEL(prmd2_kernel, ctx, dict_len);
}

void md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prmd2_run_kernel);
}

/* 18-round MD2 permutation over the 48-byte state. */
__device__ __forceinline__ void prmd2_permute(uint8_t X[48], const uint8_t* sbox) {
    int t = 0;
#pragma unroll
    for (int j = 0; j < 18; ++j) {
#pragma unroll
        for (int k = 0; k < 48; k += 8) {
            t = (X[k + 0] ^= sbox[t]);
            t = (X[k + 1] ^= sbox[t]);
            t = (X[k + 2] ^= sbox[t]);
            t = (X[k + 3] ^= sbox[t]);
            t = (X[k + 4] ^= sbox[t]);
            t = (X[k + 5] ^= sbox[t]);
            t = (X[k + 6] ^= sbox[t]);
            t = (X[k + 7] ^= sbox[t]);
        }
        t = (t + j) & 0xFF;
    }
}

/**
 * One-shot MD2 for passwords shorter than one block (GPU_ATTEMPT_SIZE-1 <= 15).
 * Skips checksum update on the final (checksum) block — unused for the digest.
 */
__device__ __forceinline__ BOOL prmd2_hash_eq(const uint8_t* password, const int length, const uint8_t* sbox) {
    uint8_t X[48];
    uint8_t C[16];
    uint8_t block[16];

    const uint8_t pad = (uint8_t)(DIGESTSIZE - length);
#pragma unroll
    for (int i = 0; i < DIGESTSIZE; ++i) {
        block[i] = (i < length) ? password[i] : pad;
        X[i] = 0;
        C[i] = 0;
    }

    /* Block 1: message || padding */
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        X[j + 16] = block[j];
        X[j + 32] = (uint8_t)(block[j] ^ X[j]);
    }
    prmd2_permute(X, sbox);

    int t = C[15];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        C[j] ^= sbox[block[j] ^ t];
        t = C[j];
    }

    /* Block 2: checksum (mix only — digest is X[0..15]) */
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        X[j + 16] = C[j];
        X[j + 32] = (uint8_t)(C[j] ^ X[j]);
    }
    prmd2_permute(X, sbox);

#pragma unroll
    for (int i = 0; i < DIGESTSIZE; ++i) {
        if (X[i] != k_hash[i]) {
            return FALSE;
        }
    }
    return TRUE;
}

__global__ void prmd2_kernel(unsigned char* result, const uint64_t start, const uint32_t count, const uint32_t pass_len,
                             const uint32_t dict_length, const uint32_t min_len) {
    __shared__ uint8_t s_sbox[256];
    for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
        s_sbox[i] = k_s_md2[i];
    }
    __syncthreads();

    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= count) {
        return;
    }

    uint64_t idx = start + ix;
    unsigned char attempt[GPU_ATTEMPT_SIZE];
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {
        attempt[pos] = k_dict[idx % dict_length];
        idx /= dict_length;
    }

    if (pass_len >= min_len && prmd2_hash_eq(attempt, (int)pass_len, s_sbox)) {
        memcpy(result, attempt, pass_len);
        return;
    }

    const uint32_t attempt_len = pass_len + 1u;
    /* One-shot path requires length < 16 (GPU max password is GPU_ATTEMPT_SIZE-1). */
    if (attempt_len < min_len || attempt_len >= DIGESTSIZE) {
        return;
    }

    for (uint32_t i = 0; i < dict_length; ++i) {
        attempt[pass_len] = k_dict[i];
        if (prmd2_hash_eq(attempt, (int)attempt_len, s_sbox)) {
            memcpy(result, attempt, attempt_len);
            return;
        }
    }
}
