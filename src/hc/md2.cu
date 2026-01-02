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
#include "gpu.h"

#define DIGESTSIZE 16

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];

typedef struct md2_ctx_t {
    uint8_t data[16];
    uint8_t state[48];
    uint8_t checksum[16];
    int len;
} md2_ctx_t;

/*
* The MD2 magic table.
*/
__constant__ static const uint8_t k_s_md2[256] = {
    41,  46,  67, 201, 162, 216, 124,   1,  61,  54,  84, 161,
    236, 240,   6,  19,  98, 167,   5, 243, 192, 199, 115, 140,
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

__global__ static void prmd2_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length);
__host__ static void prmd2_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len);

__device__ static BOOL prmd2_compare(unsigned char* password, const int length);

__device__ static void prmd2_init(md2_ctx_t* ctx);
__device__ static void prmd2_update(md2_ctx_t* ctx, const uint8_t data[], size_t len);
__device__ static void prmd2_final(md2_ctx_t* ctx, uint8_t hash[]);   // size of hash must be MD2_BLOCK_SIZE


__host__ void md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE, 0, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ void prmd2_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prmd2_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>>(dev_result, dev_variants, static_cast<uint32_t>(dict_len));
}

void md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prmd2_run_kernel);
}

KERNEL_WITHOUT_ALLOCATION(prmd2_kernel, prmd2_compare)

__device__ __forceinline__ BOOL prmd2_compare(unsigned char* password, const int length) {
    uint8_t hash[DIGESTSIZE];

    md2_ctx_t ctx = { 0 };
    prmd2_init(&ctx);
    prmd2_update(&ctx, password, length);
    prmd2_final(&ctx, hash);

    BOOL result = TRUE;

    for(int i = 0; i < DIGESTSIZE && result; ++i) {
        result &= hash[i] == k_hash[i];
    }

    return result;
}

__device__ __forceinline__ void prmd2_transform(md2_ctx_t* ctx, uint8_t data[]) {
    int j, k, t;

#pragma unroll (16)
    for (j = 0; j < 16; ++j) {
        ctx->state[j + 16] = data[j];
        ctx->state[j + 32] = (ctx->state[j + 16] ^ ctx->state[j]);
    }

    t = 0;
#pragma unroll (2)
    for (j = 0; j < 18; ++j) {
/*
* We unroll 8 steps. 8 steps are good; this has been
* empirically determined to be the right unroll length
* (6 steps yield slightly worse performance; 16 steps
* are no better than 8).
*/
#pragma unroll (8)
        for (k = 0; k < 48; ++k) {
            ctx->state[k] ^= k_s_md2[t];
            t = ctx->state[k];
        }
        t = (t + j) & 0xFF;
    }

    t = ctx->checksum[15];
#pragma unroll (16)
    for (j = 0; j < 16; ++j) {
        ctx->checksum[j] ^= k_s_md2[data[j] ^ t];
        t = ctx->checksum[j];
    }
}

__device__ __forceinline__ void prmd2_init(md2_ctx_t* ctx) {
    int i;
#pragma unroll (48)
    for (i = 0; i < 48; ++i)
        ctx->state[i] = 0;
#pragma unroll (16)
    for (i = 0; i < 16; ++i)
        ctx->checksum[i] = 0;
    ctx->len = 0;
}

__device__ __forceinline__ void prmd2_update(md2_ctx_t* ctx, const uint8_t data[], size_t len) {
    size_t i;

    for (i = 0; i < len; ++i) {
        ctx->data[ctx->len] = data[i];
        ctx->len++;
        if (ctx->len == DIGESTSIZE) {
            prmd2_transform(ctx, ctx->data);
            ctx->len = 0;
        }
    }
}

__device__ __forceinline__ void prmd2_final(md2_ctx_t* ctx, uint8_t hash[]) {
    int to_pad;

    to_pad = DIGESTSIZE - ctx->len;

#pragma unroll (16)
    while (ctx->len < DIGESTSIZE)
        ctx->data[ctx->len++] = to_pad;

    prmd2_transform(ctx, ctx->data);
    prmd2_transform(ctx, ctx->checksum);

    memcpy(hash, ctx->state, DIGESTSIZE);
}