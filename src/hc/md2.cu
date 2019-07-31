/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains MD2 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-11-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include <stdint.h>
#include "md2.h"
#include "cuda_runtime.h"
#include "gpu.h"

#define DIGESTSIZE 16

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];

typedef struct {
    unsigned char buf[16];    /* first field, for alignment */
    union {
        unsigned char X[48];
        uint32_t W[12];
    } u;
    unsigned char C[16];
    unsigned L, count;
} md2_context;

/*
* The MD2 magic table.
*/
__constant__ static const unsigned char S[256] = {
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
__device__ static BOOL prmd2_compare(unsigned char* password, const int length);
__host__ static void prmd2_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len);
__device__ static void prmd2_round(md2_context *mc);


__host__ void md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, unsigned char** variants, size_t variants_len) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE));
    CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(variants), variants_len * sizeof(unsigned char), cudaHostAllocDefault));
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

    md2_context mc = {0};
    unsigned char* data = password;
    size_t len = length;

    memset(&mc.u.X, 0, 16);
    memset(&mc.C, 0, 16);
    mc.L = 0;
    mc.count = 0;

    unsigned current = mc.count;

    if(current > 0) {

        unsigned clen = 16U - current;

        if(clen > len)
            clen = len;
        memcpy(mc.u.X + 16 + current, data, clen);
        data = data + clen;
        current += clen;
        len -= clen;
        if(current == 16) {
            prmd2_round(&mc);
            current = 0;
        }
    }
    while(len >= 16) {
        memcpy(mc.u.X + 16, data, 16);
        prmd2_round(&mc);
        data = data + 16;
        len -= 16;
    }
    memcpy(mc.u.X + 16, data, len);
    mc.count = len;


    const unsigned u = mc.count;
    const unsigned v = 16 - u;
    memset(mc.u.X + 16 + u, v, v);
    prmd2_round(&mc);
    memcpy(mc.u.X + 16, mc.C, 16);
    prmd2_round(&mc);
    memcpy(hash, mc.u.X, DIGESTSIZE);

    BOOL result = TRUE;

#pragma unroll (DIGESTSIZE)
    for(int i = 0; i < DIGESTSIZE && result; ++i) {
        result &= hash[i] == k_hash[i];
    }

    return result;
}

__device__ static void prmd2_round(md2_context* mc) {
    int j;

    unsigned L = mc->L;
#pragma unroll (DIGESTSIZE)
    for(j = 0; j < DIGESTSIZE; ++j) {
        /*
        * WARNING: RFC 1319 pseudo-code in chapter 3.2 is
        * incorrect. This implementation matches the reference
        * implementation and the reference test vectors. The
        * RFC 1319 flaw is documented in the official errata:
        * http://www.rfc-editor.org/errata.html
        */
        L = mc->C[j] = mc->C[j] ^ S[mc->u.X[j + 16] ^ L];
    }
    mc->L = L;

    mc->u.W[8] = mc->u.W[4] ^ mc->u.W[0];
    mc->u.W[9] = mc->u.W[5] ^ mc->u.W[1];
    mc->u.W[10] = mc->u.W[6] ^ mc->u.W[2];
    mc->u.W[11] = mc->u.W[7] ^ mc->u.W[3];

    unsigned t = 0;
#pragma unroll (4)
    for(j = 0; j < 18; ++j) {

        /*
        * We unroll 8 steps. 8 steps are good; this has been
        * empirically determined to be the right unroll length
        * (6 steps yield slightly worse performance; 16 steps
        * are no better than 8).
        */
#pragma unroll (8)
        for(int k = 0; k < 48; ++k) {
            t = (mc->u.X[k] ^= S[t]);
        }
        t = (t + j) & 0xFF;
    }
}
