/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains SHA-1 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <stdint.h>
#include "sha1.h"
#include "cuda_runtime.h"

#define CUDA_SAFE_CALL(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        fprintf(stderr, "Error:%s \"%s\" at %s:%d\n", cudaGetErrorName(err), cudaGetErrorString(err), \
        __FILE__, __LINE__); return; \
    }} while (0);

 /* f1 to f4 */

__device__ inline uint32_t f1(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) | (~x & z)); }
__device__ inline uint32_t f2(uint32_t x, uint32_t y, uint32_t z) { return (x ^ y ^ z); }
__device__ inline uint32_t f3(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) | (x & z) | (y & z)); }
__device__ inline uint32_t f4(uint32_t x, uint32_t y, uint32_t z) { return (x ^ y ^ z); }

/* SHA init values */

__constant__ uint32_t I1 = 0x67452301L;
__constant__ uint32_t I2 = 0xEFCDAB89L;
__constant__ uint32_t I3 = 0x98BADCFEL;
__constant__ uint32_t I4 = 0x10325476L;
__constant__ uint32_t I5 = 0xC3D2E1F0L;

/* SHA constants */

__constant__ uint32_t C1 = 0x5A827999L;
__constant__ uint32_t C2 = 0x6Ed9EBA1L;
__constant__ uint32_t C3 = 0x8F1BBCDCL;
__constant__ uint32_t C4 = 0xCA62C1D6L;

/* 32-bit rotate */

__device__ inline uint32_t ROT(uint32_t x, int n) { return ((x << n) | (x >> (32 - n))); }

/* main function */

#define CALC(n,i) temp =  ROT ( A , 5 ) + f##n( B , C, D ) +  W[i] + E + C##n  ; E = D; D = C; C = ROT ( B , 30 ); B = A; A = temp

__device__ void prsha1_mem_init(uint32_t*, const unsigned char*, const int);
__device__ BOOL prsha1_compare(unsigned char* password, const int length);

__constant__ unsigned char k_dict[CHAR_MAX];
__constant__ unsigned char k_hash[DIGESTSIZE];

__global__ void sha1_kernel(unsigned char* result, unsigned char* variants, const uint32_t attempt_length, const uint32_t dict_length);

__host__ void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, char** variants, size_t variants_len) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(char)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE));
    CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(variants), variants_len * sizeof(char), cudaHostAllocDefault));
}

__host__ void sha1_on_gpu_cleanup(gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaFreeHost(ctx->variants_));
}

__host__ void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, size_t dict_len, unsigned char* variants, const size_t variants_size) {
    unsigned char* dev_result = nullptr;
    unsigned char* dev_variants = nullptr;

    size_t result_size_in_bytes = (MAX_DEFAULT + 1) * sizeof(unsigned char); // include trailing zero

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result), result_size_in_bytes));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_variants), variants_size * sizeof(unsigned char)));
    
    CUDA_SAFE_CALL(cudaMemset(dev_result, 0x0, result_size_in_bytes));

    CUDA_SAFE_CALL(cudaMemcpy(dev_variants, variants, variants_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    sha1_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>>(dev_result, dev_variants, ctx->pass_length_, static_cast<uint32_t>(dict_len));

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaMemcpy(ctx->result_, dev_result, result_size_in_bytes, cudaMemcpyDeviceToHost));

    if(ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }

    CUDA_SAFE_CALL(cudaFree(dev_result));
    CUDA_SAFE_CALL(cudaFree(dev_variants));
}


__global__ void sha1_kernel(unsigned char* result, unsigned char* variants, const uint32_t attempt_length, const uint32_t dict_length) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char* attempt = variants + ix * MAX_DEFAULT;

    size_t len = 0;

    while (len <= attempt_length && attempt[len]) {
        ++len;
    }

    if (prsha1_compare(attempt, len)) {
        memcpy(result, attempt, len);
        return;
    }

    const size_t attempt_len = len + 1;
    for (int i = 0; i < dict_length; i++)
    {
        attempt[len] = k_dict[i];

        if (prsha1_compare(attempt, attempt_len)) {
            memcpy(result, attempt, attempt_len);
            return;
        }
    }
}

__device__ BOOL prsha1_compare(unsigned char* password, const int length) {
    // load into register
    const uint32_t h0 = (unsigned)k_hash[3] | (unsigned)k_hash[2] << 8 | (unsigned)k_hash[1] << 16 | (unsigned)k_hash[0] << 24;
    const uint32_t h1 = (unsigned)k_hash[7] | (unsigned)k_hash[6] << 8 | (unsigned)k_hash[5] << 16 | (unsigned)k_hash[4] << 24;
    const uint32_t h2 = (unsigned)k_hash[11] | (unsigned)k_hash[10] << 8 | (unsigned)k_hash[9] << 16 | (unsigned)k_hash[8] << 24;
    const uint32_t h3 = (unsigned)k_hash[15] | (unsigned)k_hash[14] << 8 | (unsigned)k_hash[13] << 16 | (unsigned)k_hash[12] << 24;
    const uint32_t h4 = (unsigned)k_hash[19] | (unsigned)k_hash[18] << 8 | (unsigned)k_hash[17] << 16 | (unsigned)k_hash[16] << 24;

    // Init words for SHA
    uint32_t W[80], temp;

    // Calculate sha for given input.
    // DO THE SHA ------------------------------------------------------

    prsha1_mem_init(W, password, length);

    
    for (int i = 16; i < 80; i++) {
        W[i] = ROT((W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16]), 1);
    }

    uint32_t A = I1;
    uint32_t B = I2;
    uint32_t C = I3;
    uint32_t D = I4;
    uint32_t E = I5;

    CALC(1, 0);  CALC(1, 1);  CALC(1, 2);  CALC(1, 3);  CALC(1, 4);
    CALC(1, 5);  CALC(1, 6);  CALC(1, 7);  CALC(1, 8);  CALC(1, 9);
    CALC(1, 10); CALC(1, 11); CALC(1, 12); CALC(1, 13); CALC(1, 14);
    CALC(1, 15); CALC(1, 16); CALC(1, 17); CALC(1, 18); CALC(1, 19);
    CALC(2, 20); CALC(2, 21); CALC(2, 22); CALC(2, 23); CALC(2, 24);
    CALC(2, 25); CALC(2, 26); CALC(2, 27); CALC(2, 28); CALC(2, 29);
    CALC(2, 30); CALC(2, 31); CALC(2, 32); CALC(2, 33); CALC(2, 34);
    CALC(2, 35); CALC(2, 36); CALC(2, 37); CALC(2, 38); CALC(2, 39);
    CALC(3, 40); CALC(3, 41); CALC(3, 42); CALC(3, 43); CALC(3, 44);
    CALC(3, 45); CALC(3, 46); CALC(3, 47); CALC(3, 48); CALC(3, 49);
    CALC(3, 50); CALC(3, 51); CALC(3, 52); CALC(3, 53); CALC(3, 54);
    CALC(3, 55); CALC(3, 56); CALC(3, 57); CALC(3, 58); CALC(3, 59);
    CALC(4, 60); CALC(4, 61); CALC(4, 62); CALC(4, 63); CALC(4, 64);
    CALC(4, 65); CALC(4, 66); CALC(4, 67); CALC(4, 68); CALC(4, 69);
    CALC(4, 70); CALC(4, 71); CALC(4, 72); CALC(4, 73); CALC(4, 74);
    CALC(4, 75); CALC(4, 76); CALC(4, 77); CALC(4, 78); CALC(4, 79);

    // That needs to be done, == with like (A + I1) =0 hash[0] 
    // is wrong all the time?!
    return A + I1 == h0 &&
        B + I2 == h1 &&
        C + I3 == h2 &&
        D + I4 == h3 &&
        E + I5 == h4;
}

/*
* device function __device__ void prsha1_mem_init(uint, uchar, int)
* Prepare word for sha-1 (expand, add length etc)
*/
__device__ inline void prsha1_mem_init(uint32_t* tmp, const unsigned char* input, const int length) {

    int stop = 0;
    // reseting tmp
    for (size_t i = 0; i < 80; i++) tmp[i] = 0;

    // fill tmp like: message char c0,c1,c2,...,cn,10000000,00...000
    for (size_t i = 0; i < length; i += 4) {
        for (size_t j = 0; j < 4; j++)
            if (i + j < length)
                tmp[i / 4] |= input[i + j] << (24 - j * 8);
            else {
                stop = 1;
                break;
            }
            if (stop)
                break;
    }
    tmp[length / 4] |= 0x80 << (24 - (length % 4) * 8); // Append 1 then zeros
                                                        // Adding length as last value
    tmp[15] |= length * 8;
}
