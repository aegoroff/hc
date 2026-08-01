/*!
 * \brief   RIPEMD-256 CUDA brute-force (short passwords, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"
#include "rmd256.h"

#define BLOCK_LEN 64
#define HASH_LEN 32

#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define ROTL32(x, n) (((0U + (x)) << (n)) | ((x) >> (32 - (n))))

#define FF(a, b, c, d, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }
#define GG(a, b, c, d, x, s) \
    { (a) += G((b), (c), (d)) + (x) + UINT32_C(0x5a827999); (a) = ROTL32((a), (s)); }
#define HH(a, b, c, d, x, s) \
    { (a) += H((b), (c), (d)) + (x) + UINT32_C(0x6ed9eba1); (a) = ROTL32((a), (s)); }
#define II(a, b, c, d, x, s) \
    { (a) += I((b), (c), (d)) + (x) + UINT32_C(0x8f1bbcdc); (a) = ROTL32((a), (s)); }
#define FFF(a, b, c, d, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)); }
#define GGG(a, b, c, d, x, s) \
    { (a) += G((b), (c), (d)) + (x) + UINT32_C(0x6d703ef3); (a) = ROTL32((a), (s)); }
#define HHH(a, b, c, d, x, s) \
    { (a) += H((b), (c), (d)) + (x) + UINT32_C(0x5c4dd124); (a) = ROTL32((a), (s)); }
#define III(a, b, c, d, x, s) \
    { (a) += I((b), (c), (d)) + (x) + UINT32_C(0x50a28be6); (a) = ROTL32((a), (s)); }

__global__ static void prrmd256_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                       const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__device__ static BOOL prrmd256_compare(unsigned char* password, const int length, uint8_t* hash);
__device__ static void prrmd256_hash(const uint8_t* message, size_t len, uint8_t* hash);
__device__ static void prrmd256_compress(uint32_t* state, const uint8_t* block);

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[HASH_LEN];

__host__ void rmd256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, HASH_LEN, 0, cudaMemcpyHostToDevice));
    ctx->dev_variants_ = nullptr;
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ static void prrmd256_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants,
                                         const size_t dict_len) {
    (void)dev_result;
    (void)dev_variants;
    GPU_LAUNCH_INDEX_KERNEL(prrmd256_kernel, ctx, dict_len);
}

__host__ void rmd256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
                                const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prrmd256_run_kernel);
}

KERNEL_WITH_ALLOCATION(prrmd256_kernel, prrmd256_compare, uint8_t, HASH_LEN)

__device__ __forceinline__ BOOL prrmd256_compare(unsigned char* password, const int length, uint8_t* hash) {
    prrmd256_hash(password, length, hash);
    BOOL result = TRUE;
    for (int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == k_hash[i];
    }
    return result;
}

__device__ __forceinline__ void prrmd256_hash(const uint8_t* message, size_t len, uint8_t* hash) {
    uint32_t state[8] = {
        UINT32_C(0x67452301), UINT32_C(0xEFCDAB89), UINT32_C(0x98BADCFE), UINT32_C(0x10325476),
        UINT32_C(0x76543210), UINT32_C(0xFEDCBA98), UINT32_C(0x89ABCDEF), UINT32_C(0x01234567),
    };

    uint8_t block[BLOCK_LEN] = {};
    memcpy(block, message, len);
    block[len] = 0x80;
    const uint64_t bitlen = static_cast<uint64_t>(len) << 3;
    block[56] = static_cast<uint8_t>(bitlen);
    block[57] = static_cast<uint8_t>(bitlen >> 8);
    block[58] = static_cast<uint8_t>(bitlen >> 16);
    block[59] = static_cast<uint8_t>(bitlen >> 24);
    block[60] = static_cast<uint8_t>(bitlen >> 32);
    block[61] = static_cast<uint8_t>(bitlen >> 40);
    block[62] = static_cast<uint8_t>(bitlen >> 48);
    block[63] = static_cast<uint8_t>(bitlen >> 56);

    prrmd256_compress(state, block);

    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = static_cast<uint8_t>(state[i >> 2] >> ((i & 3) << 3));
}

__device__ __forceinline__ void prrmd256_compress(uint32_t* state, const uint8_t* block) {
    uint32_t X[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int i = j * 4;
        X[j] = static_cast<uint32_t>(block[i + 0])
            | (static_cast<uint32_t>(block[i + 1]) << 8)
            | (static_cast<uint32_t>(block[i + 2]) << 16)
            | (static_cast<uint32_t>(block[i + 3]) << 24);
    }

    uint32_t aa = state[0], bb = state[1], cc = state[2], dd = state[3];
    uint32_t aaa = state[4], bbb = state[5], ccc = state[6], ddd = state[7];
    uint32_t tmp;

    FF(aa, bb, cc, dd, X[ 0], 11);
    FF(dd, aa, bb, cc, X[ 1], 14);
    FF(cc, dd, aa, bb, X[ 2], 15);
    FF(bb, cc, dd, aa, X[ 3], 12);
    FF(aa, bb, cc, dd, X[ 4],  5);
    FF(dd, aa, bb, cc, X[ 5],  8);
    FF(cc, dd, aa, bb, X[ 6],  7);
    FF(bb, cc, dd, aa, X[ 7],  9);
    FF(aa, bb, cc, dd, X[ 8], 11);
    FF(dd, aa, bb, cc, X[ 9], 13);
    FF(cc, dd, aa, bb, X[10], 14);
    FF(bb, cc, dd, aa, X[11], 15);
    FF(aa, bb, cc, dd, X[12],  6);
    FF(dd, aa, bb, cc, X[13],  7);
    FF(cc, dd, aa, bb, X[14],  9);
    FF(bb, cc, dd, aa, X[15],  8);
    III(aaa, bbb, ccc, ddd, X[ 5],  8);
    III(ddd, aaa, bbb, ccc, X[14],  9);
    III(ccc, ddd, aaa, bbb, X[ 7],  9);
    III(bbb, ccc, ddd, aaa, X[ 0], 11);
    III(aaa, bbb, ccc, ddd, X[ 9], 13);
    III(ddd, aaa, bbb, ccc, X[ 2], 15);
    III(ccc, ddd, aaa, bbb, X[11], 15);
    III(bbb, ccc, ddd, aaa, X[ 4],  5);
    III(aaa, bbb, ccc, ddd, X[13],  7);
    III(ddd, aaa, bbb, ccc, X[ 6],  7);
    III(ccc, ddd, aaa, bbb, X[15],  8);
    III(bbb, ccc, ddd, aaa, X[ 8], 11);
    III(aaa, bbb, ccc, ddd, X[ 1], 14);
    III(ddd, aaa, bbb, ccc, X[10], 14);
    III(ccc, ddd, aaa, bbb, X[ 3], 12);
    III(bbb, ccc, ddd, aaa, X[12],  6);
    tmp = aa; aa = aaa; aaa = tmp;
    GG(aa, bb, cc, dd, X[ 7],  7);
    GG(dd, aa, bb, cc, X[ 4],  6);
    GG(cc, dd, aa, bb, X[13],  8);
    GG(bb, cc, dd, aa, X[ 1], 13);
    GG(aa, bb, cc, dd, X[10], 11);
    GG(dd, aa, bb, cc, X[ 6],  9);
    GG(cc, dd, aa, bb, X[15],  7);
    GG(bb, cc, dd, aa, X[ 3], 15);
    GG(aa, bb, cc, dd, X[12],  7);
    GG(dd, aa, bb, cc, X[ 0], 12);
    GG(cc, dd, aa, bb, X[ 9], 15);
    GG(bb, cc, dd, aa, X[ 5],  9);
    GG(aa, bb, cc, dd, X[ 2], 11);
    GG(dd, aa, bb, cc, X[14],  7);
    GG(cc, dd, aa, bb, X[11], 13);
    GG(bb, cc, dd, aa, X[ 8], 12);
    HHH(aaa, bbb, ccc, ddd, X[ 6],  9);
    HHH(ddd, aaa, bbb, ccc, X[11], 13);
    HHH(ccc, ddd, aaa, bbb, X[ 3], 15);
    HHH(bbb, ccc, ddd, aaa, X[ 7],  7);
    HHH(aaa, bbb, ccc, ddd, X[ 0], 12);
    HHH(ddd, aaa, bbb, ccc, X[13],  8);
    HHH(ccc, ddd, aaa, bbb, X[ 5],  9);
    HHH(bbb, ccc, ddd, aaa, X[10], 11);
    HHH(aaa, bbb, ccc, ddd, X[14],  7);
    HHH(ddd, aaa, bbb, ccc, X[15],  7);
    HHH(ccc, ddd, aaa, bbb, X[ 8], 12);
    HHH(bbb, ccc, ddd, aaa, X[12],  7);
    HHH(aaa, bbb, ccc, ddd, X[ 4],  6);
    HHH(ddd, aaa, bbb, ccc, X[ 9], 15);
    HHH(ccc, ddd, aaa, bbb, X[ 1], 13);
    HHH(bbb, ccc, ddd, aaa, X[ 2], 11);
    tmp = bb; bb = bbb; bbb = tmp;
    HH(aa, bb, cc, dd, X[ 3], 11);
    HH(dd, aa, bb, cc, X[10], 13);
    HH(cc, dd, aa, bb, X[14],  6);
    HH(bb, cc, dd, aa, X[ 4],  7);
    HH(aa, bb, cc, dd, X[ 9], 14);
    HH(dd, aa, bb, cc, X[15],  9);
    HH(cc, dd, aa, bb, X[ 8], 13);
    HH(bb, cc, dd, aa, X[ 1], 15);
    HH(aa, bb, cc, dd, X[ 2], 14);
    HH(dd, aa, bb, cc, X[ 7],  8);
    HH(cc, dd, aa, bb, X[ 0], 13);
    HH(bb, cc, dd, aa, X[ 6],  6);
    HH(aa, bb, cc, dd, X[13],  5);
    HH(dd, aa, bb, cc, X[11], 12);
    HH(cc, dd, aa, bb, X[ 5],  7);
    HH(bb, cc, dd, aa, X[12],  5);
    GGG(aaa, bbb, ccc, ddd, X[15],  9);
    GGG(ddd, aaa, bbb, ccc, X[ 5],  7);
    GGG(ccc, ddd, aaa, bbb, X[ 1], 15);
    GGG(bbb, ccc, ddd, aaa, X[ 3], 11);
    GGG(aaa, bbb, ccc, ddd, X[ 7],  8);
    GGG(ddd, aaa, bbb, ccc, X[14],  6);
    GGG(ccc, ddd, aaa, bbb, X[ 6],  6);
    GGG(bbb, ccc, ddd, aaa, X[ 9], 14);
    GGG(aaa, bbb, ccc, ddd, X[11], 12);
    GGG(ddd, aaa, bbb, ccc, X[ 8], 13);
    GGG(ccc, ddd, aaa, bbb, X[12],  5);
    GGG(bbb, ccc, ddd, aaa, X[ 2], 14);
    GGG(aaa, bbb, ccc, ddd, X[10], 13);
    GGG(ddd, aaa, bbb, ccc, X[ 0], 13);
    GGG(ccc, ddd, aaa, bbb, X[ 4],  7);
    GGG(bbb, ccc, ddd, aaa, X[13],  5);
    tmp = cc; cc = ccc; ccc = tmp;
    II(aa, bb, cc, dd, X[ 1], 11);
    II(dd, aa, bb, cc, X[ 9], 12);
    II(cc, dd, aa, bb, X[11], 14);
    II(bb, cc, dd, aa, X[10], 15);
    II(aa, bb, cc, dd, X[ 0], 14);
    II(dd, aa, bb, cc, X[ 8], 15);
    II(cc, dd, aa, bb, X[12],  9);
    II(bb, cc, dd, aa, X[ 4],  8);
    II(aa, bb, cc, dd, X[13],  9);
    II(dd, aa, bb, cc, X[ 3], 14);
    II(cc, dd, aa, bb, X[ 7],  5);
    II(bb, cc, dd, aa, X[15],  6);
    II(aa, bb, cc, dd, X[14],  8);
    II(dd, aa, bb, cc, X[ 5],  6);
    II(cc, dd, aa, bb, X[ 6],  5);
    II(bb, cc, dd, aa, X[ 2], 12);
    FFF(aaa, bbb, ccc, ddd, X[ 8], 15);
    FFF(ddd, aaa, bbb, ccc, X[ 6],  5);
    FFF(ccc, ddd, aaa, bbb, X[ 4],  8);
    FFF(bbb, ccc, ddd, aaa, X[ 1], 11);
    FFF(aaa, bbb, ccc, ddd, X[ 3], 14);
    FFF(ddd, aaa, bbb, ccc, X[11], 14);
    FFF(ccc, ddd, aaa, bbb, X[15],  6);
    FFF(bbb, ccc, ddd, aaa, X[ 0], 14);
    FFF(aaa, bbb, ccc, ddd, X[ 5],  6);
    FFF(ddd, aaa, bbb, ccc, X[12],  9);
    FFF(ccc, ddd, aaa, bbb, X[ 2], 12);
    FFF(bbb, ccc, ddd, aaa, X[13],  9);
    FFF(aaa, bbb, ccc, ddd, X[ 9], 12);
    FFF(ddd, aaa, bbb, ccc, X[ 7],  5);
    FFF(ccc, ddd, aaa, bbb, X[10], 15);
    FFF(bbb, ccc, ddd, aaa, X[14],  8);
    tmp = dd; dd = ddd; ddd = tmp;

    state[0] += aa;
    state[1] += bb;
    state[2] += cc;
    state[3] += dd;
    state[4] += aaa;
    state[5] += bbb;
    state[6] += ccc;
    state[7] += ddd;
}
