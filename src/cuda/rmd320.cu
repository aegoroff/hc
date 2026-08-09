/*!
 * \brief   RIPEMD-320 CUDA brute-force (short passwords, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"
#include "rmd320.h"

#define BLOCK_LEN 64
#define HASH_LEN 40

#define F(x, y, z) ((x) ^ (y) ^ (z))
#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define H(x, y, z) (((x) | ~(y)) ^ (z))
#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define J(x, y, z) ((x) ^ ((y) | ~(z)))
#define ROTL32(x, n) (((0U + (x)) << (n)) | ((x) >> (32 - (n))))

#define FF(a, b, c, d, e, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define GG(a, b, c, d, e, x, s) \
    { (a) += G((b), (c), (d)) + (x) + UINT32_C(0x5a827999); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define HH(a, b, c, d, e, x, s) \
    { (a) += H((b), (c), (d)) + (x) + UINT32_C(0x6ed9eba1); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define II(a, b, c, d, e, x, s) \
    { (a) += I((b), (c), (d)) + (x) + UINT32_C(0x8f1bbcdc); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define JJ(a, b, c, d, e, x, s) \
    { (a) += J((b), (c), (d)) + (x) + UINT32_C(0xa953fd4e); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define FFF(a, b, c, d, e, x, s) \
    { (a) += F((b), (c), (d)) + (x); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define GGG(a, b, c, d, e, x, s) \
    { (a) += G((b), (c), (d)) + (x) + UINT32_C(0x7a6d76e9); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define HHH(a, b, c, d, e, x, s) \
    { (a) += H((b), (c), (d)) + (x) + UINT32_C(0x6d703ef3); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define III(a, b, c, d, e, x, s) \
    { (a) += I((b), (c), (d)) + (x) + UINT32_C(0x5c4dd124); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }
#define JJJ(a, b, c, d, e, x, s) \
    { (a) += J((b), (c), (d)) + (x) + UINT32_C(0x50a28be6); (a) = ROTL32((a), (s)) + (e); (c) = ROTL32((c), 10); }

__global__ static void prrmd320_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                       const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__device__ static BOOL prrmd320_compare(unsigned char* password, const int length, uint8_t* hash);
__device__ static void prrmd320_hash(const uint8_t* message, size_t len, uint8_t* hash);
__device__ static void prrmd320_compress(uint32_t* state, const uint8_t* block);

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[HASH_LEN];

__host__ void rmd320_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, HASH_LEN, 0, cudaMemcpyHostToDevice));
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ static void prrmd320_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    GPU_LAUNCH_INDEX_KERNEL(prrmd320_kernel, ctx, dict_len);
}

__host__ void rmd320_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prrmd320_run_kernel);
}

KERNEL_WITH_ALLOCATION(prrmd320_kernel, prrmd320_compare, uint8_t, HASH_LEN)

__device__ __forceinline__ BOOL prrmd320_compare(unsigned char* password, const int length, uint8_t* hash) {
    prrmd320_hash(password, length, hash);
    BOOL result = TRUE;
    for (int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == k_hash[i];
    }
    return result;
}

__device__ __forceinline__ void prrmd320_hash(const uint8_t* message, size_t len, uint8_t* hash) {
    uint32_t state[10] = {
        UINT32_C(0x67452301), UINT32_C(0xEFCDAB89), UINT32_C(0x98BADCFE), UINT32_C(0x10325476), UINT32_C(0xC3D2E1F0),
        UINT32_C(0x76543210), UINT32_C(0xFEDCBA98), UINT32_C(0x89ABCDEF), UINT32_C(0x01234567), UINT32_C(0x3C2D1E0F),
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

    prrmd320_compress(state, block);

    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = static_cast<uint8_t>(state[i >> 2] >> ((i & 3) << 3));
}

__device__ __forceinline__ void prrmd320_compress(uint32_t* state, const uint8_t* block) {
    uint32_t X[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int i = j * 4;
        X[j] = static_cast<uint32_t>(block[i + 0])
            | (static_cast<uint32_t>(block[i + 1]) << 8)
            | (static_cast<uint32_t>(block[i + 2]) << 16)
            | (static_cast<uint32_t>(block[i + 3]) << 24);
    }

    uint32_t aa = state[0], bb = state[1], cc = state[2], dd = state[3], ee = state[4];
    uint32_t aaa = state[5], bbb = state[6], ccc = state[7], ddd = state[8], eee = state[9];
    uint32_t tmp;

    FF(aa, bb, cc, dd, ee, X[ 0], 11);
    FF(ee, aa, bb, cc, dd, X[ 1], 14);
    FF(dd, ee, aa, bb, cc, X[ 2], 15);
    FF(cc, dd, ee, aa, bb, X[ 3], 12);
    FF(bb, cc, dd, ee, aa, X[ 4],  5);
    FF(aa, bb, cc, dd, ee, X[ 5],  8);
    FF(ee, aa, bb, cc, dd, X[ 6],  7);
    FF(dd, ee, aa, bb, cc, X[ 7],  9);
    FF(cc, dd, ee, aa, bb, X[ 8], 11);
    FF(bb, cc, dd, ee, aa, X[ 9], 13);
    FF(aa, bb, cc, dd, ee, X[10], 14);
    FF(ee, aa, bb, cc, dd, X[11], 15);
    FF(dd, ee, aa, bb, cc, X[12],  6);
    FF(cc, dd, ee, aa, bb, X[13],  7);
    FF(bb, cc, dd, ee, aa, X[14],  9);
    FF(aa, bb, cc, dd, ee, X[15],  8);
    JJJ(aaa, bbb, ccc, ddd, eee, X[ 5],  8);
    JJJ(eee, aaa, bbb, ccc, ddd, X[14],  9);
    JJJ(ddd, eee, aaa, bbb, ccc, X[ 7],  9);
    JJJ(ccc, ddd, eee, aaa, bbb, X[ 0], 11);
    JJJ(bbb, ccc, ddd, eee, aaa, X[ 9], 13);
    JJJ(aaa, bbb, ccc, ddd, eee, X[ 2], 15);
    JJJ(eee, aaa, bbb, ccc, ddd, X[11], 15);
    JJJ(ddd, eee, aaa, bbb, ccc, X[ 4],  5);
    JJJ(ccc, ddd, eee, aaa, bbb, X[13],  7);
    JJJ(bbb, ccc, ddd, eee, aaa, X[ 6],  7);
    JJJ(aaa, bbb, ccc, ddd, eee, X[15],  8);
    JJJ(eee, aaa, bbb, ccc, ddd, X[ 8], 11);
    JJJ(ddd, eee, aaa, bbb, ccc, X[ 1], 14);
    JJJ(ccc, ddd, eee, aaa, bbb, X[10], 14);
    JJJ(bbb, ccc, ddd, eee, aaa, X[ 3], 12);
    JJJ(aaa, bbb, ccc, ddd, eee, X[12],  6);
    tmp = aa; aa = aaa; aaa = tmp;
    GG(ee, aa, bb, cc, dd, X[ 7],  7);
    GG(dd, ee, aa, bb, cc, X[ 4],  6);
    GG(cc, dd, ee, aa, bb, X[13],  8);
    GG(bb, cc, dd, ee, aa, X[ 1], 13);
    GG(aa, bb, cc, dd, ee, X[10], 11);
    GG(ee, aa, bb, cc, dd, X[ 6],  9);
    GG(dd, ee, aa, bb, cc, X[15],  7);
    GG(cc, dd, ee, aa, bb, X[ 3], 15);
    GG(bb, cc, dd, ee, aa, X[12],  7);
    GG(aa, bb, cc, dd, ee, X[ 0], 12);
    GG(ee, aa, bb, cc, dd, X[ 9], 15);
    GG(dd, ee, aa, bb, cc, X[ 5],  9);
    GG(cc, dd, ee, aa, bb, X[ 2], 11);
    GG(bb, cc, dd, ee, aa, X[14],  7);
    GG(aa, bb, cc, dd, ee, X[11], 13);
    GG(ee, aa, bb, cc, dd, X[ 8], 12);
    III(eee, aaa, bbb, ccc, ddd, X[ 6],  9);
    III(ddd, eee, aaa, bbb, ccc, X[11], 13);
    III(ccc, ddd, eee, aaa, bbb, X[ 3], 15);
    III(bbb, ccc, ddd, eee, aaa, X[ 7],  7);
    III(aaa, bbb, ccc, ddd, eee, X[ 0], 12);
    III(eee, aaa, bbb, ccc, ddd, X[13],  8);
    III(ddd, eee, aaa, bbb, ccc, X[ 5],  9);
    III(ccc, ddd, eee, aaa, bbb, X[10], 11);
    III(bbb, ccc, ddd, eee, aaa, X[14],  7);
    III(aaa, bbb, ccc, ddd, eee, X[15],  7);
    III(eee, aaa, bbb, ccc, ddd, X[ 8], 12);
    III(ddd, eee, aaa, bbb, ccc, X[12],  7);
    III(ccc, ddd, eee, aaa, bbb, X[ 4],  6);
    III(bbb, ccc, ddd, eee, aaa, X[ 9], 15);
    III(aaa, bbb, ccc, ddd, eee, X[ 1], 13);
    III(eee, aaa, bbb, ccc, ddd, X[ 2], 11);
    tmp = bb; bb = bbb; bbb = tmp;
    HH(dd, ee, aa, bb, cc, X[ 3], 11);
    HH(cc, dd, ee, aa, bb, X[10], 13);
    HH(bb, cc, dd, ee, aa, X[14],  6);
    HH(aa, bb, cc, dd, ee, X[ 4],  7);
    HH(ee, aa, bb, cc, dd, X[ 9], 14);
    HH(dd, ee, aa, bb, cc, X[15],  9);
    HH(cc, dd, ee, aa, bb, X[ 8], 13);
    HH(bb, cc, dd, ee, aa, X[ 1], 15);
    HH(aa, bb, cc, dd, ee, X[ 2], 14);
    HH(ee, aa, bb, cc, dd, X[ 7],  8);
    HH(dd, ee, aa, bb, cc, X[ 0], 13);
    HH(cc, dd, ee, aa, bb, X[ 6],  6);
    HH(bb, cc, dd, ee, aa, X[13],  5);
    HH(aa, bb, cc, dd, ee, X[11], 12);
    HH(ee, aa, bb, cc, dd, X[ 5],  7);
    HH(dd, ee, aa, bb, cc, X[12],  5);
    HHH(ddd, eee, aaa, bbb, ccc, X[15],  9);
    HHH(ccc, ddd, eee, aaa, bbb, X[ 5],  7);
    HHH(bbb, ccc, ddd, eee, aaa, X[ 1], 15);
    HHH(aaa, bbb, ccc, ddd, eee, X[ 3], 11);
    HHH(eee, aaa, bbb, ccc, ddd, X[ 7],  8);
    HHH(ddd, eee, aaa, bbb, ccc, X[14],  6);
    HHH(ccc, ddd, eee, aaa, bbb, X[ 6],  6);
    HHH(bbb, ccc, ddd, eee, aaa, X[ 9], 14);
    HHH(aaa, bbb, ccc, ddd, eee, X[11], 12);
    HHH(eee, aaa, bbb, ccc, ddd, X[ 8], 13);
    HHH(ddd, eee, aaa, bbb, ccc, X[12],  5);
    HHH(ccc, ddd, eee, aaa, bbb, X[ 2], 14);
    HHH(bbb, ccc, ddd, eee, aaa, X[10], 13);
    HHH(aaa, bbb, ccc, ddd, eee, X[ 0], 13);
    HHH(eee, aaa, bbb, ccc, ddd, X[ 4],  7);
    HHH(ddd, eee, aaa, bbb, ccc, X[13],  5);
    tmp = cc; cc = ccc; ccc = tmp;
    II(cc, dd, ee, aa, bb, X[ 1], 11);
    II(bb, cc, dd, ee, aa, X[ 9], 12);
    II(aa, bb, cc, dd, ee, X[11], 14);
    II(ee, aa, bb, cc, dd, X[10], 15);
    II(dd, ee, aa, bb, cc, X[ 0], 14);
    II(cc, dd, ee, aa, bb, X[ 8], 15);
    II(bb, cc, dd, ee, aa, X[12],  9);
    II(aa, bb, cc, dd, ee, X[ 4],  8);
    II(ee, aa, bb, cc, dd, X[13],  9);
    II(dd, ee, aa, bb, cc, X[ 3], 14);
    II(cc, dd, ee, aa, bb, X[ 7],  5);
    II(bb, cc, dd, ee, aa, X[15],  6);
    II(aa, bb, cc, dd, ee, X[14],  8);
    II(ee, aa, bb, cc, dd, X[ 5],  6);
    II(dd, ee, aa, bb, cc, X[ 6],  5);
    II(cc, dd, ee, aa, bb, X[ 2], 12);
    GGG(ccc, ddd, eee, aaa, bbb, X[ 8], 15);
    GGG(bbb, ccc, ddd, eee, aaa, X[ 6],  5);
    GGG(aaa, bbb, ccc, ddd, eee, X[ 4],  8);
    GGG(eee, aaa, bbb, ccc, ddd, X[ 1], 11);
    GGG(ddd, eee, aaa, bbb, ccc, X[ 3], 14);
    GGG(ccc, ddd, eee, aaa, bbb, X[11], 14);
    GGG(bbb, ccc, ddd, eee, aaa, X[15],  6);
    GGG(aaa, bbb, ccc, ddd, eee, X[ 0], 14);
    GGG(eee, aaa, bbb, ccc, ddd, X[ 5],  6);
    GGG(ddd, eee, aaa, bbb, ccc, X[12],  9);
    GGG(ccc, ddd, eee, aaa, bbb, X[ 2], 12);
    GGG(bbb, ccc, ddd, eee, aaa, X[13],  9);
    GGG(aaa, bbb, ccc, ddd, eee, X[ 9], 12);
    GGG(eee, aaa, bbb, ccc, ddd, X[ 7],  5);
    GGG(ddd, eee, aaa, bbb, ccc, X[10], 15);
    GGG(ccc, ddd, eee, aaa, bbb, X[14],  8);
    tmp = dd; dd = ddd; ddd = tmp;
    JJ(bb, cc, dd, ee, aa, X[ 4],  9);
    JJ(aa, bb, cc, dd, ee, X[ 0], 15);
    JJ(ee, aa, bb, cc, dd, X[ 5],  5);
    JJ(dd, ee, aa, bb, cc, X[ 9], 11);
    JJ(cc, dd, ee, aa, bb, X[ 7],  6);
    JJ(bb, cc, dd, ee, aa, X[12],  8);
    JJ(aa, bb, cc, dd, ee, X[ 2], 13);
    JJ(ee, aa, bb, cc, dd, X[10], 12);
    JJ(dd, ee, aa, bb, cc, X[14],  5);
    JJ(cc, dd, ee, aa, bb, X[ 1], 12);
    JJ(bb, cc, dd, ee, aa, X[ 3], 13);
    JJ(aa, bb, cc, dd, ee, X[ 8], 14);
    JJ(ee, aa, bb, cc, dd, X[11], 11);
    JJ(dd, ee, aa, bb, cc, X[ 6],  8);
    JJ(cc, dd, ee, aa, bb, X[15],  5);
    JJ(bb, cc, dd, ee, aa, X[13],  6);
    FFF(bbb, ccc, ddd, eee, aaa, X[12] ,  8);
    FFF(aaa, bbb, ccc, ddd, eee, X[15] ,  5);
    FFF(eee, aaa, bbb, ccc, ddd, X[10] , 12);
    FFF(ddd, eee, aaa, bbb, ccc, X[ 4] ,  9);
    FFF(ccc, ddd, eee, aaa, bbb, X[ 1] , 12);
    FFF(bbb, ccc, ddd, eee, aaa, X[ 5] ,  5);
    FFF(aaa, bbb, ccc, ddd, eee, X[ 8] , 14);
    FFF(eee, aaa, bbb, ccc, ddd, X[ 7] ,  6);
    FFF(ddd, eee, aaa, bbb, ccc, X[ 6] ,  8);
    FFF(ccc, ddd, eee, aaa, bbb, X[ 2] , 13);
    FFF(bbb, ccc, ddd, eee, aaa, X[13] ,  6);
    FFF(aaa, bbb, ccc, ddd, eee, X[14] ,  5);
    FFF(eee, aaa, bbb, ccc, ddd, X[ 0] , 15);
    FFF(ddd, eee, aaa, bbb, ccc, X[ 3] , 13);
    FFF(ccc, ddd, eee, aaa, bbb, X[ 9] , 11);
    FFF(bbb, ccc, ddd, eee, aaa, X[11] , 11);
    tmp = ee; ee = eee; eee = tmp;

    state[0] += aa;
    state[1] += bb;
    state[2] += cc;
    state[3] += dd;
    state[4] += ee;
    state[5] += aaa;
    state[6] += bbb;
    state[7] += ccc;
    state[8] += ddd;
    state[9] += eee;
}
