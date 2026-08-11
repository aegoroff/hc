/*!
 * \brief   SHA-3 / Keccak CUDA brute-force (short passwords, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"
#include "sha3.h"

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#define MAX_HASH_LEN 64
#define MAX_RATE 144

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[MAX_HASH_LEN];

__constant__ static const uint64_t k_rc[24] = {
    UINT64_C(0x0000000000000001), UINT64_C(0x0000000000008082), UINT64_C(0x800000000000808A), UINT64_C(0x8000000080008000),
    UINT64_C(0x000000000000808B), UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008009),
    UINT64_C(0x000000000000008A), UINT64_C(0x0000000000000088), UINT64_C(0x0000000080008009), UINT64_C(0x000000008000000A),
    UINT64_C(0x000000008000808B), UINT64_C(0x800000000000008B), UINT64_C(0x8000000000008089), UINT64_C(0x8000000000008003),
    UINT64_C(0x8000000000008002), UINT64_C(0x8000000000000080), UINT64_C(0x000000000000800A), UINT64_C(0x800000008000000A),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008080), UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008008),
};

__device__ __forceinline__ void prkeccak_f1600(uint64_t* A) {
#pragma unroll 1
    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5];
#pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        D[0] = ROTL64(C[1], 1) ^ C[4];
        D[1] = ROTL64(C[2], 1) ^ C[0];
        D[2] = ROTL64(C[3], 1) ^ C[1];
        D[3] = ROTL64(C[4], 1) ^ C[2];
        D[4] = ROTL64(C[0], 1) ^ C[3];
#pragma unroll
        for (int x = 0; x < 5; x++) {
            A[x] ^= D[x];
            A[x + 5] ^= D[x];
            A[x + 10] ^= D[x];
            A[x + 15] ^= D[x];
            A[x + 20] ^= D[x];
        }

        A[1] = ROTL64(A[1], 1);
        A[2] = ROTL64(A[2], 62);
        A[3] = ROTL64(A[3], 28);
        A[4] = ROTL64(A[4], 27);
        A[5] = ROTL64(A[5], 36);
        A[6] = ROTL64(A[6], 44);
        A[7] = ROTL64(A[7], 6);
        A[8] = ROTL64(A[8], 55);
        A[9] = ROTL64(A[9], 20);
        A[10] = ROTL64(A[10], 3);
        A[11] = ROTL64(A[11], 10);
        A[12] = ROTL64(A[12], 43);
        A[13] = ROTL64(A[13], 25);
        A[14] = ROTL64(A[14], 39);
        A[15] = ROTL64(A[15], 41);
        A[16] = ROTL64(A[16], 45);
        A[17] = ROTL64(A[17], 15);
        A[18] = ROTL64(A[18], 21);
        A[19] = ROTL64(A[19], 8);
        A[20] = ROTL64(A[20], 18);
        A[21] = ROTL64(A[21], 2);
        A[22] = ROTL64(A[22], 61);
        A[23] = ROTL64(A[23], 56);
        A[24] = ROTL64(A[24], 14);

        {
            uint64_t A1 = A[1];
            A[1] = A[6];
            A[6] = A[9];
            A[9] = A[22];
            A[22] = A[14];
            A[14] = A[20];
            A[20] = A[2];
            A[2] = A[12];
            A[12] = A[13];
            A[13] = A[19];
            A[19] = A[23];
            A[23] = A[15];
            A[15] = A[4];
            A[4] = A[24];
            A[24] = A[21];
            A[21] = A[8];
            A[8] = A[16];
            A[16] = A[5];
            A[5] = A[3];
            A[3] = A[18];
            A[18] = A[17];
            A[17] = A[11];
            A[11] = A[7];
            A[7] = A[10];
            A[10] = A1;
        }

#pragma unroll
        for (int i = 0; i < 25; i += 5) {
            uint64_t A0 = A[0 + i], A1 = A[1 + i];
            A[0 + i] ^= ~A1 & A[2 + i];
            A[1 + i] ^= ~A[2 + i] & A[3 + i];
            A[2 + i] ^= ~A[3 + i] & A[4 + i];
            A[3 + i] ^= ~A[4 + i] & A0;
            A[4 + i] ^= ~A0 & A1;
        }

        A[0] ^= k_rc[round];
    }
}

__device__ __forceinline__ void prsha3_hash(const uint8_t* message, size_t len, uint8_t* out,
                                             unsigned rate, unsigned out_len, uint8_t pad) {
    uint64_t state[25] = {};
    uint8_t block[MAX_RATE] = {};
    memcpy(block, message, len);
    block[len] |= pad;
    block[rate - 1] |= 0x80;

    const unsigned nq = rate / 8;
    for (unsigned i = 0; i < nq; i++) {
        const unsigned o = i * 8;
        state[i] ^= static_cast<uint64_t>(block[o + 0])
            | (static_cast<uint64_t>(block[o + 1]) << 8)
            | (static_cast<uint64_t>(block[o + 2]) << 16)
            | (static_cast<uint64_t>(block[o + 3]) << 24)
            | (static_cast<uint64_t>(block[o + 4]) << 32)
            | (static_cast<uint64_t>(block[o + 5]) << 40)
            | (static_cast<uint64_t>(block[o + 6]) << 48)
            | (static_cast<uint64_t>(block[o + 7]) << 56);
    }
    prkeccak_f1600(state);

    for (unsigned i = 0; i < out_len; i++) {
        out[i] = static_cast<uint8_t>(state[i >> 3] >> ((i & 7) << 3));
    }
}

__host__ static void prsha3_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                    const unsigned char* hash, unsigned hash_len, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, hash_len, 0, cudaMemcpyHostToDevice));
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

#define SHA3_VARIANT(name, rate, hash_len, pad)                                                          \
__device__ static BOOL pr##name##_compare(unsigned char* password, const int length, uint8_t* hash) {    \
    prsha3_hash(password, length, hash, rate, hash_len, pad);                                            \
    BOOL result = TRUE;                                                                                  \
    for (int i = 0; i < (int)(hash_len) && result; ++i) {                                                \
        result &= hash[i] == k_hash[i];                                                                  \
    }                                                                                                    \
    return result;                                                                                       \
}                                                                                                        \
__global__ static void pr##name##_kernel(unsigned char* result, const uint64_t start, const uint32_t count,\
                                         const uint32_t pass_len, const uint32_t dict_length,             \
                                         const uint32_t min_len) {                                       \
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;                                           \
    if (ix >= count) return;                                                                             \
    uint64_t idx = start + ix;                                                                           \
    unsigned char attempt[GPU_ATTEMPT_SIZE];                                                              \
    uint8_t hash[hash_len];                                                                              \
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {                                                 \
        attempt[pos] = k_dict[idx % dict_length];                                                        \
        idx /= dict_length;                                                                              \
    }                                                                                                    \
    for (uint32_t i = 0; i < dict_length; ++i) {                                                         \
        attempt[pass_len] = k_dict[i];                                                                   \
        if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {                                           \
            if (pr##name##_compare(attempt, (int)(pass_len + 1u), hash)) {                               \
                memcpy(result, attempt, pass_len + 1u);                                                  \
                return;                                                                                  \
            }                                                                                            \
        }                                                                                                \
        if (pass_len + 2u < min_len) continue;                                                           \
        for (uint32_t j = 0; j < dict_length; ++j) {                                                     \
            attempt[pass_len + 1u] = k_dict[j];                                                          \
            if (pr##name##_compare(attempt, (int)(pass_len + 2u), hash)) {                               \
                memcpy(result, attempt, pass_len + 2u);                                                  \
                return;                                                                                  \
            }                                                                                            \
        }                                                                                                \
    }                                                                                                    \
}                                                                                                        \
__host__ static void pr##name##_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {                 \
    GPU_LAUNCH_INDEX_KERNEL(pr##name##_kernel, ctx, dict_len);                                           \
}                                                                                                        \
__host__ void name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,           \
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx) {                   \
    prsha3_prepare(device_ix, dict, dict_len, hash, hash_len, ctx);                                      \
}                                                                                                        \
__host__ void name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {                          \
    gpu_run(ctx, dict_len, &pr##name##_run_kernel);                                                      \
}

SHA3_VARIANT(sha3_224, 144, 28, 0x06)
SHA3_VARIANT(sha3_256, 136, 32, 0x06)
SHA3_VARIANT(sha3_384, 104, 48, 0x06)
SHA3_VARIANT(sha3_512, 72, 64, 0x06)
SHA3_VARIANT(keccak_224, 144, 28, 0x01)
SHA3_VARIANT(keccak_256, 136, 32, 0x01)
SHA3_VARIANT(keccak_384, 104, 48, 0x01)
SHA3_VARIANT(keccak_512, 72, 64, 0x01)
