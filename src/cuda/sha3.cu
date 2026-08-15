/*!
 * \brief   SHA-3 / Keccak CUDA brute-force (short passwords, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 *
 * Keccak state lives in scalar registers (not a uint64_t[25] array). On sm_75
 * an array form spills ~200 B/thread to local memory and the kernel is no
 * faster than a CPU core; scalars keep the permutation in-register.
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#define MAX_HASH_LEN 64

__constant__ static unsigned char k_dict[GPU_DICT_MAX];
__constant__ static unsigned char k_hash[MAX_HASH_LEN];

__constant__ static const uint64_t k_rc[24] = {
    UINT64_C(0x0000000000000001), UINT64_C(0x0000000000008082), UINT64_C(0x800000000000808A), UINT64_C(0x8000000080008000),
    UINT64_C(0x000000000000808B), UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008009),
    UINT64_C(0x000000000000008A), UINT64_C(0x0000000000000088), UINT64_C(0x0000000080008009), UINT64_C(0x000000008000000A),
    UINT64_C(0x000000008000808B), UINT64_C(0x800000000000008B), UINT64_C(0x8000000000008089), UINT64_C(0x8000000000008003),
    UINT64_C(0x8000000000008002), UINT64_C(0x8000000000000080), UINT64_C(0x000000000000800A), UINT64_C(0x800000008000000A),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008080), UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008008),
};

/// One Keccak-f1600 round. a00..a04 = linear state[0..4], a10..=state[5..], …
/// Rho+pi fusion matches the array implementation in this file's history.
#define KECCAK_ROUND(rc)                                                                                 \
    do {                                                                                                 \
        const uint64_t c0 = a00 ^ a10 ^ a20 ^ a30 ^ a40;                                                 \
        const uint64_t c1 = a01 ^ a11 ^ a21 ^ a31 ^ a41;                                                 \
        const uint64_t c2 = a02 ^ a12 ^ a22 ^ a32 ^ a42;                                                 \
        const uint64_t c3 = a03 ^ a13 ^ a23 ^ a33 ^ a43;                                                 \
        const uint64_t c4 = a04 ^ a14 ^ a24 ^ a34 ^ a44;                                                 \
        const uint64_t d0 = ROTL64(c1, 1) ^ c4;                                                          \
        const uint64_t d1 = ROTL64(c2, 1) ^ c0;                                                          \
        const uint64_t d2 = ROTL64(c3, 1) ^ c1;                                                          \
        const uint64_t d3 = ROTL64(c4, 1) ^ c2;                                                          \
        const uint64_t d4 = ROTL64(c0, 1) ^ c3;                                                          \
        const uint64_t b00 = (a00 ^ d0);                                                                 \
        const uint64_t b01 = ROTL64(a11 ^ d1, 44);                                                       \
        const uint64_t b02 = ROTL64(a22 ^ d2, 43);                                                       \
        const uint64_t b03 = ROTL64(a33 ^ d3, 21);                                                       \
        const uint64_t b04 = ROTL64(a44 ^ d4, 14);                                                       \
        const uint64_t b10 = ROTL64(a03 ^ d3, 28);                                                       \
        const uint64_t b11 = ROTL64(a14 ^ d4, 20);                                                       \
        const uint64_t b12 = ROTL64(a20 ^ d0, 3);                                                        \
        const uint64_t b13 = ROTL64(a31 ^ d1, 45);                                                       \
        const uint64_t b14 = ROTL64(a42 ^ d2, 61);                                                       \
        const uint64_t b20 = ROTL64(a01 ^ d1, 1);                                                        \
        const uint64_t b21 = ROTL64(a12 ^ d2, 6);                                                        \
        const uint64_t b22 = ROTL64(a23 ^ d3, 25);                                                       \
        const uint64_t b23 = ROTL64(a34 ^ d4, 8);                                                        \
        const uint64_t b24 = ROTL64(a40 ^ d0, 18);                                                       \
        const uint64_t b30 = ROTL64(a04 ^ d4, 27);                                                       \
        const uint64_t b31 = ROTL64(a10 ^ d0, 36);                                                       \
        const uint64_t b32 = ROTL64(a21 ^ d1, 10);                                                       \
        const uint64_t b33 = ROTL64(a32 ^ d2, 15);                                                       \
        const uint64_t b34 = ROTL64(a43 ^ d3, 56);                                                       \
        const uint64_t b40 = ROTL64(a02 ^ d2, 62);                                                       \
        const uint64_t b41 = ROTL64(a13 ^ d3, 55);                                                       \
        const uint64_t b42 = ROTL64(a24 ^ d4, 39);                                                       \
        const uint64_t b43 = ROTL64(a30 ^ d0, 41);                                                       \
        const uint64_t b44 = ROTL64(a41 ^ d1, 2);                                                        \
        a00 = b00 ^ ((~b01) & b02) ^ (rc);                                                               \
        a01 = b01 ^ ((~b02) & b03);                                                                      \
        a02 = b02 ^ ((~b03) & b04);                                                                      \
        a03 = b03 ^ ((~b04) & b00);                                                                      \
        a04 = b04 ^ ((~b00) & b01);                                                                      \
        a10 = b10 ^ ((~b11) & b12);                                                                      \
        a11 = b11 ^ ((~b12) & b13);                                                                      \
        a12 = b12 ^ ((~b13) & b14);                                                                      \
        a13 = b13 ^ ((~b14) & b10);                                                                      \
        a14 = b14 ^ ((~b10) & b11);                                                                      \
        a20 = b20 ^ ((~b21) & b22);                                                                      \
        a21 = b21 ^ ((~b22) & b23);                                                                      \
        a22 = b22 ^ ((~b23) & b24);                                                                      \
        a23 = b23 ^ ((~b24) & b20);                                                                      \
        a24 = b24 ^ ((~b20) & b21);                                                                      \
        a30 = b30 ^ ((~b31) & b32);                                                                      \
        a31 = b31 ^ ((~b32) & b33);                                                                      \
        a32 = b32 ^ ((~b33) & b34);                                                                      \
        a33 = b33 ^ ((~b34) & b30);                                                                      \
        a34 = b34 ^ ((~b30) & b31);                                                                      \
        a40 = b40 ^ ((~b41) & b42);                                                                      \
        a41 = b41 ^ ((~b42) & b43);                                                                      \
        a42 = b42 ^ ((~b43) & b44);                                                                      \
        a43 = b43 ^ ((~b44) & b40);                                                                      \
        a44 = b44 ^ ((~b40) & b41);                                                                      \
    } while (0)

#define KECCAK_F1600()                                                                                   \
    do {                                                                                                 \
        KECCAK_ROUND(k_rc[0]);                                                                           \
        KECCAK_ROUND(k_rc[1]);                                                                           \
        KECCAK_ROUND(k_rc[2]);                                                                           \
        KECCAK_ROUND(k_rc[3]);                                                                           \
        KECCAK_ROUND(k_rc[4]);                                                                           \
        KECCAK_ROUND(k_rc[5]);                                                                           \
        KECCAK_ROUND(k_rc[6]);                                                                           \
        KECCAK_ROUND(k_rc[7]);                                                                           \
        KECCAK_ROUND(k_rc[8]);                                                                           \
        KECCAK_ROUND(k_rc[9]);                                                                           \
        KECCAK_ROUND(k_rc[10]);                                                                          \
        KECCAK_ROUND(k_rc[11]);                                                                          \
        KECCAK_ROUND(k_rc[12]);                                                                          \
        KECCAK_ROUND(k_rc[13]);                                                                          \
        KECCAK_ROUND(k_rc[14]);                                                                          \
        KECCAK_ROUND(k_rc[15]);                                                                          \
        KECCAK_ROUND(k_rc[16]);                                                                          \
        KECCAK_ROUND(k_rc[17]);                                                                          \
        KECCAK_ROUND(k_rc[18]);                                                                          \
        KECCAK_ROUND(k_rc[19]);                                                                          \
        KECCAK_ROUND(k_rc[20]);                                                                          \
        KECCAK_ROUND(k_rc[21]);                                                                          \
        KECCAK_ROUND(k_rc[22]);                                                                          \
        KECCAK_ROUND(k_rc[23]);                                                                          \
    } while (0)

/// Absorb a short password (len < 16 ≤ every SHA-3 rate) into lane scalars.
/// Layout: a00..a04 = state[0..4], a10..a14 = state[5..9], … (row-major sheets).
__device__ __forceinline__ void prsha3_absorb_short(
    uint64_t& a00, uint64_t& a01, uint64_t& a02, uint64_t& a03, uint64_t& a04,
    uint64_t& a10, uint64_t& a11, uint64_t& a12, uint64_t& a13, uint64_t& a14,
    uint64_t& a20, uint64_t& a21, uint64_t& a22, uint64_t& a23, uint64_t& a24,
    uint64_t& a30, uint64_t& a31, uint64_t& a32, uint64_t& a33, uint64_t& a34,
    uint64_t& a40, uint64_t& a41, uint64_t& a42, uint64_t& a43, uint64_t& a44,
    const uint8_t* message, int len, unsigned rate, uint8_t pad) {
    a00 = a01 = a02 = a03 = a04 = 0;
    a10 = a11 = a12 = a13 = a14 = 0;
    a20 = a21 = a22 = a23 = a24 = 0;
    a30 = a31 = a32 = a33 = a34 = 0;
    a40 = a41 = a42 = a43 = a44 = 0;

    uint64_t lane0 = 0;
    uint64_t lane1 = 0;
    for (int i = 0; i < len; i++) {
        if (i < 8) {
            lane0 |= static_cast<uint64_t>(message[i]) << (i << 3);
        } else {
            lane1 |= static_cast<uint64_t>(message[i]) << ((i - 8) << 3);
        }
    }
    if (len < 8) {
        lane0 |= static_cast<uint64_t>(pad) << (len << 3);
    } else {
        lane1 |= static_cast<uint64_t>(pad) << ((len - 8) << 3);
    }
    a00 = lane0;
    a01 = lane1;

    // Multi-rate 0x80 at byte (rate - 1). rate is always a multiple of 8.
    // last ∈ {8,12,16,17} for rates {72,104,136,144} → lanes a13,a22,a31,a32.
    const unsigned last = (rate - 1u) >> 3;
    const uint64_t sep = UINT64_C(0x80) << 56;
    if (last == 8) a13 ^= sep;
    else if (last == 12) a22 ^= sep;
    else if (last == 16) a31 ^= sep;
    else a32 ^= sep; // last == 17
}

__device__ __forceinline__ BOOL prsha3_digest_eq(
    uint64_t a00, uint64_t a01, uint64_t a02, uint64_t a03, uint64_t a04,
    uint64_t a10, uint64_t a11, uint64_t a12, uint64_t a13, uint64_t a14,
    uint64_t a20, uint64_t a21, uint64_t a22, uint64_t a23, uint64_t a24,
    uint64_t a30, uint64_t a31, uint64_t a32, uint64_t a33, uint64_t a34,
    uint64_t a40, uint64_t a41, uint64_t a42, uint64_t a43, uint64_t a44,
    unsigned hash_len) {
    // Output lanes are a00,a01,a02,... in linear index order (x+5y).
    const uint64_t lanes[8] = { a00, a01, a02, a03, a04, a10, a11, a12 };
    for (unsigned i = 0; i < hash_len; i++) {
        const uint8_t b = static_cast<uint8_t>(lanes[i >> 3] >> ((i & 7) << 3));
        if (b != k_hash[i]) return FALSE;
    }
    return TRUE;
}

__host__ static void prsha3_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                    const unsigned char* hash, unsigned hash_len, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    GPU_COPY_DICT_TO_SYMBOL(k_dict, dict, dict_len);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, hash_len, 0, cudaMemcpyHostToDevice));
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

#define SHA3_VARIANT(name, rate, hash_len, pad)                                                          \
__device__ static BOOL pr##name##_compare(unsigned char* password, const int length) {                   \
    uint64_t a00, a01, a02, a03, a04, a10, a11, a12, a13, a14;                                           \
    uint64_t a20, a21, a22, a23, a24, a30, a31, a32, a33, a34;                                           \
    uint64_t a40, a41, a42, a43, a44;                                                                    \
    prsha3_absorb_short(a00, a01, a02, a03, a04, a10, a11, a12, a13, a14,                                \
                        a20, a21, a22, a23, a24, a30, a31, a32, a33, a34,                                \
                        a40, a41, a42, a43, a44, password, length, rate, pad);                           \
    KECCAK_F1600();                                                                                      \
    return prsha3_digest_eq(a00, a01, a02, a03, a04, a10, a11, a12, a13, a14,                            \
                            a20, a21, a22, a23, a24, a30, a31, a32, a33, a34,                            \
                            a40, a41, a42, a43, a44, hash_len);                                          \
}                                                                                                        \
__global__ static void pr##name##_kernel(unsigned char* result, const uint64_t start, const uint32_t count,\
                                         const uint32_t pass_len, const uint32_t dict_length,             \
                                         const uint32_t min_len) {                                       \
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;                                           \
    if (ix >= count) return;                                                                             \
    uint64_t idx = start + ix;                                                                           \
    unsigned char attempt[GPU_ATTEMPT_SIZE];                                                              \
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {                                                 \
        attempt[pos] = k_dict[idx % dict_length];                                                        \
        idx /= dict_length;                                                                              \
    }                                                                                                    \
    for (uint32_t i = 0; i < dict_length; ++i) {                                                         \
        attempt[pass_len] = k_dict[i];                                                                   \
        if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {                                           \
            if (pr##name##_compare(attempt, (int)(pass_len + 1u))) {                                     \
                memcpy(result, attempt, pass_len + 1u);                                                  \
                return;                                                                                  \
            }                                                                                            \
        }                                                                                                \
        if (pass_len + 2u < min_len) continue;                                                           \
        for (uint32_t j = 0; j < dict_length; ++j) {                                                     \
            attempt[pass_len + 1u] = k_dict[j];                                                          \
            if (pr##name##_compare(attempt, (int)(pass_len + 2u))) {                                     \
                memcpy(result, attempt, pass_len + 2u);                                                  \
                return;                                                                                  \
            }                                                                                            \
        }                                                                                                \
    }                                                                                                    \
}                                                                                                        \
__host__ static void pr##name##_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {                 \
    GPU_LAUNCH_INDEX_KERNEL(pr##name##_kernel, ctx, dict_len);                                           \
}                                                                                                        \
__host__ void HC_GPU_FN(name##_on_gpu_prepare)(int device_ix, const unsigned char* dict, size_t dict_len, \
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx) {                   \
    prsha3_prepare(device_ix, dict, dict_len, hash, hash_len, ctx);                                      \
}                                                                                                        \
__host__ void HC_GPU_FN(name##_run_on_gpu)(gpu_tread_ctx_t * ctx, const size_t dict_len) {               \
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
