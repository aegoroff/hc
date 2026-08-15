/*!
 * OpenCL brute-force algo ABI — embeds + thin entries over hc_ocl_algo_entry_*.
 */
#include "gpu_abi.h"
#include "ocl_common.h"

#include <stddef.h>

static const char k_md5_src[] = {
#embed "kernels/md5.cl"
, 0
};
static const char k_md4_src[] = {
#embed "kernels/md4.cl"
, 0
};
static const char k_md2_src[] = {
#embed "kernels/md2.cl"
, 0
};
static const char k_sha1_src[] = {
#embed "kernels/sha1.cl"
, 0
};
static const char k_sha224_src[] = {
#embed "kernels/sha224.cl"
, 0
};
static const char k_sha256_src[] = {
#embed "kernels/sha256.cl"
, 0
};
static const char k_sha384_src[] = {
#embed "kernels/sha384.cl"
, 0
};
static const char k_sha512_src[] = {
#embed "kernels/sha512.cl"
, 0
};
static const char k_crc32_src[] = {
#embed "kernels/crc32.cl"
, 0
};
static const char k_rmd128_src[] = {
#embed "kernels/rmd128.cl"
, 0
};
static const char k_rmd160_src[] = {
#embed "kernels/rmd160.cl"
, 0
};
static const char k_rmd256_src[] = {
#embed "kernels/rmd256.cl"
, 0
};
static const char k_rmd320_src[] = {
#embed "kernels/rmd320.cl"
, 0
};
static const char k_blake2s_src[] = {
#embed "kernels/blake2s.cl"
, 0
};
static const char k_blake2b_src[] = {
#embed "kernels/blake2b.cl"
, 0
};
static const char k_blake3_src[] = {
#embed "kernels/blake3.cl"
, 0
};
static const char k_sha3_224_src[] = {
#embed "kernels/sha3_224.cl"
, 0
};
static const char k_sha3_256_src[] = {
#embed "kernels/sha3_256.cl"
, 0
};
static const char k_sha3_384_src[] = {
#embed "kernels/sha3_384.cl"
, 0
};
static const char k_sha3_512_src[] = {
#embed "kernels/sha3_512.cl"
, 0
};
static const char k_keccak_224_src[] = {
#embed "kernels/keccak_224.cl"
, 0
};
static const char k_keccak_256_src[] = {
#embed "kernels/keccak_256.cl"
, 0
};
static const char k_keccak_384_src[] = {
#embed "kernels/keccak_384.cl"
, 0
};
static const char k_keccak_512_src[] = {
#embed "kernels/keccak_512.cl"
, 0
};
static const char k_tiger_src[] = {
#embed "kernels/tiger.cl"
, 0
};
static const char k_tiger2_src[] = {
#embed "kernels/tiger2.cl"
, 0
};
static const char k_whirl_src[] = {
#embed "kernels/whirl.cl"
, 0
};

#define OCL_ENTRY(sym, hash_len, pass_wide)                                                              \
    static hc_ocl_algo_t g_##sym;                                                                        \
    void HC_GPU_FN(sym##_on_gpu_prepare)(int device_ix, const unsigned char* dict, size_t dict_len,      \
                                         const unsigned char* hash, gpu_tread_ctx_t* ctx) {              \
        hc_ocl_algo_entry_prepare(&g_##sym, k_##sym##_src, "pr" #sym "_kernel", (hash_len), (pass_wide), \
                                  device_ix, dict, dict_len, hash, ctx);                                 \
    }                                                                                                    \
    void HC_GPU_FN(sym##_run_on_gpu)(gpu_tread_ctx_t* ctx, const size_t dict_len) {                       \
        hc_ocl_algo_entry_run(ctx, dict_len);                                                            \
    }

OCL_ENTRY(md5, 16, 0)
OCL_ENTRY(md4, 16, 1)
OCL_ENTRY(md2, 16, 0)
OCL_ENTRY(sha1, 20, 0)
OCL_ENTRY(sha224, 28, 0)
OCL_ENTRY(sha256, 32, 0)
OCL_ENTRY(sha384, 48, 0)
OCL_ENTRY(sha512, 64, 0)
OCL_ENTRY(crc32, 4, 0)
OCL_ENTRY(rmd128, 16, 0)
OCL_ENTRY(rmd160, 20, 0)
OCL_ENTRY(rmd256, 32, 0)
OCL_ENTRY(rmd320, 40, 0)
OCL_ENTRY(blake2s, 32, 0)
OCL_ENTRY(blake2b, 64, 0)
OCL_ENTRY(blake3, 32, 0)
OCL_ENTRY(sha3_224, 28, 0)
OCL_ENTRY(sha3_256, 32, 0)
OCL_ENTRY(sha3_384, 48, 0)
OCL_ENTRY(sha3_512, 64, 0)
OCL_ENTRY(keccak_224, 28, 0)
OCL_ENTRY(keccak_256, 32, 0)
OCL_ENTRY(keccak_384, 48, 0)
OCL_ENTRY(keccak_512, 64, 0)
OCL_ENTRY(tiger, 24, 0)
OCL_ENTRY(tiger2, 24, 0)
OCL_ENTRY(whirl, 64, 0)

#undef OCL_ENTRY
