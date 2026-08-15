#ifndef HC_GPU_HASHES_H_
#define HC_GPU_HASHES_H_

/*
 * Single list of GPU hash ABI symbols (run + prepare).
 * Dual-backend builds wrap names via HC_GPU_FN (see gpu_prefix.h).
 */
#ifndef HC_GPU_FN
#define HC_GPU_FN(name) name
#endif

#define HC_GPU_HASHES(X) \
    X(md5)               \
    X(md2)               \
    X(md4)               \
    X(sha1)              \
    X(sha224)            \
    X(sha256)            \
    X(sha384)            \
    X(sha512)            \
    X(sha3_224)          \
    X(sha3_256)          \
    X(sha3_384)          \
    X(sha3_512)          \
    X(keccak_224)        \
    X(keccak_256)        \
    X(keccak_384)        \
    X(keccak_512)        \
    X(rmd128)            \
    X(rmd160)            \
    X(rmd256)            \
    X(rmd320)            \
    X(blake2s)           \
    X(blake2b)           \
    X(blake3)            \
    X(tiger)             \
    X(tiger2)            \
    X(whirl)             \
    X(crc32)

#define HC_GPU_HASH_DECL(name)                                                                       \
    void HC_GPU_FN(name##_run_on_gpu)(gpu_tread_ctx_t* ctx, const size_t dict_len);                   \
    void HC_GPU_FN(name##_on_gpu_prepare)(int device_ix, const unsigned char* dict, size_t dict_len, \
                                          const unsigned char* hash, gpu_tread_ctx_t* ctx);

#define HC_GPU_HASH_DECL_PFX(pfx, name)                                                                 \
    void pfx##name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);                           \
    void pfx##name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,          \
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx);

#endif /* HC_GPU_HASHES_H_ */
