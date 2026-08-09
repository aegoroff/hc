#ifndef LINQ2HASH_BLAKE2S_H_
#define LINQ2HASH_BLAKE2S_H_
#include <stddef.h>
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
    void blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                const unsigned char* hash, gpu_tread_ctx_t* ctx);
#ifdef __cplusplus
}
#endif
#endif
