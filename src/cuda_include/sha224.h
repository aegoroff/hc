#ifndef LINQ2HASH_SHA224_H_
#define LINQ2HASH_SHA224_H_
#include <stddef.h>
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
    void sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                             const unsigned char* hash, gpu_tread_ctx_t* ctx);
#ifdef __cplusplus
}
#endif
#endif
