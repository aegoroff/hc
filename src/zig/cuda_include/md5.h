#ifndef LINQ2HASH_MD5_H_
#define LINQ2HASH_MD5_H_
#include <stddef.h>
#include "hashes.h"
#ifdef __cplusplus
extern "C" {
#endif
    void md5_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
                         const size_t variants_size);
    void md5_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                             const unsigned char* hash, gpu_tread_ctx_t* ctx);
#ifdef __cplusplus
}
#endif
#endif
