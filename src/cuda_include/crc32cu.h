#ifndef LINQ2HASH_CRC32CU_H_
#define LINQ2HASH_CRC32CU_H_
#include <stddef.h>
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
    void crc32_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void crc32_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                             const unsigned char* hash, gpu_tread_ctx_t* ctx);
#ifdef __cplusplus
}
#endif
#endif
