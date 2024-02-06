/*!
 * \brief   The file contains GPU Ripemd-160 related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-11-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#ifndef LINQ2HASH_RMD160_H_
#define LINQ2HASH_RMD160_H_

#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif
    void rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
                           const size_t variants_size);
    void rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                               const unsigned char* hash, gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_RMD160_H_
