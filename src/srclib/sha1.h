/*!
 * \brief   The file contains GPU SHA1 related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#ifndef LINQ2HASH_SHA1_H_
#define LINQ2HASH_SHA1_H_

#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif
    void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
                         const size_t variants_size);
    void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                             const unsigned char* hash, gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_SHA1_H_
