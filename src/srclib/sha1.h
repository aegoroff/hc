/*!
 * \brief   The file contains GPU SHA1 related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_SHA1_H_
#define LINQ2HASH_SHA1_H_

#include "bf.h"
#include <crt/host_defines.h>

#define DIGESTSIZE 20

#ifdef __cplusplus
extern "C" {
#endif
    __host__ void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, size_t dict_len, unsigned char* variants, const size_t variants_size);
    __host__ void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, char** variants, size_t variants_len);
    __host__ void sha1_on_gpu_cleanup(gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_SHA1_H_
