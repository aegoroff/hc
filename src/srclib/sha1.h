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
#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif
    void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, device_props_t* device_props, const char* dict, size_t dict_len, const char* hash);
    void sha1_run_on_gpu2(gpu_tread_ctx_t* ctx, const char* dict, size_t dict_len, const char* hash, char* variants, const int variants_size);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_SHA1_H_
