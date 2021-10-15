/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains GPU MD4 related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-09-28
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#ifndef LINQ2HASH_MD4_H_
#define LINQ2HASH_MD4_H_

#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif
    void md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
                           const size_t variants_size);
    void md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                               const unsigned char* hash, gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_MD4_H_
