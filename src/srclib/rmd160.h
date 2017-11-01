/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains GPU Ripemd-160 related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-11-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
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
                               const unsigned char* hash, unsigned char** variants, size_t variants_len);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_RMD160_H_
