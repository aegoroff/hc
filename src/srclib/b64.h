/*!
 * \brief   The file contains base64 encode/decode interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-10-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#ifndef LINQ2HASH_B64_H_
#define LINQ2HASH_B64_H_

#include "apr_pools.h"

#ifdef __cplusplus
extern "C" {
#endif

char* b64_encode(const unsigned char* data,
                 size_t input_length,
                 size_t* output_length,
                 apr_pool_t* pool);

unsigned char* b64_decode(const char* data,
                          size_t input_length,
                          size_t* output_length,
                          apr_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_B64_H_
