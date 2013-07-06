/*!
 * \brief   The file contains hashes from libtom lib API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef HASHES_HCALC_H_
#define HASHES_HCALC_H_

#include "apr.h"
#include "apr_errno.h"
#include "apr_pools.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
   Hash sizes:
 */

#define SZ_GOST         32
#define SZ_WHIRLPOOL    64
#define SZ_SHA512       64
#define SZ_SHA384       48
#define SZ_RIPEMD320    40
#define SZ_SHA256       32
#define SZ_RIPEMD256    32
#define SZ_SHA224       28
#define SZ_TIGER192     24
#define SZ_SHA1         20
#define SZ_RIPEMD160    20
#define SZ_RIPEMD128    16
#define SZ_MD5          16
#define SZ_MD4          16
#define SZ_MD2          16
#define SZ_SNEFRU128    16
#define SZ_SNEFRU256    32
#define SZ_TTH          24

typedef struct HashDefinition {
    size_t ContextSize;
    apr_size_t  HashLength;
    int         Weight;
    void (*PfnDigest)(apr_byte_t * digest, const void* input,
                                         const apr_size_t inputLen);
    void (*PfnInit)(void* context);
    void (*PfnFinal)(void* context, apr_byte_t * digest);
    void (*PfnUpdate)(void* context, const void* input,
                                         const apr_size_t inputLen);
} HashDefinition;

HashDefinition* GetHash(const char* attr);
void InitializeHashes(apr_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif // HASHES_HCALC_H_
