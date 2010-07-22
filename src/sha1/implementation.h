/*!
 * \brief   The file contains SHA1 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef SHA1_IMPLEMENTATION_H_
#define SHA1_IMPLEMENTATION_H_

#include "apr_sha1.h"

typedef apr_sha1_ctx_t hash_context_t;

#define DIGESTSIZE APR_SHA1_DIGESTSIZE
#define APP_NAME "SHA1 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA1"
#define OPT_HASH_LONG "sha1"

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen);
apr_status_t InitContext(hash_context_t* context);
apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context);
apr_status_t UpdateHash(hash_context_t* context, const void* input, apr_size_t inputLen);

#endif // SHA1_IMPLEMENTATION_H_
