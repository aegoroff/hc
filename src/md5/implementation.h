/*!
 * \brief   The file contains MD5 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef MD5_IMPLEMENTATION_H_
#define MD5_IMPLEMENTATION_H_

#include "apr_md5.h"

typedef apr_md5_ctx_t hash_context_t;

#define DIGESTSIZE APR_MD5_DIGESTSIZE
#define APP_NAME "MD5 Calculator " PRODUCT_VERSION
#define HASH_NAME "MD5"
#define OPT_HASH_LONG "md5"

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen);
apr_status_t InitContext(hash_context_t* context);
apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context);
apr_status_t UpdateHash(hash_context_t* context, const void* input, apr_size_t inputLen);

#endif // MD5_IMPLEMENTATION_H_
