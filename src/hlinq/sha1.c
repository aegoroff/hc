/*!
 * \brief   The file contains SHA1 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#include "sha1.h"

apr_status_t SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    apr_sha1_ctx_t context = { 0 };
    
    SHA1InitContext(&context);
    SHA1UpdateHash(&context, input, inputLen);
    SHA1FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t SHA1InitContext(void* context)
{
    apr_sha1_init((apr_sha1_ctx_t*)context);
    return APR_SUCCESS;
}

apr_status_t SHA1FinalHash(apr_byte_t* digest, void* context)
{
    apr_sha1_final(digest, (apr_sha1_ctx_t*)context);
    return APR_SUCCESS;
}

apr_status_t SHA1UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    apr_sha1_update((apr_sha1_ctx_t*)context, input, inputLen);
    return APR_SUCCESS;
}
