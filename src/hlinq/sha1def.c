/*!
 * \brief   The file contains SHA1 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "sha1def.h"

apr_status_t SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha1_context context = { 0 };

    SHA1InitContext(&context);
    SHA1UpdateHash(&context, input, inputLen);
    SHA1FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t SHA1InitContext(void* context)
{
    sph_sha1_init(context);
    return APR_SUCCESS;
}

apr_status_t SHA1FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha1_close(context, digest);
    return APR_SUCCESS;
}

apr_status_t SHA1UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha1(context, input, inputLen);
    return APR_SUCCESS;
}
