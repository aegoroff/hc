/*!
 * \brief   The file contains SHA1 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "implementation.h"
#include "apr.h"
#include "apr_errno.h"

apr_status_t InitContext(hash_context_t* context)
{
    sph_sha1_init(context);
    return APR_SUCCESS;
}

apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context)
{
    sph_sha1_close(context, digest);
    return APR_SUCCESS;
}

apr_status_t UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen)
{
    sph_sha1(context, input, inputLen);
    return APR_SUCCESS;
}