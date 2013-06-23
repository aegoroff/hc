/*!
 * \brief   The file contains MD5 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "implementation.h"
#include "apr.h"
#include "apr_errno.h"

apr_status_t InitContext(hash_context_t* context)
{
    sph_md5_init(context);
    return APR_SUCCESS;
}

apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context)
{
    sph_md5_close(context, digest);
    return APR_SUCCESS;
}

apr_status_t UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen)
{
    sph_md5(context, input, inputLen);
    return APR_SUCCESS;
}
