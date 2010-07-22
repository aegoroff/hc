/*!
 * \brief   The file contains MD5 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#include "targetver.h"
#include "implementation.h"

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen)
{
    return apr_md5(digest, input, inputLen);
}

apr_status_t InitContext(hash_context_t* context)
{
    return apr_md5_init(context);
}

apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context)
{
    return apr_md5_final(digest, context);
}

apr_status_t UpdateHash(hash_context_t* context, const void* input, apr_size_t inputLen)
{
    return apr_md5_update(context, input, inputLen);
}
