/*!
 * \brief   The file contains SHA256 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "sha256def.h"

apr_status_t SHA256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    SHA256Context context = { 0 };
    
    SHA256InitContext(&context);
    SHA256UpdateHash(&context, input, inputLen);
    SHA256FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t SHA256InitContext(void* context)
{
    SHA256Init((SHA256Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA256FinalHash(apr_byte_t* digest, void* context)
{
    SHA256Final(digest, (SHA256Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA256UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    SHA256Update((SHA256Context*)context, input, inputLen);
    return APR_SUCCESS;
}
