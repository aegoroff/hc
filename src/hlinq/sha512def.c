/*!
 * \brief   The file contains SHA512 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#include "sha512def.h"

apr_status_t SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    SHA512Context context = { 0 };
    
    SHA512InitContext(&context);
    SHA512UpdateHash(&context, input, inputLen);
    SHA512FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t SHA512InitContext(void* context)
{
    SHA512Init((SHA512Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA512FinalHash(apr_byte_t* digest, void* context)
{
    SHA512Final(digest, (SHA512Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA512UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    SHA512Update((SHA512Context*)context, input, inputLen);
    return APR_SUCCESS;
}
