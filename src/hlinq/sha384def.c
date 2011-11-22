/*!
 * \brief   The file contains SHA384 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#include "sha384def.h"

apr_status_t SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    SHA384Context context = { 0 };
    
    SHA384InitContext(&context);
    SHA384UpdateHash(&context, input, inputLen);
    SHA384FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t SHA384InitContext(void* context)
{
    SHA384Init((SHA384Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA384FinalHash(apr_byte_t* digest, void* context)
{
    SHA384Final(digest, (SHA384Context*)context);
    return APR_SUCCESS;
}

apr_status_t SHA384UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    SHA384Update((SHA384Context*)context, input, inputLen);
    return APR_SUCCESS;
}
