/*!
 * \brief   The file contains WHIRLPOOL API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "whirl.h"

apr_status_t WHIRLPOOLCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    WHIRLPOOL_CTX context = { 0 };
    
    WHIRLPOOLInitContext(&context);
    WHIRLPOOLUpdateHash(&context, input, inputLen);
    WHIRLPOOLFinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLInitContext(void* context)
{
    WHIRLPOOL_Init((WHIRLPOOL_CTX*)context);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLFinalHash(apr_byte_t* digest, void* context)
{
    WHIRLPOOL_Final(digest, (WHIRLPOOL_CTX*)context);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    WHIRLPOOL_Update((WHIRLPOOL_CTX*)context, input, inputLen);
    return APR_SUCCESS;
}
