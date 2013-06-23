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

apr_status_t WHIRLPOOLCalculateDigest(apr_byte_t*      digest,
                                      const void*      input,
                                      const apr_size_t inputLen)
{
    sph_whirlpool_context context = { 0 };

    WHIRLPOOLInitContext(&context);
    WHIRLPOOLUpdateHash(&context, input, inputLen);
    WHIRLPOOLFinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLInitContext(void* context)
{
    sph_whirlpool_init(context);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLFinalHash(apr_byte_t* digest, void* context)
{
    sph_whirlpool_close(context, digest);
    return APR_SUCCESS;
}

apr_status_t WHIRLPOOLUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_whirlpool(context, input, inputLen);
    return APR_SUCCESS;
}
