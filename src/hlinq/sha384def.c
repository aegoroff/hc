/*!
 * \brief   The file contains SHA384 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "sha384def.h"

void SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha384_context context = { 0 };

    SHA384InitContext(&context);
    SHA384UpdateHash(&context, input, inputLen);
    SHA384FinalHash(digest, &context);
}

void SHA384InitContext(void* context)
{
    sph_sha384_init(context);
}

void SHA384FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha384_close(context, digest);
}

void SHA384UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha384(context, input, inputLen);
}
