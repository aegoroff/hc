/*!
 * \brief   The file contains SHA512 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "sha512def.h"

void SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha512_context context = { 0 };

    SHA512InitContext(&context);
    SHA512UpdateHash(&context, input, inputLen);
    SHA512FinalHash(digest, &context);
}

void SHA512InitContext(void* context)
{
    sph_sha512_init(context);
}

void SHA512FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha512_close(context, digest);
}

void SHA512UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha512(context, input, inputLen);
}

