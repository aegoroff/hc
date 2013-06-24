/*!
 * \brief   The file contains MD4 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "md4def.h"

void MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md4_context context = { 0 };

    MD4InitContext(&context);
    MD4UpdateHash(&context, input, inputLen);
    MD4FinalHash(digest, &context);
}


void MD4InitContext(void* context)
{
    sph_md4_init(context);
}

void MD4FinalHash(apr_byte_t* digest, void* context)
{
    sph_md4_close(context, digest);
}

void MD4UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_md4(context, input, inputLen);
}
