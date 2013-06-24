/*!
 * \brief   The file contains MD5 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "md5def.h"

void MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md5_context context = { 0 };
    MD5InitContext(&context);
    MD5UpdateHash(&context, input, inputLen);
    MD5FinalHash(digest, &context);
}

void MD5InitContext(void* context)
{
    sph_md5_init(context);
}

void MD5FinalHash(apr_byte_t* digest, void* context)
{
    sph_md5_close(context, digest);
}

void MD5UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_md5(context, input, inputLen);
}
