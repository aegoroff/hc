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

#include "md4.h"

apr_status_t MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return apr_md4(digest, input, inputLen);
}

apr_status_t MD4InitContext(void* context)
{
    return apr_md4_init((apr_md4_ctx_t*)context);
}

apr_status_t MD4FinalHash(apr_byte_t* digest, void* context)
{
    return apr_md4_final(digest, (apr_md4_ctx_t*)context);
}

apr_status_t MD4UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return apr_md4_update((apr_md4_ctx_t*)context, input, inputLen);
}
