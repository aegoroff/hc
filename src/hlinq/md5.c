/*!
 * \brief   The file contains MD5 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#include "md5.h"

apr_status_t MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return apr_md5(digest, input, inputLen);
}

apr_status_t MD5InitContext(void* context)
{
    return apr_md5_init((apr_md5_ctx_t*)context);
}

apr_status_t MD5FinalHash(apr_byte_t* digest, void* context)
{
    return apr_md5_final(digest, (apr_md5_ctx_t*)context);
}

apr_status_t MD5UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return apr_md5_update((apr_md5_ctx_t*)context, input, inputLen);
}
