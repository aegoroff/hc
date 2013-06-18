/*!
 * \brief   The file contains MD2 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include    <tomcrypt.h>
#include "md2.h"


apr_status_t MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    hash_state context = { 0 };
    
    MD2InitContext(&context);
    MD2UpdateHash(&context, input, inputLen);
    MD2FinalHash(digest, &context);
    return APR_SUCCESS;
}

apr_status_t MD2InitContext(void* context)
{
    md2_init((hash_state*)context);
    return APR_SUCCESS;
}

apr_status_t MD2FinalHash(apr_byte_t* digest, void* context)
{
    md2_done((hash_state*)context, digest);
    return APR_SUCCESS;
}

apr_status_t MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    md2_process((hash_state*)context, input, inputLen);
    return APR_SUCCESS;
}
