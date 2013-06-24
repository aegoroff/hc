/*!
 * \brief   The file contains MD4 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "implementation.h"
#include "apr.h"
#include "apr_errno.h"


void InitContext(hash_context_t* context)
{
    sph_md4_init(context);
}

void FinalHash(apr_byte_t* digest, hash_context_t* context)
{
    sph_md4_close(context, digest);
}

void UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen)
{
    sph_md4(context, input, inputLen);
}
