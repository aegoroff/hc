/*!
 * \brief   The file contains SHA384 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-08-24
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "implementation.h"

void InitContext(hash_context_t* context)
{
    sph_sha384_init(context);
}

void FinalHash(hash_context_t* context, apr_byte_t* digest)
{
    sph_sha384_close(context, digest);
}

void UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen)
{
    sph_sha384(context, input, inputLen);
}
