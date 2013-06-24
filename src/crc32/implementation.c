/*!
 * \brief   The file contains CRC32 calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "targetver.h"
#include "implementation.h"

void InitContext(hash_context_t* context)
{
    Crc32Init(context);
}

void FinalHash(apr_byte_t* digest, hash_context_t* context)
{
    Crc32Final(digest, context);
}

void UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen)
{
    Crc32Update(context, input, (uint32_t)inputLen);
}
