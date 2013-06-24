/*!
 * \brief   The file contains CRC32 API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "crc32def.h"

void CRC32CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    Crc32Context context = { 0 };
    
    CRC32InitContext(&context);
    CRC32UpdateHash(&context, input, inputLen);
    CRC32FinalHash(digest, &context);
}

void CRC32InitContext(void* context)
{
    Crc32Init((Crc32Context*)context);
}

void CRC32FinalHash(apr_byte_t* digest, void* context)
{
    Crc32Final(digest, (Crc32Context*)context);
}

void CRC32UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    Crc32Update((Crc32Context*)context, input, inputLen);
}
