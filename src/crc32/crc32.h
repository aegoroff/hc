/*!
 * \brief   The file contains CRC32 library declarations
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef _CRC32_H
#define _CRC32_H

#include "..\srclib\types.h"

#define CRC32_HASH_SIZE 4 // hash size in bytes

typedef struct Crc32Context {
    uint32_t crc;
} Crc32Context;

#ifdef __cplusplus
extern "C" {
#endif

void Crc32Init(Crc32Context* ctx);
void Crc32Update(Crc32Context* ctx, const void* data, uint32_t len);
void Crc32Final(uint8_t* hash, Crc32Context* ctx);

#ifdef __cplusplus
}
#endif

#endif /* !_CRC32_H */
