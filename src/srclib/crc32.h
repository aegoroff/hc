/*!
 * \brief   The file contains CRC32 library declarations
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef LINQ2HASH_CRC32_H_
#define LINQ2HASH_CRC32_H_

#include "..\srclib\types.h"

#define CRC32_HASH_SIZE 4 // hash size in bytes

typedef struct crc32_context_t {
    uint32_t crc;
} crc32_context_t;

#ifdef __cplusplus
extern "C" {
#endif

void crc32_init(crc32_context_t* ctx);
void crc32_update(crc32_context_t* ctx, const void* data, size_t len);
void crc32_final(crc32_context_t* ctx, uint8_t* hash);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_CRC32_H_
