/*!
 * \brief   The file contains CRC32 library declarations
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#ifndef LINQ2HASH_CRC32_H_
#define LINQ2HASH_CRC32_H_

#include "types.h"

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

// CRC32C (Castagnoli): x86/x86_64 (SSE4.2 HW or software) and aarch64
// (ARMv8 CRC32 HW when compiled with +crc, else software table).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || \
    defined(__aarch64__) || defined(_M_ARM64)
#define HC_HAVE_CRC32C 1
#if (defined(__SSE4_2__) || defined(__CRC32__) || defined(__ARM_FEATURE_CRC32))
#define HC_CRC32C_HW 1
#else
#define HC_CRC32C_HW 0
#endif
void crc32c_init(crc32_context_t* ctx);
void crc32c_update(crc32_context_t* ctx, const void* data, size_t len);
void crc32c_final(crc32_context_t* ctx, uint8_t* hash);
#else
#define HC_HAVE_CRC32C 0
#define HC_CRC32C_HW 0
#endif

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_CRC32_H_
