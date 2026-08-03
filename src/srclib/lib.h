/*!
 * \brief   The file contains common solution library interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#ifndef LINQ2HASH_LIB_H_
#define LINQ2HASH_LIB_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BYTE_CHARS_SIZE
#define BYTE_CHARS_SIZE 2   // byte representation string length
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

extern void lib_hex_str_2_byte_array(const char* str, uint8_t* bytes, size_t sz);

extern uint32_t lib_htoi(const char* ptr, int size);

#ifdef __cplusplus
}
#endif
#endif // LINQ2HASH_LIB_H_
