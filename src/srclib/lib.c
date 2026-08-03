/*!
 * \brief   The file contains common solution library implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <string.h>

#include "lib.h"

/*
   lib_ - public members
   Size/time/timer helpers live in Zig (`src/hc/lib.zig`); this C unit keeps
   only hex parse for bf_shim.
*/

uint32_t lib_htoi(const char *ptr, int size) {
    uint32_t value = 0;
    while (size-- > 0 && ptr != NULL) {
        if (*ptr >= '0' && *ptr <= '9') {
            value = (value << 4U) + (*ptr - '0');
        } else if (*ptr >= 'A' && *ptr <= 'F') {
            value = (value << 4U) + ((*ptr - 'A') + 10);
        } else if (*ptr >= 'a' && *ptr <= 'f') {
            value = (value << 4U) + ((*ptr - 'a') + 10);
        } else if (value > 0) {
            return value;
        }
        ++ptr;
    }
    return value;
}

void lib_hex_str_2_byte_array(const char* str, uint8_t* bytes, size_t sz) {
    size_t i = 0;
    const size_t to = MIN(sz, strlen(str) / BYTE_CHARS_SIZE);

    for(; i < to; i++) {
        bytes[i] = (uint8_t) lib_htoi(str + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}
