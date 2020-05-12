/*!
 * \brief   The file contains encoding functions interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#ifndef PCTRL_ENCODING_H_
#define PCTRL_ENCODING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "apr_pools.h"
#include "apr_file_io.h"

typedef enum {
    bom_unknown = 0,
    bom_utf8 = 1,
    bom_utf16le = 2,
    bom_utf16be = 3,
    bom_utf32be = 4
} bom_t;

#define BOM_MAX_LEN 5

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_utf8_to_ansi(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_ansi_to_utf8(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
wchar_t* enc_from_ansi_to_unicode(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_unicode_to_ansi(const wchar_t* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_unicode_to_utf8(const wchar_t* from, apr_pool_t* pool);

BOOL enc_is_valid_utf8(const char* str);

bom_t enc_detect_bom(apr_file_t* f);
bom_t enc_detect_bom_memory(const unsigned char* buffer, size_t len, size_t* offset);

const char* enc_get_encoding_name(bom_t bom);

#ifdef WIN32

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_decode_utf8_ansi(const char* from, UINT from_code_page, UINT to_code_page, apr_pool_t* pool);

#endif

#ifdef __cplusplus
}
#endif

#endif // PCTRL_ENCODING_H_
