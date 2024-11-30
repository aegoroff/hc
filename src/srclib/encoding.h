/*!
 * \brief   The file contains encoding functions interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#ifndef LINQ2HASH_ENCODING_H_
#define LINQ2HASH_ENCODING_H_

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#ifdef __linux__
#include <stddef.h>
#endif

#include "apr_file_io.h"
#include "apr_pools.h"

typedef enum { bom_unknown = 0, bom_utf8 = 1, bom_utf16le = 2, bom_utf16be = 3, bom_utf32be = 4 } bom_t;

#define BOM_MAX_LEN 5

#ifndef _MSC_VER

#ifndef _UINT
#define _UINT
typedef unsigned int UINT;
#endif

#include <uchar.h>

#endif

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char *enc_from_utf8_to_ansi(const char *from, apr_pool_t *pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char *enc_from_ansi_to_utf8(const char *from, apr_pool_t *pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
wchar_t *enc_from_ansi_to_unicode(const char *from, apr_pool_t *pool);

#ifndef _MSC_VER
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char16_t *enc_from_ansi_to_wide_chars(const char *from, apr_pool_t *pool);
#endif

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
wchar_t *enc_from_utf8_to_unicode(const char *from, apr_pool_t *pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char *enc_from_unicode_to_ansi(const wchar_t *from, apr_pool_t *pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char *enc_from_unicode_to_utf8(const wchar_t *from, apr_pool_t *pool);

bool enc_is_valid_utf8(const char *str);

bom_t enc_detect_bom(apr_file_t *f);

bom_t enc_detect_bom_memory(const char *buffer, size_t len, size_t *offset);

const char *enc_get_encoding_name(bom_t bom);

#ifdef _MSC_VER

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char *enc_decode_utf8_ansi(const char *from, UINT from_code_page, UINT to_code_page, apr_pool_t *pool);

#endif

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
wchar_t *enc_from_code_page_to_unicode(const char *from, UINT code_page, apr_pool_t *pool);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_ENCODING_H_
