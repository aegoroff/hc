/*!
 * \brief   The file contains encoding functions implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#include <stdio.h>
#include <apr.h>
#include "lib.h"
#include "encoding.h"

#ifndef _MSC_VER

#include <stdlib.h>

#define CP_ACP                    0           // default to ANSI code page
#define CP_OEMCP                  1           // default to OEM  code page
#define CP_MACCP                  2           // default to MAC  code page
#define CP_THREAD_ACP             3           // current thread's ANSI code page
#define CP_SYMBOL                 42          // SYMBOL translations

#define CP_UTF7                   65000       // UTF-7 translation
#define CP_UTF8                   65001       // UTF-8 translation
#endif

typedef struct bom_def {
    bom_t bom;
    size_t length;
    unsigned char signature[BOM_MAX_LEN];
} bom_def_t;

static char* prenc_from_unicode_to_code_page(const wchar_t* from, UINT code_page, apr_pool_t* pool);

static const char* enc_bom_names[] = {
        "Unknown", "UTF-8", "UTF-16 (LE)", "UTF-16 (BE)", "UTF-32 (BE)",
};

static bom_def_t boms[] = {
        // Various UTF encodings
        {bom_utf8,    3, {0xEF, 0xBB, 0xBF}},          // UTF8
        {bom_utf16le, 2, {0xFF, 0xFE}},             // UTF16LE
        {bom_utf16be, 2, {0xFE, 0xFF}},             // UTF16BE
        {bom_utf32be, 4, {0x00, 0x00, 0xFE, 0xFF}}, // UTF32BE
        // Add others as desired.  https://en.wikipedia.org/wiki/Byte_order_mark
        {bom_unknown, 0, {0}}
};

const char* enc_get_encoding_name(bom_t bom) {
    if(bom < 0 || bom > bom_utf32be - 1) {
        return NULL;
    }
    return enc_bom_names[bom];
}

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_utf8_to_ansi(const char* from, apr_pool_t* pool) {
#ifdef _MSC_VER
    return enc_decode_utf8_ansi(from, CP_UTF8, CP_ACP, pool);
#else
    return NULL;
#endif
}

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_ansi_to_utf8(const char* from, apr_pool_t* pool) {
#ifdef _MSC_VER
    return enc_decode_utf8_ansi(from, CP_ACP, CP_UTF8, pool);
#else
    return NULL;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
wchar_t* enc_from_ansi_to_unicode(const char* from, apr_pool_t* pool) {
    return enc_from_code_page_to_unicode(from, CP_ACP, pool);
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
wchar_t* enc_from_utf8_to_unicode(const char* from, apr_pool_t* pool) {
    return enc_from_code_page_to_unicode(from, CP_UTF8, pool);
}

wchar_t* enc_from_code_page_to_unicode(const char* from, UINT code_page, apr_pool_t* pool) {
#ifdef _MSC_VER

    /**
     * cbMultiByte
     * Size, in bytes, of the string indicated by the lpMultiByteStr parameter.
     * Alternatively, this parameter can be set to -1 if the string is null-terminated. Note that, if cbMultiByte is 0, the function fails.
     * If this parameter is -1, the function processes the entire input string, including the terminating null character.
     * Therefore, the resulting Unicode string has a terminating null character, and the length returned by the function includes this character.
     * If this parameter is set to a positive integer, the function processes exactly the specified number of bytes.
     * If the provided size does not include a terminating null character, the resulting Unicode string is not null-terminated
     * and the returned length does not include this character.
     */
    const int multi_byte_size = -1;

    const int length_wide = MultiByteToWideChar(code_page, 0, from, multi_byte_size, NULL, 0);
    // including null terminator
    const apr_size_t wide_buffer_size = sizeof(wchar_t) * (apr_size_t) length_wide;
    wchar_t* wide_str = (wchar_t*) apr_pcalloc(pool, wide_buffer_size);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wide_buffer_size, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(code_page, 0, from, multi_byte_size, wide_str, length_wide);
    return wide_str;
#else
    wchar_t* result = NULL;
    size_t length_wide = mbstowcs(NULL, from, 0);
    result = (wchar_t*) apr_pcalloc(pool, (length_wide + 1) * sizeof(wchar_t));
    mbstowcs(result, from, length_wide + 1);
    return result;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* enc_from_unicode_to_ansi(const wchar_t* from, apr_pool_t* pool) {
    return prenc_from_unicode_to_code_page(from, CP_ACP, pool);
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* enc_from_unicode_to_utf8(const wchar_t* from, apr_pool_t* pool) {
    return prenc_from_unicode_to_code_page(from, CP_UTF8, pool);
}

bool enc_is_valid_utf8(const char* str) {
    if(!str) {
        return false;
    }

    const unsigned char* bytes = (const unsigned char*) str;
    unsigned int cp;
    int num;

    while(*bytes != 0x00) {
        if((*bytes & 0x80U) == 0x00) {
            // U+0000 to U+007F
            cp = (*bytes & 0x7FU);
            num = 1;
        } else if((*bytes & 0xE0U) == 0xC0) {
            // U+0080 to U+07FF
            cp = (*bytes & 0x1FU);
            num = 2;
        } else if((*bytes & 0xF0U) == 0xE0) {
            // U+0800 to U+FFFF
            cp = (*bytes & 0x0FU);
            num = 3;
        } else if((*bytes & 0xF8U) == 0xF0) {
            // U+10000 to U+10FFFF
            cp = (*bytes & 0x07U);
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for(int i = 1; i < num; ++i) {
            if((*bytes & 0xC0U) != 0x80) {
                return false;
            }
            cp = (cp << 6U) | (*bytes & 0x3FU);
            bytes += 1;
        }

        if((cp > 0x10FFFF) ||
           ((cp >= 0xD800) && (cp <= 0xDFFF)) ||
           ((cp <= 0x007F) && (num != 1)) ||
           ((cp >= 0x0080) && (cp <= 0x07FF) && (num != 2)) ||
           ((cp >= 0x0800) && (cp <= 0xFFFF) && (num != 3)) ||
           ((cp >= 0x10000) && (cp <= 0x1FFFFF) && (num != 4))) {
            return false;
        }
    }

    return true;
}

bom_t enc_detect_bom(apr_file_t* f) {
    char bom_signature[BOM_MAX_LEN];
    apr_off_t apr_offset = 0;
    apr_status_t status = apr_file_seek(f, APR_SET, &apr_offset); // Only file beginning
    if(status != APR_SUCCESS) {
        return bom_unknown;
    }

    apr_size_t nbytes = BOM_MAX_LEN;
    status = apr_file_read(f, bom_signature, &nbytes);
    if(status != APR_SUCCESS) {
        return bom_unknown;
    }

    size_t offset = 0;
    bom_t result = enc_detect_bom_memory(bom_signature, nbytes, &offset);

    apr_offset = (apr_off_t) offset;
    apr_file_seek(f, APR_SET, &apr_offset); // Leave file position to just after BOM

    return result;
}

bom_t enc_detect_bom_memory(const char* buffer, size_t len, size_t* offset) {
    for(size_t i = 0; boms[i].length; i++) {
        if(len >= boms[i].length && memcmp(buffer, boms[i].signature, boms[i].length) == 0) {
            *offset = boms[i].length;
            return boms[i].bom;
        }
    }
    return bom_unknown;
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* prenc_from_unicode_to_code_page(const wchar_t* from, UINT code_page, apr_pool_t* pool) {
#ifdef _MSC_VER
    char* ansi_str = NULL;

    /**
     * cchWideChar
     * Size, in characters, of the string indicated by lpWideCharStr.
     * Alternatively, this parameter can be set to -1 if the string is null-terminated. If cchWideChar is set to 0, the function fails.
     * If this parameter is -1, the function processes the entire input string, including the terminating null character.
     * Therefore, the resulting character string has a terminating null character, and the length returned by the function includes this character.
     * If this parameter is set to a positive integer, the function processes exactly the specified number of characters.
     * If the provided size does not include a terminating null character, the resulting character
     * string is not null-terminated, and the returned length does not include this character.
     */
    const int wide_size = -1;

    const int length_ansi = WideCharToMultiByte(code_page, 0, from, wide_size, ansi_str, 0, NULL, NULL);
    // null terminator included
    ansi_str = (char*) apr_pcalloc(pool, (apr_size_t) ((apr_size_t) length_ansi));

    if(ansi_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(code_page, 0, from, wide_size, ansi_str, length_ansi, NULL, NULL);

    return ansi_str;
#else
    char* result = NULL;
    size_t length_ansi = wcstombs(NULL, from, 0);
    result = (char*) apr_pcalloc(pool, (length_ansi + 1) * sizeof(char));
    wcstombs(result, from, length_ansi + 1);
    return result;
#endif
}

#ifdef _MSC_VER

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_decode_utf8_ansi(const char* from, UINT from_code_page, UINT to_code_page, apr_pool_t* pool) {
    char* ansi_str = NULL;
    /**
     * cbMultiByte
     * Size, in bytes, of the string indicated by the lpMultiByteStr parameter.
     * Alternatively, this parameter can be set to -1 if the string is null-terminated. Note that, if cbMultiByte is 0, the function fails.
     * If this parameter is -1, the function processes the entire input string, including the terminating null character.
     * Therefore, the resulting Unicode string has a terminating null character, and the length returned by the function includes this character.
     * If this parameter is set to a positive integer, the function processes exactly the specified number of bytes.
     * If the provided size does not include a terminating null character, the resulting Unicode string is not null-terminated
     * and the returned length does not include this character.
     */
    const int multi_byte_size = -1;

    const int length_wide = MultiByteToWideChar(from_code_page, 0, from, multi_byte_size, NULL, 0);

    const apr_size_t wide_buffer_size = sizeof(wchar_t) * (apr_size_t) length_wide;
    wchar_t* wide_str = (wchar_t*) apr_pcalloc(pool, wide_buffer_size);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wide_buffer_size, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(from_code_page, 0, from, multi_byte_size, wide_str, length_wide);

    const int length_ansi = WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansi_str, 0, NULL, NULL);
    // null terminator included
    ansi_str = (char*) apr_pcalloc(pool, (apr_size_t) (length_ansi));

    if(ansi_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansi_str, length_ansi, NULL, NULL);

    return ansi_str;
}

#endif
