/*!
 * \brief   The file contains encoding functions implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include <stdio.h>
#include <apr.h>
#include "lib.h"
#include "encoding.h"

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_utf8_to_ansi(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    return enc_decode_utf8_ansi(from, CP_UTF8, CP_ACP, pool);
#else
    return NULL;
#endif
}

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_ansi_to_utf8(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    return enc_decode_utf8_ansi(from, CP_ACP, CP_UTF8, pool);
#else
    return NULL;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
wchar_t* enc_from_ansi_to_unicode(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    int length_wide = 0;
    size_t cbFrom = 0;
    wchar_t* wide_str = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1; // IMPORTANT!!! including null terminator

    length_wide = MultiByteToWideChar(CP_ACP, 0, from, (int)cbFrom, NULL, 0); // including null terminator
    wideBufferSize = sizeof(wchar_t) * (apr_size_t)length_wide;
    wide_str = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(CP_ACP, 0, from, (int)cbFrom, wide_str, length_wide);
    return wide_str;
#else
    return NULL;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* enc_from_unicode_to_ansi(const wchar_t* from, apr_pool_t* pool) {
#ifdef WIN32
    int length_ansi = 0;
    char* ansiStr = NULL;

    length_ansi = WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansiStr, 0, NULL, NULL); // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, (apr_size_t)(length_ansi + 1));

    if(ansiStr == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansiStr, length_ansi, NULL, NULL);

    return ansiStr;
#else
    return NULL;
#endif
}


#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_decode_utf8_ansi(const char* from, UINT from_code_page, UINT to_code_page, apr_pool_t* pool) {
    int length_wide = 0;
    int length_ansi = 0;
    size_t cbFrom = 0;
    wchar_t* wide_str = NULL;
    char* ansiStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1; // IMPORTANT!!! including null terminator

    length_wide = MultiByteToWideChar(from_code_page, 0, from, (int)cbFrom, NULL, 0); // including null terminator
    wideBufferSize = sizeof(wchar_t) * (apr_size_t)length_wide;
    wide_str = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(from_code_page, 0, from, (int)cbFrom, wide_str, length_wide);

    length_ansi = WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansiStr, 0, NULL, NULL); // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, (apr_size_t)(length_ansi));

    if(ansiStr == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansiStr, length_ansi, NULL, NULL);

    return ansiStr;
}
#endif
