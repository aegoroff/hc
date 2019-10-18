/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains encoding functions implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include <stdio.h>
#include <apr.h>
#include "lib.h"
#include "encoding.h"

#ifndef WIN32
#include <stdlib.h>
#endif

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
    const size_t cb_from = strlen(from) + 1;
    // IMPORTANT!!! including null terminator
    const int length_wide = MultiByteToWideChar(CP_ACP, 0, from, (int)cb_from, NULL, 0);
    // including null terminator
    const apr_size_t wide_buffer_size = sizeof(wchar_t) * (apr_size_t)length_wide;
    wchar_t* wide_str                 = (wchar_t*)apr_pcalloc(pool, wide_buffer_size);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wide_buffer_size, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(CP_ACP, 0, from, (int)cb_from, wide_str, length_wide);
    return wide_str;
#else
    wchar_t* result = NULL;
    size_t length_wide = mbstowcs(NULL, from, 0);
    result = (wchar_t*)apr_pcalloc(pool, length_wide + 1, sizeof(wchar_t));
    mbstowcs(result, from, length_wide + 1);
    return result;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* enc_from_unicode_to_ansi(const wchar_t* from, apr_pool_t* pool) {
#ifdef WIN32
    char* ansi_str        = NULL;
    const int length_ansi = WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansi_str, 0, NULL, NULL);
    // null terminator included
    ansi_str = (char*)apr_pcalloc(pool, (apr_size_t)((apr_size_t)length_ansi + 1));

    if(ansi_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansi_str, length_ansi, NULL, NULL);

    return ansi_str;
#else
    char* result = NULL;
    size_t length_ansi = wcstombs(NULL, from, 0);
    result = (char*)apr_pcalloc(pool, length_ansi + 1, sizeof(char));
    wcstombs(result, from, length_ansi + 1);
    return result;
#endif
}

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_decode_utf8_ansi(const char* from, UINT from_code_page, UINT to_code_page, apr_pool_t* pool) {
    char* ansi_str        = NULL;
    const size_t cb_from  = strlen(from) + 1; // IMPORTANT!!! including null terminator
    const int length_wide = MultiByteToWideChar(from_code_page, 0, from, (int)cb_from, NULL, 0);
    // including null terminator
    const apr_size_t wide_buffer_size = sizeof(wchar_t) * (apr_size_t)length_wide;
    wchar_t* wide_str                 = (wchar_t*)apr_pcalloc(pool, wide_buffer_size);
    if(wide_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, wide_buffer_size, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(from_code_page, 0, from, (int)cb_from, wide_str, length_wide);

    const int length_ansi = WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansi_str, 0, NULL, NULL);
    // null terminator included
    ansi_str = (char*)apr_pcalloc(pool, (apr_size_t)(length_ansi));

    if(ansi_str == NULL) {
        lib_printf(ALLOCATION_FAILURE_MESSAGE, length_ansi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(to_code_page, 0, wide_str, length_wide, ansi_str, length_ansi, NULL, NULL);

    return ansi_str;
}
#endif
