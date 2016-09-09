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
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    return DecodeUtf8Ansi(from, CP_UTF8, CP_ACP, pool);
#else
    return NULL;
#endif
}

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromAnsiToUtf8(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    return DecodeUtf8Ansi(from, CP_ACP, CP_UTF8, pool);
#else
    return NULL;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
wchar_t* FromAnsiToUnicode(const char* from, apr_pool_t* pool) {
#ifdef WIN32
    int lengthWide = 0;
    size_t cbFrom = 0;
    wchar_t* wideStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1; // IMPORTANT!!! including null terminator

    lengthWide = MultiByteToWideChar(CP_ACP, 0, from, (int)cbFrom, NULL, 0); // including null terminator
    wideBufferSize = sizeof(wchar_t) * (apr_size_t)lengthWide;
    wideStr = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if(wideStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(CP_ACP, 0, from, (int)cbFrom, wideStr, lengthWide);
    return wideStr;
#else
    return NULL;
#endif
}

/*!
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* FromUnicodeToAnsi(const wchar_t* from, apr_pool_t* pool) {
#ifdef WIN32
    int lengthAnsi = 0;
    char* ansiStr = NULL;

    lengthAnsi = WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansiStr, 0, NULL, NULL); // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, (apr_size_t)(lengthAnsi + 1));

    if(ansiStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, lengthAnsi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(CP_ACP, 0, from, wcslen(from), ansiStr, lengthAnsi, NULL, NULL);

    return ansiStr;
#else
    return NULL;
#endif
}


#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, UINT fromCodePage, UINT toCodePage, apr_pool_t* pool) {
    int lengthWide = 0;
    int lengthAnsi = 0;
    size_t cbFrom = 0;
    wchar_t* wideStr = NULL;
    char* ansiStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1; // IMPORTANT!!! including null terminator

    lengthWide = MultiByteToWideChar(fromCodePage, 0, from, (int)cbFrom, NULL, 0); // including null terminator
    wideBufferSize = sizeof(wchar_t) * (apr_size_t)lengthWide;
    wideStr = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if(wideStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(fromCodePage, 0, from, (int)cbFrom, wideStr, lengthWide);

    lengthAnsi = WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, 0, NULL, NULL); // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, (apr_size_t)(lengthAnsi));

    if(ansiStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, lengthAnsi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, lengthAnsi, NULL, NULL);

    return ansiStr;
}
#endif
