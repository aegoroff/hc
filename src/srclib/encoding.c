/*!
 * \brief   The file contains encoding functions implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#include <stdio.h>
#include "lib.h"
#include "encoding.h"

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool)
{
#ifdef WIN32
    return DecodeUtf8Ansi(from, CP_UTF8, CP_ACP, pool);
#else
    return NULL;
#endif
}

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, UINT fromCodePage, UINT toCodePage, apr_pool_t* pool)
{
    int lengthWide = 0;
    int lengthAnsi = 0;
    size_t cbFrom = 0;
    wchar_t* wideStr = NULL;
    char* ansiStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1;  // IMPORTANT!!! including null terminator

    lengthWide = MultiByteToWideChar(fromCodePage, 0, from, cbFrom, NULL, 0);   // including null terminator
    wideBufferSize = sizeof(wchar_t) * lengthWide;
    wideStr = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if (wideStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(fromCodePage, 0, from, cbFrom, wideStr, lengthWide);

    lengthAnsi = WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, 0, NULL, NULL);   // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, lengthAnsi);

    if (ansiStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, lengthAnsi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, lengthAnsi, NULL, NULL);

    return ansiStr;
}
#endif
