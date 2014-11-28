/*!
 * \brief   he file contains debugging helpers interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef HC_DEBUGHELPERS_H_
#define HC_DEBUGHELPERS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <windows.h>
#include <Dbghelp.h>

typedef BOOL (WINAPI * MINIDUMPWRITEDUMP)
    (HANDLE,
    DWORD,
    HANDLE,
    MINIDUMP_TYPE,
    PMINIDUMP_EXCEPTION_INFORMATION, PMINIDUMP_USER_STREAM_INFORMATION,
    PMINIDUMP_CALLBACK_INFORMATION);

/*!
 * \brief Application top level exception handler that creates (if it's possible) core dump
 * @param pExceptionInfo pointer to exception information
 */
LONG WINAPI TopLevelFilter(struct _EXCEPTION_POINTERS* pExceptionInfo);

#ifdef __cplusplus
}
#endif
#endif // HC_DEBUGHELPERS_H_
