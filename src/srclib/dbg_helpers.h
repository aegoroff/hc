// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   he file contains debugging helpers interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_DEBUGHELPERS_H_
#define LINQ2HASH_DEBUGHELPERS_H_

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
 * @param p_exception_info pointer to exception information
 */
LONG WINAPI dbg_top_level_filter(struct _EXCEPTION_POINTERS* p_exception_info);

#ifdef __cplusplus
}
#endif
#endif // LINQ2HASH_DEBUGHELPERS_H_
