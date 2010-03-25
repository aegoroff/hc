#ifndef MD5_DEBUGHELPERS_H_
#define MD5_DEBUGHELPERS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <windows.h>
#include <Dbghelp.h>

    typedef BOOL(WINAPI * MINIDUMPWRITEDUMP)
     (HANDLE,
      DWORD,
      HANDLE,
      MINIDUMP_TYPE,
      PMINIDUMP_EXCEPTION_INFORMATION, PMINIDUMP_USER_STREAM_INFORMATION, PMINIDUMP_CALLBACK_INFORMATION);

    /*!
     * \brief Application top level exception handler that creates (if it's possible) core dump
     * @param pExceptionInfo pointer to exception information
     */
    LONG WINAPI TopLevelFilter(struct _EXCEPTION_POINTERS *pExceptionInfo);

#ifdef __cplusplus
}
#endif
#endif // MD5_DEBUGHELPERS_H_
