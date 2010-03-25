#include "targetver.h"
#include <assert.h>
#include <stdio.h>
#include <shlwapi.h>
#include "DebugHelplers.h"

#define DBG_HELP_DLL L"DBGHELP.DLL"
#define DUMP_FILE_EXT L"dmp"

void LogDumpFileError(const wchar_t * pMsg, const wchar_t * pDumpFileName);

/*!
* Caller must free memory allocated in the function using VirtualFree
*/
wchar_t *ConcatenateTwoStrings(const wchar_t * pTmpBuffer, const wchar_t * pSecond);

/*!
* Caller must free memory allocated in the function using VirtualFree
*/
wchar_t *ConcatenateThreeStrings(const wchar_t * pTmpBuffer, const wchar_t * pSecond, const wchar_t * pThird);

void PrintWin32Error(const wchar_t * message);

LONG WINAPI TopLevelFilter(struct _EXCEPTION_POINTERS *pExceptionInfo)
{
    LONG result = EXCEPTION_CONTINUE_SEARCH;    // finalize process in standart way by default
    HMODULE hDll = NULL;
    wchar_t szFullAppPath[_MAX_PATH + 1 /* Trailing zero */ ];
    wchar_t *pTmpBuffer = NULL;
    wchar_t *pDumpFile = NULL;
    MINIDUMP_EXCEPTION_INFORMATION exInfo;
    BOOL isOK = FALSE;
    MINIDUMPWRITEDUMP pfnDump;
    HANDLE hFile;

    if (!GetModuleFileNameW(NULL, szFullAppPath, _MAX_PATH + 1 /* Trailing zero */ )) {
        goto cleanup;   // cannot define executable module
    }

    pDumpFile = ConcatenateThreeStrings(PathFindFileNameW(szFullAppPath), L".", DUMP_FILE_EXT);
    if (pDumpFile == NULL) {
        goto cleanup;
    }

    hDll = LoadLibraryW(DBG_HELP_DLL);

    if (hDll == NULL) {
        pTmpBuffer = ConcatenateTwoStrings(L" Cannot load dll ", DBG_HELP_DLL);
        PrintWin32Error(pTmpBuffer);
        goto cleanup;
    }
    // get func address
    pfnDump = (MINIDUMPWRITEDUMP) GetProcAddress(hDll, "MiniDumpWriteDump");
    if (!pfnDump) {
        PrintWin32Error(L" Cannot get address of MiniDumpWriteDump function");
        goto cleanup;
    }

    hFile = CreateFileW(pDumpFile, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        LogDumpFileError(L" An unhandled exception occured. Error on creating dump file: ", pDumpFile);
        goto cleanup;
    }

    exInfo.ThreadId = GetCurrentThreadId();
    exInfo.ExceptionPointers = pExceptionInfo;
    exInfo.ClientPointers = 0;

    // Write pDumpFile
    isOK = pfnDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &exInfo, NULL, NULL);
    if (isOK) {
        wprintf_s(L" An unhandled exception occured. Dump saved to: %s", pDumpFile);
        result = EXCEPTION_EXECUTE_HANDLER;
    } else {
        LogDumpFileError(L" An unhandled exception occured. Error saving dump file: ", pDumpFile);
    }
    CloseHandle(hFile);

cleanup:
    if (pTmpBuffer != NULL) {
        VirtualFree(pTmpBuffer, 0, MEM_RELEASE);
    }
    if (pDumpFile != NULL) {
        VirtualFree(pDumpFile, 0, MEM_RELEASE);
    }
    pTmpBuffer = NULL;
    return result;
}

void LogDumpFileError(const wchar_t * pMsg, const wchar_t * pDumpFileName)
{
    wchar_t *buffer = NULL;

    buffer = ConcatenateTwoStrings(pMsg, pDumpFileName);
    PrintWin32Error(buffer);
    if (buffer != NULL) {
        VirtualFree(buffer, 0, MEM_RELEASE);
    }
}

wchar_t *ConcatenateTwoStrings(const wchar_t * pFirst, const wchar_t * pSecond)
{
    size_t sz = 0;
    wchar_t *buffer = NULL;

    assert(pFirst != NULL);
    assert(pSecond != NULL);

    sz = (wcslen(pFirst) + wcslen(pSecond) + 1 /* trailing zero */ ) * sizeof(wchar_t);
    buffer = (wchar_t *) VirtualAlloc(NULL, sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

    if (buffer == NULL) {
        PrintWin32Error(L"\nMemory allocation error");
        return NULL;
    }
    wcscpy_s(buffer, sz / sizeof(wchar_t), pFirst);
    wcscat_s(buffer, sz / sizeof(wchar_t), pSecond);
    return buffer;
}

wchar_t *ConcatenateThreeStrings(const wchar_t * pFirst, const wchar_t * pSecond, const wchar_t * pThird)
{
    wchar_t *result = NULL;
    wchar_t *buffer = NULL;

    assert(pThird != NULL);
    buffer = ConcatenateTwoStrings(pFirst, pSecond);

    if (buffer == NULL) {
        return NULL;
    }
    result = ConcatenateTwoStrings(buffer, pThird);
    VirtualFree(buffer, 0, MEM_RELEASE);
    return result;
}

void PrintWin32Error(const wchar_t * message)
{
    DWORD errorCode;
    void *buffer = NULL;

    assert(message != NULL);
    __try {
        errorCode = GetLastError();
        FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
                       FORMAT_MESSAGE_MAX_WIDTH_MASK | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                       NULL, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (wchar_t *) & buffer, 0, NULL);
        wprintf_s(L"%s. Windows error %#x: %s", message, errorCode, (wchar_t *) buffer);
    } __finally {
        if (buffer != NULL) {
            LocalFree(buffer);
        }
    }
}
