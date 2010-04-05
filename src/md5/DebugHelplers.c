#include "targetver.h"
#include <stdio.h>
#include "DebugHelplers.h"

#define DBG_HELP_DLL L"DBGHELP.DLL"
#define DUMP_FILE_NAME L"md5.exe.dmp"

void PrintWin32Error(const wchar_t * message);

LONG WINAPI TopLevelFilter(struct _EXCEPTION_POINTERS *pExceptionInfo)
{
    LONG result = EXCEPTION_CONTINUE_SEARCH;    // finalize process in standart way by default
    HMODULE hDll = NULL;
    MINIDUMP_EXCEPTION_INFORMATION exInfo;
    BOOL isOK = FALSE;
    MINIDUMPWRITEDUMP pfnDump;
    HANDLE hFile;

    hDll = LoadLibraryW(DBG_HELP_DLL);

    if (hDll == NULL) {
        PrintWin32Error(L" Cannot load dll " DBG_HELP_DLL);
        return result;
    }
    // get func address
    pfnDump = (MINIDUMPWRITEDUMP) GetProcAddress(hDll, "MiniDumpWriteDump");
    if (!pfnDump) {
        PrintWin32Error(L" Cannot get address of MiniDumpWriteDump function");
        return result;
    }

    hFile = CreateFileW(DUMP_FILE_NAME, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE) {
        PrintWin32Error(L" An unhandled exception occured. Error on creating dump file: " DUMP_FILE_NAME);
        return result;
    }

    exInfo.ThreadId = GetCurrentThreadId();
    exInfo.ExceptionPointers = pExceptionInfo;
    exInfo.ClientPointers = 0;

    // Write pDumpFile
    isOK = pfnDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &exInfo, NULL, NULL);
    if (isOK) {
        wprintf_s(L" An unhandled exception occured. Dump saved to: %s", DUMP_FILE_NAME);
        result = EXCEPTION_EXECUTE_HANDLER;
    } else {
        PrintWin32Error(L" An unhandled exception occured. Error saving dump file: " DUMP_FILE_NAME);
    }
    CloseHandle(hFile);
    return result;
}

void PrintWin32Error(const wchar_t * message)
{
    DWORD errorCode;
    void *buffer = NULL;

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
