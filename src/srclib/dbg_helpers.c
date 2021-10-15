/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   he file contains debugging helpers implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#include "targetver.h"
#include <stdio.h>
#include "dbg_helpers.h"

#define DBG_HELP_DLL "DBGHELP.DLL"
#define DUMP_FILE_NAME PROGRAM_NAME ".exe.dmp"
#define DUMP_FUNCTION "MiniDumpWriteDump"
#define UNHANDLED_EXCEPTION_OCCURRED " An unhandled exception occurred. "

/*
   dbg_ - public members
   prdbg_ - private members
*/

static void prdbg_print_win32_error(const char* message);

LONG WINAPI dbg_top_level_filter(struct _EXCEPTION_POINTERS* p_exception_info) {
    LONG result = EXCEPTION_CONTINUE_SEARCH; // finalize process in standard way by default
    HMODULE dll = NULL;
    MINIDUMP_EXCEPTION_INFORMATION ex_info = { 0 };
    BOOL is_ok = FALSE;
    MINIDUMPWRITEDUMP pfn_dump = NULL;
    HANDLE h_file = NULL;

    dll = LoadLibraryA(DBG_HELP_DLL);

    if(dll == NULL) {
        prdbg_print_win32_error(" Cannot load dll " DBG_HELP_DLL);
        return result;
    }
    // get func address
    pfn_dump = (MINIDUMPWRITEDUMP)GetProcAddress(dll, DUMP_FUNCTION);
    if(!pfn_dump) {
        prdbg_print_win32_error(" Cannot get address of " DUMP_FUNCTION " function");
        return result;
    }

    h_file = CreateFileA(DUMP_FILE_NAME,
                                      GENERIC_WRITE,
                                      0,
                                      NULL,
                                      CREATE_ALWAYS,
                                      FILE_ATTRIBUTE_NORMAL,
                                      NULL);

    if(h_file == INVALID_HANDLE_VALUE) {
        prdbg_print_win32_error(UNHANDLED_EXCEPTION_OCCURRED "Error on creating dump file: " DUMP_FILE_NAME);
        return result;
    }

    ex_info.ThreadId = GetCurrentThreadId();
    ex_info.ExceptionPointers = p_exception_info;
    ex_info.ClientPointers = 0;

    // Write pDumpFile
    is_ok = pfn_dump(GetCurrentProcess(),
                     GetCurrentProcessId(), h_file, MiniDumpNormal, &ex_info, NULL, NULL);
    if(is_ok) {
        printf_s(UNHANDLED_EXCEPTION_OCCURRED "Dump saved to: %s", DUMP_FILE_NAME);
        result = EXCEPTION_EXECUTE_HANDLER;
    } else {
        prdbg_print_win32_error(UNHANDLED_EXCEPTION_OCCURRED "Error saving dump file: " DUMP_FILE_NAME);
    }
    CloseHandle(h_file);
    return result;
}

void prdbg_print_win32_error(const char* message) {
    DWORD error_code = 0;
    void* buffer = NULL;

    __try {
        error_code = GetLastError();
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
                                                 FORMAT_MESSAGE_MAX_WIDTH_MASK | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                                                 NULL, error_code, MAKELANGID(LANG_NEUTRAL,
                                                     SUBLANG_DEFAULT), (char*)&buffer, 0, NULL);
        printf_s("%s. Windows error %#x: %s", message, error_code, (char*)buffer);
    } __finally {
        if(buffer != NULL) {
            LocalFree(buffer);
        }
    }
}
