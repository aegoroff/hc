/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#include <locale.h>
#include <stdlib.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif

#include "encoding.h"
#include "targetver.h"
#include "bf.h"
#include "hashes.h"
#include "hc.h"

#include <apr_strings.h>

#include "str.h"
#include "hash.h"
#include "file.h"
#include "dir.h"
#include "configuration.h"
#include "argtable3.h"
#include "intl.h"
#ifdef _MSC_VER
#include "dbg_helpers.h"
#else
#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0
#endif

#define PROG_EXE PROGRAM_NAME ".exe"

typedef enum {
    mode_none = 0,
    mode_string = 1,
    mode_hash = 2,
    mode_file = 3,
    mode_dir = 4
} hc_mode_t;

static void prhc_print_table_syntax(void* argtable);
static void prhc_on_string(builtin_ctx_t* bctx, string_builtin_ctx_t* sctx, apr_pool_t* pool);
static void prhc_on_hash(builtin_ctx_t* bctx, hash_builtin_ctx_t* hctx, apr_pool_t* pool);
static void prhc_on_file(builtin_ctx_t* bctx, file_builtin_ctx_t* fctx, apr_pool_t* pool);
static void prhc_on_dir(builtin_ctx_t* bctx, dir_builtin_ctx_t* dctx, apr_pool_t* pool);
#ifdef _MSC_VER
static BOOL WINAPI prhc_ctrl_handler(DWORD fdw_ctrl_type);
#else
static void prhc_ctrl_handler(int flag);
#endif
static const char* prhc_get_executable_path(apr_pool_t* pool);
static void prhc_split_path(const char* path, const char** dir, const char** file, apr_pool_t* pool);

apr_pool_t* g_pool = NULL;
hc_mode_t g_mode = mode_none;

int main(int argc, const char* const argv[]) {
    
    apr_status_t status = APR_SUCCESS;

#ifdef _MSC_VER
#ifndef _DEBUG // only Release configuration dump generating
    SetUnhandledExceptionFilter(dbg_top_level_filter);
#endif
    SetConsoleCtrlHandler(prhc_ctrl_handler, TRUE);
    setlocale(LC_ALL, ".ACP");
#else
    signal(SIGINT, prhc_ctrl_handler);
    setlocale(LC_ALL, "C.UTF-8");
#endif
    
    setlocale(LC_NUMERIC, "C");

    status = apr_app_initialize(&argc, &argv, NULL);
    if(status != APR_SUCCESS) {
        lib_printf("Couldn't initialize APR");
        lib_new_line();
        out_print_error(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&g_pool, NULL);

#ifdef USE_GETTEXT
    const char* exe = prhc_get_executable_path(g_pool);
    const char* exe_file_name;
    const char* hc_base_dir;

    prhc_split_path(exe, &hc_base_dir, &exe_file_name, g_pool);

    bindtextdomain("hc", hc_base_dir); /* set the text message domain */
    textdomain("hc");
#endif /* USE_GETTEXT */

    hsh_initialize_hashes(g_pool);

    configuration_ctx_t* configuration_ctx = apr_pcalloc(g_pool, sizeof(configuration_ctx_t));
    configuration_ctx->pool = g_pool;
    configuration_ctx->argc = argc;
    configuration_ctx->argv = argv;
    configuration_ctx->pfn_on_string = &prhc_on_string;
    configuration_ctx->pfn_on_hash = &prhc_on_hash;
    configuration_ctx->pfn_on_file = &prhc_on_file;
    configuration_ctx->pfn_on_dir = &prhc_on_dir;

    conf_run_app(configuration_ctx);

    apr_pool_destroy(g_pool);
    return EXIT_SUCCESS;
}

void prhc_on_string(builtin_ctx_t* bctx, string_builtin_ctx_t* sctx, apr_pool_t* pool) {
    g_mode = mode_string;
    builtin_run(bctx, sctx, (void (*)(void*))str_run, pool);
}

void prhc_on_hash(builtin_ctx_t* bctx, hash_builtin_ctx_t* hctx, apr_pool_t* pool) {
    g_mode = mode_hash;
    builtin_run(bctx, hctx, (void (*)(void*))hash_run, pool);
}

void prhc_on_file(builtin_ctx_t* bctx, file_builtin_ctx_t* fctx, apr_pool_t* pool) {
    g_mode = mode_file;
    builtin_run(bctx, fctx, (void (*)(void*))file_run, pool);
}

void prhc_on_dir(builtin_ctx_t* bctx, dir_builtin_ctx_t* dctx, apr_pool_t* pool) {
    g_mode = mode_dir;
    builtin_run(bctx, dctx, (void (*)(void*))dir_run, pool);
}

void hc_print_copyright(void) {
    lib_printf(COPYRIGHT_FMT, APP_NAME);
}

void hc_print_cmd_syntax(void* argtable, void* end) {
    hc_print_copyright();
    prhc_print_table_syntax(argtable);
    arg_print_errors(stdout, end, PROGRAM_NAME);
}

void hc_print_syntax(void* argtable_s, void* argtable_h, void* argtable_f, void* argtable_d) {
    hc_print_copyright();
    prhc_print_table_syntax(argtable_s);
    prhc_print_table_syntax(argtable_h);
    prhc_print_table_syntax(argtable_f);
    prhc_print_table_syntax(argtable_d);
    hsh_print_hashes();
}

void prhc_print_table_syntax(void* argtable) {
    lib_printf(PROG_EXE);
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    arg_print_glossary_gnu(stdout, argtable);
    lib_new_line();
    lib_new_line();
}

#ifdef _MSC_VER
BOOL WINAPI prhc_ctrl_handler(DWORD fdw_ctrl_type) {
    if(fdw_ctrl_type != CTRL_C_EVENT) {
        return FALSE;
    }

    if (g_mode == mode_hash) {
        bf_output_timings(g_pool);
    }

    apr_pool_destroy(g_pool);
    apr_terminate();
    return FALSE;
}
#else
void prhc_ctrl_handler(int flag) {
    if (g_mode == mode_hash) {
        bf_output_timings(g_pool);
    }

    apr_pool_destroy(g_pool);
    apr_terminate();
}
#endif

const char* prhc_get_executable_path(apr_pool_t* pool) {
    uint32_t size = 512;
    char* buf = (char*)apr_pcalloc(pool, size);
    int do_realloc = 1;
    do {
#ifdef __APPLE_CC__
        int result = _NSGetExecutablePath(buf, &size);
        do_realloc = result == -1;
        if(do_realloc) {
            // if the buffer is not large enough, and * bufsize is set to the
            //     size required.
            // size + 1 made buffer null terminated
            buf = (char*) apr_pcalloc(pool, size + 1);
        } else {
            char* real_path = realpath(buf, NULL);
            if(real_path != NULL) {
                size_t len = strnlen(real_path, PATH_MAX);
                buf = (char*) apr_pcalloc(pool, len + 1);
                memcpy(buf, real_path, len);
                free(real_path);
            }
        }
#else
#ifdef _MSC_VER
        // size - 1 made buffer null terminated
        DWORD result = GetModuleFileNameA(NULL, buf, size - 1);
        DWORD lastError = GetLastError();

        do_realloc = result == (size - 1)
            && (lastError == ERROR_INSUFFICIENT_BUFFER || lastError == ERROR_SUCCESS);
#else
        // size - 1 made buffer null terminated
        ssize_t result = readlink("/proc/self/exe", buf, size - 1);

        do_realloc = result >= (size - 1);
#endif
        if(do_realloc) {
            size *= 2;
            buf = (char*)apr_pcalloc(pool, size);
        }
#endif
    } while(do_realloc);
    return buf;
}

#ifdef USE_GETTEXT
void prhc_split_path(const char* path, const char** d, const char** f, apr_pool_t* pool) {
#ifdef _MSC_VER
    char* dir = (char*)apr_pcalloc(pool, sizeof(char) * MAX_PATH);
    char* filename = (char*)apr_pcalloc(pool, sizeof(char) * MAX_PATH);
    char* drive = (char*)apr_pcalloc(pool, sizeof(char) * MAX_PATH);
    char* ext = (char*)apr_pcalloc(pool, sizeof(char) * MAX_PATH);
    _splitpath_s(path,
        drive, MAX_PATH, // Drive
        dir, MAX_PATH, // Directory
        filename, MAX_PATH, // Filename
        ext, MAX_PATH); // Extension

    *d = apr_pstrcat(pool, drive, dir, NULL);
    *f = apr_pstrcat(pool, filename, ext, NULL);
#else
    char* dir = apr_pstrdup(pool, path);
    *d = dirname(dir);
#ifdef __APPLE_CC__
    * f = basename(dir);
#else
    * f = path + strlen(dir) + 1;
#endif
#endif
}
#endif 