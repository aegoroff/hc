// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <locale.h>

#include "encoding.h"
#include "targetver.h"
#include "../srclib/bf.h"
#include "../l2h/hashes.h"
#include "hc.h"
#include "str.h"
#include "hash.h"
#include "file.h"
#include "dir.h"
#include "configuration.h"
#include "argtable3.h"
#ifdef WIN32
#include "../srclib/dbg_helpers.h"
#endif

#define PROG_EXE PROGRAM_NAME ".exe"

static void prhc_print_table_syntax(void* argtable);
static void prhc_on_string(builtin_ctx_t* bctx, string_builtin_ctx_t* sctx, apr_pool_t* pool);
static void prhc_on_hash(builtin_ctx_t* bctx, hash_builtin_ctx_t* hctx, apr_pool_t* pool);
static void prhc_on_file(builtin_ctx_t* bctx, file_builtin_ctx_t* fctx, apr_pool_t* pool);
static void prhc_on_dir(builtin_ctx_t* bctx, dir_builtin_ctx_t* dctx, apr_pool_t* pool);

int main(int argc, const char* const argv[]) {
    apr_pool_t* pool = NULL;
    apr_status_t status = APR_SUCCESS;

#ifdef WIN32
#ifndef _DEBUG // only Release configuration dump generating
    SetUnhandledExceptionFilter(dbg_top_level_filter);
#endif
#endif

    setlocale(LC_ALL, ".ACP");
    setlocale(LC_NUMERIC, "C");

#ifdef USE_GETTEXT
    bindtextdomain("hc", LOCALEDIR); /* set the text message domain */
    textdomain("hc");
#endif /* USE_GETTEXT */

    status = apr_app_initialize(&argc, &argv, NULL);
    if(status != APR_SUCCESS) {
        lib_printf(_("Couldn't initialize APR"));
        lib_new_line();
        out_print_error(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&pool, NULL);
    hsh_initialize_hashes(pool);

    configuration_ctx_t* configuration_ctx = apr_pcalloc(pool, sizeof(configuration_ctx_t));
    configuration_ctx->pool = pool;
    configuration_ctx->argc = argc;
    configuration_ctx->argv = argv;
    configuration_ctx->pfn_on_string = &prhc_on_string;
    configuration_ctx->pfn_on_hash = &prhc_on_hash;
    configuration_ctx->pfn_on_file = &prhc_on_file;
    configuration_ctx->pfn_on_dir = &prhc_on_dir;
    
    conf_run_app(configuration_ctx);
    
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

void prhc_on_string(builtin_ctx_t* bctx, string_builtin_ctx_t* sctx, apr_pool_t* pool) {
    builtin_run(bctx, sctx, str_run, pool);
}

void prhc_on_hash(builtin_ctx_t* bctx, hash_builtin_ctx_t* hctx, apr_pool_t* pool) {
    builtin_run(bctx, hctx, hash_run, pool);
}

void prhc_on_file(builtin_ctx_t* bctx, file_builtin_ctx_t* fctx, apr_pool_t* pool) {
    builtin_run(bctx, fctx, file_run, pool);
}

void prhc_on_dir(builtin_ctx_t* bctx, dir_builtin_ctx_t* dctx, apr_pool_t* pool) {
    builtin_run(bctx, dctx, dir_run, pool);
}

void hc_print_copyright(void) {
    lib_printf(COPYRIGHT_FMT, APP_NAME);
}

void hc_print_cmd_syntax(void* argtable, void* end) {
    hc_print_copyright();
    prhc_print_table_syntax(argtable);
    arg_print_errors(stdout, end, PROGRAM_NAME);
}

void hc_print_syntax(void* argtableS, void* argtableH, void* argtableF, void* argtableD) {
    hc_print_copyright();
    prhc_print_table_syntax(argtableS);
    prhc_print_table_syntax(argtableH);
    prhc_print_table_syntax(argtableF);
    prhc_print_table_syntax(argtableD);
    hsh_print_hashes();
}

void prhc_print_table_syntax(void* argtable) {
    lib_printf(PROG_EXE);
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);
    arg_print_glossary_gnu(stdout, argtable);
    lib_new_line();
    lib_new_line();
}