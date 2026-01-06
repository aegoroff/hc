/*!
 * \brief   The file contains configuration module implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-09-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include "targetver.h"
#include "lib.h"
#include "argtable3.h"
#include "configuration.h"

#define OPT_HELP_SHORT "h"
#define OPT_HELP_LONG "help"
#define OPT_HELP_DESCR "print this help and exit"

#define OPT_F_SHORT "f"
#define OPT_F_LONG "file"
#define OPT_F_DESCR "query from one or more files"

#define OPT_Q_SHORT "q"
#define OPT_Q_LONG "query"
#define OPT_C_DESCR "query text from command line"

/*
    conf_ - public members
    prconf_ - private members
*/

static void prconf_print_copyright(void);
static void prconf_print_syntax(void* argtable);

void conf_configure_app(configuration_ctx_t* ctx) {
    struct arg_file* files = arg_filen(OPT_F_SHORT, OPT_F_LONG, NULL, 0, ctx->argc + 2, OPT_F_DESCR);
    struct arg_str* query = arg_str0(OPT_Q_SHORT, OPT_Q_LONG, NULL, OPT_C_DESCR);
    struct arg_lit* help = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_end* end = arg_end(10);

    void* argtable[] = { help, query, files, end };

    if(arg_nullcheck(argtable) != 0) {
        prconf_print_syntax(argtable);
        goto cleanup;
    }

    const int nerrors = arg_parse(ctx->argc, (char**)ctx->argv, argtable);

    if(nerrors > 0 || help->count > 0 || query->count == 0 && files->count == 0) { // -V648
        prconf_print_syntax(argtable);
        if(help->count == 0 && ctx->argc > 1) {
            arg_print_errors(stdout, end, PROGRAM_NAME);
        }
        goto cleanup;
    }

    if(query->count > 0) {
        ctx->on_string(query->sval[0]);
    } else if(files->count > 0) {
        ctx->on_file(files);
    }

cleanup:
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
}

void prconf_print_copyright(void) {
    lib_printf(COPYRIGHT_FMT, APP_NAME);
}

void prconf_print_syntax(void* argtable) {
    prconf_print_copyright();

    lib_printf(PROG_EXE);
    arg_print_syntax(stdout, argtable, NEW_LINE NEW_LINE);

    arg_print_glossary_gnu(stdout, argtable);
}
