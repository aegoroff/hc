// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains configuration module implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-09-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <argtable2.h>
#include "targetver.h"
#include "configuration.h"

#define OPT_HELP_SHORT "h"
#define OPT_HELP_LONG "help"
#define OPT_HELP_DESCR "print this help and exit"

#define OPT_F_SHORT "f"
#define OPT_F_LONG "file"
#define OPT_F_DESCR "one or more files"

#define OPT_C_SHORT "C"
#define OPT_C_LONG "command"
#define OPT_C_DESCR "query text from command line"

/*
    conf_ - public members
    prconf_ - private members
*/

static void prconf_print_copyright(void);
static void prconf_print_syntax(void* argtable);

void conf_configure_app(configuration_ctx_t* ctx) {
    struct arg_file* files = arg_filen(OPT_F_SHORT, OPT_F_LONG, NULL, 0, ctx->argc + 2, OPT_F_DESCR);
    struct arg_str* command = arg_str0(OPT_C_SHORT, OPT_C_LONG, NULL, OPT_C_DESCR);
    struct arg_lit* help = arg_lit0(OPT_HELP_SHORT, OPT_HELP_LONG, OPT_HELP_DESCR);
    struct arg_end* end = arg_end(10);

    void* argtable[] = {help, command, files, end};

    if(arg_nullcheck(argtable) != 0) {
        prconf_print_syntax(argtable);
        goto cleanup;
    }

    int nerrors = arg_parse(ctx->argc, ctx->argv, argtable);

    if(nerrors > 0 || help->count > 0 || command->count == 0 && files->count == 0) {
        prconf_print_syntax(argtable);
        if(help->count == 0 && ctx->argc > 1) {
            arg_print_errors(stdout, end, PROGRAM_NAME);
        }
        goto cleanup;
    }

    if(command->count > 0) {
        ctx->on_string(command->sval[0]);
    }
    else if(files->count > 0) {
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
