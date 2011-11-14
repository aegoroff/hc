/*!
 * \brief   The file contains Hash LINQ implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#include "targetver.h"
#include <assert.h>
#include "implementation.h"
#include "hlinq.h"
#include "..\srclib\bf.h"
#include "..\srclib\encoding.h"

#ifdef WIN32
#include "..\srclib\DebugHelplers.h"
#endif

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define LINE_FEED '\n'

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s" NEW_LINE NEW_LINE
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
#define FILE_INFO_COLUMN_SEPARATOR " | "
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "

#define OPT_HELP '?'
#define OPT_FILE 'f'

#define MAX_LINE_SIZE 32 * BINARY_THOUSAND - 1

static struct apr_getopt_option_t options[] = {
    {"file", OPT_FILE, TRUE, "full path to password's file"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_getopt_t* opt = NULL;
    int c = 0;
    const char* optarg = NULL;
    apr_status_t status = APR_SUCCESS;
    const char* file = NULL;

#ifdef WIN32
#ifndef _DEBUG  // only Release configuration dump generating
    SetUnhandledExceptionFilter(TopLevelFilter);
#endif
#endif

    setlocale(LC_ALL, ".ACP");
    setlocale(LC_NUMERIC, "C");

    status = apr_app_initialize(&argc, &argv, NULL);
    if (status != APR_SUCCESS) {
        CrtPrintf("Couldn't initialize APR");
        NewLine();
        PrintError(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&pool, NULL);
    apr_getopt_init(&opt, pool, argc, argv);

    while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
        switch (c) {
            case OPT_HELP:
                PrintUsage();
                goto cleanup;
            case OPT_FILE:
                file = apr_pstrdup(pool, optarg);
                break;
        }
    }

    if ((status != APR_EOF) || (argc < 2)) {
        PrintUsage();
        goto cleanup;
    }


cleanup:
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

void PrintError(apr_status_t status)
{
    char errbuf[ERROR_BUFFER_SIZE];
    apr_strerror(status, errbuf, ERROR_BUFFER_SIZE);
    CrtPrintf("%s", errbuf);
    NewLine();
}

const char* CreateErrorMessage(apr_status_t status, apr_pool_t* pool)
{
    char* message = (char*)apr_pcalloc(pool, ERROR_BUFFER_SIZE);
    apr_strerror(status, message, ERROR_BUFFER_SIZE);
    return message;
}

void OutputErrorMessage(apr_status_t status, void (* PfnOutput)(
                            OutputContext* ctx), apr_pool_t* pool)
{
    OutputContext ctx = { 0 };
    ctx.StringToPrint = CreateErrorMessage(status, pool);
    ctx.IsPrintSeparator = FALSE;
    ctx.IsFinishLine = TRUE;
    PfnOutput(&ctx);
}

void PrintUsage(void)
{
    int i = 0;
    PrintCopyright();
    CrtPrintf("usage: " PROGRAM_NAME " [OPTION] ..." NEW_LINE NEW_LINE "Options:" NEW_LINE NEW_LINE);
    for (; i < sizeof(options) / sizeof(apr_getopt_option_t); ++i) {
        CrtPrintf(options[i].has_arg ? HLP_ARG : HLP_NO_ARG,
                  (char)options[i].optch, options[i].name, options[i].description);
    }
}

void PrintCopyright(void)
{
    CrtPrintf(COPYRIGHT_FMT, APP_NAME);
}

void OutputToConsole(OutputContext* ctx)
{
    if (ctx == NULL) {
        assert(ctx != NULL);
        return;
    }
    CrtPrintf("%s", ctx->StringToPrint);
    if (ctx->IsPrintSeparator) {
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->IsFinishLine) {
        NewLine();
    }
}