/*!
 * \brief   The file contains common hash calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#include "targetver.h"
#include <assert.h>
#include "implementation.h"
#include "apc.h"
#include "..\srclib\bf.h"
#include "..\srclib\encoding.h"

#ifdef WIN32
#include "..\srclib\DebugHelplers.h"
#endif

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define BYTE_CHARS_SIZE 2   // byte representation string length

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s\n\n"
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number\n"
#define FILE_INFO_COLUMN_SEPARATOR " | "
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "

#define OPT_DICT 'a'
#define OPT_MIN 'n'
#define OPT_MIN_FULL "min"
#define OPT_MAX 'x'
#define OPT_MAX_FULL "max"
#define OPT_HELP '?'
#define OPT_SEARCH 'h'
#define OPT_SAVE 'o'

static struct apr_getopt_option_t options[] = {
    {"dict", OPT_DICT, TRUE,
     "initial string's dictionary by default all\n\t\t\t\tdigits, upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the string to\n\t\t\t\trestore. 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the string to\n\t\t\t\trestore.\n\t\t\t\tThe length of the dictionary by default"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

static char* alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_getopt_t* opt = NULL;
    int c = 0;
    const char* optarg = NULL;
    const char* dict = NULL;
    apr_status_t status = APR_SUCCESS;
    uint32_t passmin = 1;   // important!
    uint32_t passmax = 0;

#ifdef WIN32
#ifndef _DEBUG  // only Release configuration dump generating
    SetUnhandledExceptionFilter(TopLevelFilter);
#endif
#endif

    setlocale(LC_ALL, ".ACP");
    setlocale(LC_NUMERIC, "C");

    status = apr_app_initialize(&argc, &argv, NULL);
    if (status != APR_SUCCESS) {
        CrtPrintf("Couldn't initialize APR\n");
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
            case OPT_DICT:
                dict = apr_pstrdup(pool, optarg);
                break;
            case OPT_MIN:
                if (!sscanf(optarg, NUMBER_PARAM_FMT_STRING, &passmin)) {
                    CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_MIN_FULL, optarg);
                    goto cleanup;
                }
                break;
            case OPT_MAX:
                if (!sscanf(optarg, NUMBER_PARAM_FMT_STRING, &passmax)) {
                    CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_MAX_FULL, optarg);
                    goto cleanup;
                }
                break;
        }
    }

    if ((status != APR_EOF) || (argc < 2)) {
        PrintUsage();
        goto cleanup;
    }
    if (dict == NULL) {
        dict = alphabet;
    }

cleanup:
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
}

int CompareHashAttempt(void* hash, const char* pass, uint32_t length)
{
    // TODO: CompareHashAttempt
    return 1;
}

void PrintError(apr_status_t status)
{
    char errbuf[ERROR_BUFFER_SIZE];
    apr_strerror(status, errbuf, ERROR_BUFFER_SIZE);
    CrtPrintf("%s\n", errbuf);
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
    CrtPrintf("usage: " PROGRAM_NAME " [OPTION] ...\n\nOptions:\n\n");
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
    CrtPrintf(ctx->StringToPrint);
    if (ctx->IsPrintSeparator) {
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->IsFinishLine) {
        NewLine();
    }
}
