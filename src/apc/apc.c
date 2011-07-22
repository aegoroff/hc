/*!
 * \brief   The file contains common Apache password recovery implementation
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
#define LINE_FEED '\n'

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
#define OPT_FILE 'f'
#define OPT_HASH 'h'
#define OPT_PWD 'p'

#define APACHE_PWD_SEPARATOR ":"

static struct apr_getopt_option_t options[] = {
    {"dict", OPT_DICT, TRUE,
     "initial string's dictionary by default all\n\t\t\t\tdigits, upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the string to\n\t\t\t\trestore. 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the string to\n\t\t\t\trestore.\n\t\t\t\tThe length of the dictionary by default"},
    {"file", OPT_FILE, TRUE, "full path to password's file"},
    {"hash", OPT_HASH, TRUE, "password to validate against (hash)"},
    {"password", OPT_PWD, TRUE, "password to validate"},
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
    const char* file = NULL;
    const char* hash = NULL;
    const char* pwd = NULL;

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
            case OPT_FILE:
                file = apr_pstrdup(pool, optarg);
                break;
            case OPT_HASH:
                hash = apr_pstrdup(pool, optarg);
                break;
            case OPT_PWD:
                pwd = apr_pstrdup(pool, optarg);
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
    if (hash != NULL && pwd != NULL && file == NULL) {
        CrackHash(dict, hash, passmin, passmax, pool);
    }
    if (file != NULL) {
        CrackFile(file, OutputToConsole, dict, passmin, passmax, pool);
    }

cleanup:
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
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

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool)
{
    char* str = NULL;
    uint64_t attempts = 0;
    Time time = { 0 };

    StartTimer();

    if (CompareHashAttempt(hash, "", 0)) {
        str = "Empty string";
    } else {
        str = BruteForce(passmin, passmax ? passmax : strlen(dict), dict, hash, &attempts, PassThrough, pool);
    }

    StopTimer();
    time = ReadElapsedTime();
    CrtPrintf("\nAttempts: %llu Time " FULL_TIME_FMT,
              attempts,
              time.hours,
              time.minutes,
              time.seconds);
    NewLine();
    if (str != NULL) {
        CrtPrintf("Password is: %s\n", str);
    } else {
        CrtPrintf("Nothing found\n");
    }
}

int CompareHashAttempt(void* hash, const char* pass, uint32_t length)
{
    const char* h = (const char*)hash;
    UNREFERENCED_PARAMETER(length);
    return apr_password_validate(pass, h) == APR_SUCCESS;
}

void* PassThrough(const char* hash, apr_pool_t* pool)
{
    UNREFERENCED_PARAMETER(pool);
    return hash;
}

void CrackFile(const char* file,
               void        (* PfnOutput)(OutputContext* ctx),
               const char* dict,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool)
{
    apr_file_t* fileHandle = NULL;
    apr_status_t status = APR_SUCCESS;
    char ch = 0;
    apr_finfo_t info = { 0 };
    char* line = NULL;
    char* p = NULL;
    char* parts = NULL;
    char* last = NULL;
    char* hash = NULL;
    OutputContext ctx = { 0 };
    int i = 0;

    status = apr_file_open(&fileHandle, file, APR_READ, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        return;
    }

    status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, fileHandle);

    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        goto cleanup;
    }

    line = (char*)apr_pcalloc(pool, info.size);

    while (apr_file_gets(line, info.size, fileHandle) != APR_EOF) {
        
        parts = apr_pstrdup(pool, line);        /* strtok wants non-const data */
        p = apr_strtok(parts, APACHE_PWD_SEPARATOR, &last);
        
        if (p != NULL) {
            ctx.IsFinishLine = FALSE;
            ctx.StringToPrint = "Login: ";
            PfnOutput(&ctx);
            ctx.StringToPrint = p;
            PfnOutput(&ctx);
            ctx.StringToPrint = " Hash: ";
            PfnOutput(&ctx);
        }

        while (p) {
            p = apr_strtok(NULL, APACHE_PWD_SEPARATOR, &last);
            if (p != NULL) {
                hash = p;
            }
        }

        for(i = strlen(hash) - 1; i >= 0; --i) {
            if(hash[i] == '\r' || hash[i] == '\n') {
                hash[i] = 0;
            }
            else {
                break;
            }
        }

        ctx.IsFinishLine = TRUE;
        ctx.StringToPrint = hash;
        PfnOutput(&ctx);

        CrackHash(dict, hash, passmin, passmax, pool);

        ctx.StringToPrint = "";
        PfnOutput(&ctx);
        
        ctx.StringToPrint = "-----------------------------------------";
        PfnOutput(&ctx);

        ctx.StringToPrint = "";
        PfnOutput(&ctx);

        memset(line, 0, info.size);
    }

cleanup:
    status = apr_file_close(fileHandle);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
}