/*!
 * \brief   The file contains common Apache passwords cracker implementation
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
#define HLP_OPT_END "\t\t%s" NEW_LINE NEW_LINE
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
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
#define OPT_LOGIN 'l'
#define OPT_LIST 'i'

#define APACHE_PWD_SEPARATOR ":"
#define MAX_DEFAULT 10
#define MAX_LINE_SIZE 8 * BINARY_THOUSAND - 1

static struct apr_getopt_option_t options[] = {
    {"dict", OPT_DICT, TRUE,
     "initial password's dictionary by default all" NEW_LINE "\t\t\t\tdigits, upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the password to" NEW_LINE "\t\t\t\tcrack. 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the password to" NEW_LINE "\t\t\t\tcrack. 10 by default"},
    {"file", OPT_FILE, TRUE, "full path to password's file"},
    {"hash", OPT_HASH, TRUE, "password to validate against (hash)"},
    {"password", OPT_PWD, TRUE, "password to validate"},
    {"login", OPT_LOGIN, TRUE, "login from password file to crack password for"},
    {"list", OPT_LIST, FALSE, "list accounts from .htpasswd file"},
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
    const char* login = NULL;
    int isListAccounts = FALSE;

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
            case OPT_LOGIN:
                login = apr_pstrdup(pool, optarg);
                break;
            case OPT_LIST:
                isListAccounts = TRUE;
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
    if (isListAccounts && file != NULL) {
        ListAccounts(file, OutputToConsole, pool);
        goto cleanup;
    }
    if (dict == NULL) {
        dict = alphabet;
    }
    if ((hash != NULL) && (pwd != NULL) && (file == NULL)) {
        CrackHash(dict, hash, passmin, passmax, pool);
    }
    if (file != NULL) {
        CrackFile(file, OutputToConsole, dict, passmin, passmax, login, pool);
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
        str = BruteForce(passmin,
                         passmax ? passmax : MAX_DEFAULT,
                         dict,
                         hash,
                         &attempts,
                         PassThrough,
                         pool);
    }

    StopTimer();
    time = ReadElapsedTime();
    CrtPrintf(NEW_LINE "Attempts: %llu Time " FULL_TIME_FMT,
              attempts,
              time.hours,
              time.minutes,
              time.seconds);
    NewLine();
    if (str != NULL) {
        CrtPrintf("Password is: %s", str);
    } else {
        CrtPrintf("Nothing found");
    }
    NewLine();
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
               const char* login,
               apr_pool_t* pool)
{
    CrackContext context = { 0 };
    context.Dict = dict;
    context.Passmin = passmin;
    context.Passmax = passmax;
    context.Login = login;

    ReadPasswdFile(file, PfnOutput, CrackFileCallback, &context, pool);
}

void ListAccounts(const char* file, void (* PfnOutput)(OutputContext* ctx), apr_pool_t * pool)
{
    ReadPasswdFile(file, PfnOutput, ListAccountsCallback, file, pool);
}

void ReadPasswdFile(
    const char* file,
    void (* PfnOutput)(OutputContext* ctx), 
    void (* PfnCallback)(OutputContext* ctx, void (* PfnOutput)(OutputContext* ctx), apr_file_t* fileHandle, void* context, apr_pool_t* pool),
    void* context,
    apr_pool_t * pool)
{
    apr_file_t* fileHandle = NULL;
    apr_status_t status = APR_SUCCESS;
    OutputContext ctx = { 0 };

    status = apr_file_open(&fileHandle, file, APR_READ, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        return;
    }

    PfnCallback(&ctx, PfnOutput, fileHandle, context, pool);
    
    status = apr_file_close(fileHandle);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
}

void ListAccountsCallback(
    OutputContext* ctx,
    void (* PfnOutput)(OutputContext* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool)
{
    char* line = NULL;
    char* p = NULL;
    char* parts = NULL;
    char* last = NULL;
    int count = 0;
    const char* file = context;

    line = (char*)apr_pcalloc(pool, MAX_LINE_SIZE + 1);

    if (line == NULL) {
        return;
    }

    ctx->IsFinishLine = FALSE;
    ctx->StringToPrint = " file: ";
    PfnOutput(ctx);
    ctx->IsFinishLine = TRUE;
    ctx->StringToPrint = file;
    PfnOutput(ctx);
    
    ctx->StringToPrint = " accounts:";
    PfnOutput(ctx);

    while (apr_file_gets(line, MAX_LINE_SIZE, fileHandle) != APR_EOF) {
        parts = apr_pstrdup(pool, line);        /* strtok wants non-const data */
        p = apr_strtok(parts, APACHE_PWD_SEPARATOR, &last);

        if (p == NULL || last == NULL || strlen(last) == 0) {
            continue;
        }
        ctx->IsFinishLine = FALSE;
        ctx->StringToPrint = "   ";
        PfnOutput(ctx);
        ctx->IsFinishLine = TRUE;
        ctx->StringToPrint = p;
        PfnOutput(ctx);
        memset(line, 0, MAX_LINE_SIZE);
        ++count;
    }

    if (count == 0) {
        ctx->IsFinishLine = TRUE;
        ctx->StringToPrint = " No accounts found in the file.";
        PfnOutput(ctx);
    }
}

void CrackFileCallback(
    OutputContext* ctx,
    void (* PfnOutput)(OutputContext* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool)
{
    char* hash = NULL;
    char* line = NULL;
    char* p = NULL;
    char* parts = NULL;
    char* last = NULL;
    int i = 0;
    int count = 0;
    CrackContext* crackContext = context;

    line = (char*)apr_pcalloc(pool, MAX_LINE_SIZE + 1);

    if (line == NULL) {
        return;
    }

    while (apr_file_gets(line, MAX_LINE_SIZE, fileHandle) != APR_EOF) {
        if (strlen(line) < 3) {
            continue;
        }

        parts = apr_pstrdup(pool, line);        /* strtok wants non-const data */
        p = apr_strtok(parts, APACHE_PWD_SEPARATOR, &last);

        if (p == NULL || last == NULL || strlen(last) == 0) {
            continue;
        }

        if (count++ > 0) {
            ctx->IsFinishLine = TRUE;
            ctx->StringToPrint = "";
            PfnOutput(ctx);

            ctx->StringToPrint = "-------------------------------------------------";
            PfnOutput(ctx);

            ctx->StringToPrint = "";
            PfnOutput(ctx);
        }

        if ((crackContext->Login != NULL) && (apr_strnatcasecmp(p, crackContext->Login) != 0)) {
            continue;
        }

        ctx->IsFinishLine = FALSE;
        ctx->StringToPrint = "Login: ";
        PfnOutput(ctx);
        ctx->StringToPrint = p;
        PfnOutput(ctx);
        ctx->StringToPrint = " Hash: ";
        PfnOutput(ctx);

        while (p) {
            p = apr_strtok(NULL, APACHE_PWD_SEPARATOR, &last);
            if (p != NULL) {
                hash = p;
            }
        }

        if (hash == NULL) {
            continue;
        }

        for (i = strlen(hash) - 1; i >= 0; --i) {
            if ((hash[i] == '\r') || (hash[i] == '\n')) {
                hash[i] = 0;
            } else   {
                break;
            }
        }

        ctx->IsFinishLine = TRUE;
        ctx->StringToPrint = hash;
        PfnOutput(ctx);

        CrackHash(crackContext->Dict, hash, crackContext->Passmin, crackContext->Passmax, pool);

        memset(line, 0, MAX_LINE_SIZE);
    }
}