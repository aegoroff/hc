/*!
 * \brief   The file contains common Apache passwords cracker implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
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
#define MAX_DEFAULT_STR "10"
#define MAX_LINE_SIZE 4 * BINARY_THOUSAND - 1

static struct apr_getopt_option_t options[] = {
    {"dict", OPT_DICT, TRUE,
     "password's dictionary by default all" NEW_LINE "\t\t\t\tdigits, upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the password to" NEW_LINE "\t\t\t\tcrack. 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the password to" NEW_LINE "\t\t\t\tcrack. " MAX_DEFAULT_STR " by default"},
    {"file", OPT_FILE, TRUE, "full path to password's file"},
    {"hash", OPT_HASH, TRUE, "password to validate against (hash)"},
    {"login", OPT_LOGIN, TRUE, "login from password file to crack password for"},
    {"list", OPT_LIST, FALSE, "list accounts from .htpasswd file"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

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

    if (isListAccounts && file == NULL) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD "file must be specified if listing accounts parameter set" NEW_LINE);
        goto cleanup;
    }
    if (isListAccounts && hash != NULL) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD "accounts listing not supported when cracking hash" NEW_LINE);
        goto cleanup;
    }
    if (login != NULL && hash != NULL) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD "login parameter is not supported when cracking hash" NEW_LINE);
        goto cleanup;
    }
    if (file != NULL && hash != NULL) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD "impossible to crack file and hash simultaneously" NEW_LINE);
        goto cleanup;
    }
    if (file == NULL && hash == NULL) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD "one of options --file or --hash must be set" NEW_LINE);
        goto cleanup;
    }
    if (isListAccounts && file != NULL) {
        ListAccounts(file, OutputToConsole, pool);
        goto cleanup;
    }
    if (dict == NULL) {
        dict = alphabet;
    }
    if ((hash != NULL) && (file == NULL)) {
        CrackHtpasswdHash(dict, hash, passmin, passmax, pool);
    }
    if (file != NULL) {
        CrackFile(file, OutputToConsole, dict, passmin, passmax, login, pool);
    }

cleanup:
    apr_pool_destroy(pool);
    return EXIT_SUCCESS;
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

void CrackHtpasswdHash(const char* dict,
               const char* hash,
               const uint32_t    passmin,
               const uint32_t    passmax,
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

int CompareHashAttempt(void* hash, const char* pass, const uint32_t length)
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
               const uint32_t    passmin,
               const uint32_t    passmax,
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

void* CreateDigest(const char* hash, apr_pool_t* pool)
{
    UNREFERENCED_PARAMETER(pool);
    UNREFERENCED_PARAMETER(hash);
    return NULL;
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
        p = apr_strtok(line, APACHE_PWD_SEPARATOR, &last);

        if (p == NULL || last == NULL || last[0] == '\0' || !IsValidAsciiString(p, MAX_LINE_SIZE)) {
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
    char* last = NULL;
    
    int count = 0;
    CrackContext* crackContext = context;

    line = (char*)apr_pcalloc(pool, MAX_LINE_SIZE + 1);

    if (line == NULL) {
        return;
    }

    while (apr_file_gets(line, MAX_LINE_SIZE, fileHandle) != APR_EOF) {
        size_t i = 0;

        if (strlen(line) < 3 || !IsValidAsciiString(line, MAX_LINE_SIZE) || strstr(line, APACHE_PWD_SEPARATOR) == NULL) {
            continue;
        }

        p = apr_strtok(line, APACHE_PWD_SEPARATOR, &last);

        if (p == NULL || last == NULL || last[0] == '\0' || p[0] == '\0') {
            continue;
        }

        if (count++ > 0 && crackContext->Login == NULL) {
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

        for (i = strlen(hash); i > 0; --i) {
            if ((hash[i - 1] == '\r') || (hash[i - 1] == '\n')) {
                hash[i - 1] = 0;
            } else   {
                break;
            }
        }

        ctx->IsFinishLine = TRUE;
        ctx->StringToPrint = hash;
        PfnOutput(ctx);

        CrackHtpasswdHash(crackContext->Dict, hash, crackContext->Passmin, crackContext->Passmax, pool);

        memset(line, 0, MAX_LINE_SIZE);
    }
}

int IsValidAsciiString(const char* string, size_t size)
{
    size_t i = 0;
    
    for (; i < size; ++i) {
        char c = string[i];
        if (c < 0 || (c < 32 && c > 0 && c != '\r' && c != '\n')) {
            return FALSE;
        }
        if (c == 0) {
            break;
        }
    }
    return TRUE;
}

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    UNREFERENCED_PARAMETER(digest);
    UNREFERENCED_PARAMETER(input);
    UNREFERENCED_PARAMETER(inputLen);
    return APR_SUCCESS;
}

int CompareHash(apr_byte_t* digest, const char* checkSum)
{
    UNREFERENCED_PARAMETER(digest);
    UNREFERENCED_PARAMETER(checkSum);
    return 0;
}