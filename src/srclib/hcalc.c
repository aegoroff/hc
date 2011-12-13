/*!
 * \brief   The file contains common hash calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#include "targetver.h"
#include <assert.h>
#include <math.h>
#include "implementation.h"
#include "hcalc.h"
#include "bf.h"
#include "encoding.h"
#include "traverse.h"

#ifdef WIN32
#include "DebugHelplers.h"
#endif

#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s" NEW_LINE NEW_LINE
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "


#define OPT_FILE 'f'
#define OPT_DIR 'd'
#define OPT_EXCLUDE 'e'
#define OPT_INCLUDE 'i'
#define OPT_STRING 's'
#define OPT_HASH 'm'
#define OPT_DICT 'a'
#define OPT_MIN 'n'
#define OPT_MIN_FULL "min"
#define OPT_MAX 'x'
#define OPT_MAX_FULL "max"
#define OPT_CRACK 'c'
#define OPT_LOWER 'l'
#define OPT_RECURSIVELY 'r'
#define OPT_TIME 't'
#define OPT_HELP '?'
#define OPT_SEARCH 'h'
#define OPT_SAVE 'o'
#define OPT_LIMIT 'z'
#define OPT_LIMIT_FULL "limit"
#define OPT_OFFSET 'q'
#define OPT_OFFSET_FULL "offset"
#define PATTERN_MATCH_DESCR_TAIL "the pattern specified" NEW_LINE "\t\t\t\tit's possible to use several patterns" NEW_LINE "\t\t\t\tseparated by ;"

#define MAX_DEFAULT "10"

static struct apr_getopt_option_t options[] = {
    {"file", OPT_FILE, TRUE, "input full file path to calculate " HASH_NAME " sum for"},
    {"dir", OPT_DIR, TRUE, "full path to dir to calculate" NEW_LINE "\t\t\t\t" HASH_NAME " of all content"},
    {"exclude", OPT_EXCLUDE, TRUE,
     "exclude files that match " PATTERN_MATCH_DESCR_TAIL},
    {"include", OPT_INCLUDE, TRUE,
     "include only files that match" NEW_LINE "\t\t\t\t" PATTERN_MATCH_DESCR_TAIL},
    {"string", OPT_STRING, TRUE, "string to calculate " HASH_NAME " sum for"},
    {OPT_HASH_LONG, OPT_HASH, TRUE,
     HASH_NAME " hash to validate file or to find" NEW_LINE "\t\t\t\tinitial string (crack)"},
    {"dict", OPT_DICT, TRUE,
     "initial string's dictionary by default all" NEW_LINE "\t\t\t\tdigits, upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the string to" NEW_LINE "\t\t\t\trestore using option crack (c). 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the string to" NEW_LINE "\t\t\t\trestore  using option crack (c). " MAX_DEFAULT " by default"},
    {OPT_LIMIT_FULL, OPT_LIMIT, TRUE,
     "set the limit in bytes of the part of the file to" NEW_LINE "\t\t\t\tcalculate hash for. The whole file by default will be applied"},
    {OPT_OFFSET_FULL, OPT_OFFSET, TRUE,
     "set start position in the file to calculate hash from" NEW_LINE "\t\t\t\tzero by default"},
    {"search", OPT_SEARCH, TRUE, HASH_NAME " hash to search file that matches it"},
    {"save", OPT_SAVE, TRUE,
     "save files' " HASH_NAME " hashes into the file" NEW_LINE "\t\t\t\tspecified by full path"},
    {"crack", OPT_CRACK, FALSE,
     "crack " HASH_NAME " hash specified" NEW_LINE "\t\t\t\t(find initial string) by option " OPT_HASH_LONG
     " (m)"},
    {"lower", OPT_LOWER, FALSE, "whether to output sum using low case"},
    {"recursively", OPT_RECURSIVELY, FALSE, "scan directory recursively"},
    {"time", OPT_TIME, FALSE, "show " HASH_NAME " calculation time (false by default)"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_getopt_t* opt = NULL;
    int c = 0;
    const char* optarg = NULL;
    const char* file = NULL;
    const char* dir = NULL;
    TraverseContext dirContext = { 0 };
    DataContext dataCtx = { 0 };
    const char* includePattern = NULL;
    const char* excludePattern = NULL;
    const char* checkSum = NULL;
    const char* string = NULL;
    const char* dict = NULL;
    const char* fileToSave = NULL;
    int isCrack = FALSE;
    apr_byte_t digest[DIGESTSIZE];
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
        CrtPrintf("Couldn't initialize APR");
        NewLine();
        PrintError(status);
        return EXIT_FAILURE;
    }
    atexit(apr_terminate);
    apr_pool_create(&pool, NULL);
    apr_getopt_init(&opt, pool, argc, argv);

    dataCtx.IsPrintCalcTime = FALSE;
    dataCtx.IsPrintLowCase = FALSE;
    dataCtx.Offset = 0;
    dataCtx.Limit = MAXLONG64;
    dataCtx.PfnOutput = OutputToConsole;

    while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
        switch (c) {
            case OPT_HELP:
                PrintUsage();
                goto cleanup;
            case OPT_FILE:
                file = apr_pstrdup(pool, optarg);
                break;
            case OPT_DIR:
                dir = apr_pstrdup(pool, optarg);
                break;
            case OPT_SAVE:
                fileToSave = apr_pstrdup(pool, optarg);
                break;
            case OPT_HASH:
                checkSum = apr_pstrdup(pool, optarg);
                break;
            case OPT_SEARCH:
                dataCtx.HashToSearch = apr_pstrdup(pool, optarg);
                break;
            case OPT_STRING:
                string = apr_pstrdup(pool, optarg);
                break;
            case OPT_EXCLUDE:
                excludePattern = apr_pstrdup(pool, optarg);
                break;
            case OPT_INCLUDE:
                includePattern = apr_pstrdup(pool, optarg);
                break;
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
            case OPT_LIMIT:
                if (!sscanf(optarg, BIG_NUMBER_PARAM_FMT_STRING, &dataCtx.Limit)) {
                    CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_LIMIT_FULL, optarg);
                    goto cleanup;
                }
                break;
            case OPT_OFFSET:
                if (!sscanf(optarg, BIG_NUMBER_PARAM_FMT_STRING, &dataCtx.Offset)) {
                    CrtPrintf(INVALID_DIGIT_PARAMETER, OPT_OFFSET_FULL, optarg);
                    goto cleanup;
                }
                break;
            case OPT_LOWER:
                dataCtx.IsPrintLowCase = TRUE;
                break;
            case OPT_CRACK:
                isCrack = TRUE;
                break;
            case OPT_RECURSIVELY:
                dirContext.IsScanDirRecursively = TRUE;
                break;
            case OPT_TIME:
                dataCtx.IsPrintCalcTime = TRUE;
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
    if (dataCtx.HashToSearch && (dir == NULL)) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD
            "hash to search can be set" NEW_LINE "only if directory specified but it wasn't" NEW_LINE);
        goto cleanup;
    }
    if ((dirContext.ExcludePattern || dirContext.IncludePattern) && (dir == NULL)) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD
            "include or exclude patterns can be set" NEW_LINE "only if directory specified but it wasn't" NEW_LINE);
        goto cleanup;
    }
    if (dirContext.IsScanDirRecursively && (dir == NULL)) {
        PrintCopyright();
        CrtPrintf(
            INCOMPATIBLE_OPTIONS_HEAD
            "recursive scanning can be set" NEW_LINE "only if directory specified but it wasn't" NEW_LINE);
        goto cleanup;
    }

    if ((file != NULL) && (checkSum == NULL) && !isCrack &&
        CalculateFileHash(file, digest, dataCtx.IsPrintCalcTime, NULL, dataCtx.Limit,
                          dataCtx.Offset, dataCtx.PfnOutput, pool)) {
        OutputDigest(digest, &dataCtx, DIGESTSIZE, pool);
    }
    if ((string != NULL) && CalculateStringHash(string, digest)) {
        OutputDigest(digest, &dataCtx, DIGESTSIZE, pool);
    }
    if ((checkSum != NULL) && (file != NULL) &&
        CalculateFileHash(file, digest, dataCtx.IsPrintCalcTime, NULL, dataCtx.Limit,
                          dataCtx.Offset, dataCtx.PfnOutput, pool)) {
        CheckHash(digest, checkSum, &dataCtx);
    }
    if (dir != NULL) {
        if (fileToSave) {
            status = apr_file_open(&dataCtx.FileToSave,
                                   fileToSave,
                                   APR_CREATE | APR_TRUNCATE | APR_WRITE,
                                   APR_REG,
                                   pool);
            if (status != APR_SUCCESS) {
                PrintError(status);
                goto cleanup;
            }
        }
        dirContext.DataCtx = &dataCtx;
        dirContext.PfnFileHandler = CalculateFile;

        CompilePattern(includePattern, &dirContext.IncludePattern, pool);
        CompilePattern(excludePattern, &dirContext.ExcludePattern, pool);

        TraverseDirectory(HackRootPath(dir, pool), &dirContext, FilterByName, pool);
        if (fileToSave) {
            status = apr_file_close(dataCtx.FileToSave);
            if (status != APR_SUCCESS) {
                PrintError(status);
                goto cleanup;
            }
        }
    }
    if ((checkSum != NULL) && isCrack) {
        CrackHash(dict, checkSum, passmin, passmax, pool);
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

void* AllocateContext(apr_pool_t* pool)
{
    return apr_pcalloc(pool, sizeof(hash_context_t));
}

apr_size_t GetDigestSize()
{
    return DIGESTSIZE;
}

int ComparisonFailure(int result)
{
    return !result;
}

void ToDigest(const char* hash, apr_byte_t* digest)
{
    int i = 0;
    int to = MIN(DIGESTSIZE, strlen(hash) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(hash + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

int CompareHash(apr_byte_t* digest, const char* checkSum)
{
    apr_byte_t bytes[DIGESTSIZE];

    ToDigest(checkSum, bytes);
    return CompareDigests(bytes, digest);
}

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool)
{
    char* str = NULL;
    const char* str1234 = NULL;
    apr_byte_t digest[DIGESTSIZE];
    uint64_t attempts = 0;
    Time time = { 0 };
    double ratio = 0;

    CalculateStringHash("1234", digest);
    str1234 = HashToString(digest, FALSE, DIGESTSIZE, pool);
    
    StartTimer();

    BruteForce(1,
                atoi(MAX_DEFAULT),
                alphabet,
                str1234,
                &attempts,
                CreateDigest,
                pool);

    StopTimer();
    time = ReadElapsedTime();
    ratio = attempts / time.seconds;

    attempts = 0;
    StartTimer();

    // Empty string validation
    CalculateDigest(digest, NULL, 0);

    passmax = passmax ? passmax : atoi(MAX_DEFAULT);

    if (!CompareHash(digest, hash)) {
        int maxTimeMsgSz = 63;
        double maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
        Time maxTime = NormalizeTime(maxAttepts / ratio);
        char* maxTimeMsg = (char*)apr_pcalloc(pool, maxTimeMsgSz + 1);
        
        TimeToString(maxTime, maxTimeMsgSz, maxTimeMsg);
        CrtPrintf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, pool);
    } else {
        str = "Empty string";
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
        CrtPrintf("Initial string is: %s", str);
    } else {
        CrtPrintf("Nothing found");
    }
    NewLine();
}

void* CreateDigest(const char* hash, apr_pool_t* pool)
{
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(pool, DIGESTSIZE);
    ToDigest(hash, result);
    return result;
}

int CompareHashAttempt(void* hash, const char* pass, const uint32_t length)
{
    apr_byte_t attempt[DIGESTSIZE];
    
    CalculateDigest(attempt, pass, length);
    return CompareDigests(attempt, hash);
}

/*!
 * It's so ugly to improve performance
 */
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2)
{
    int i = 0;

    for (; i <= DIGESTSIZE - (DIGESTSIZE >> 2); i += 4) {
        if (digest1[i] != digest2[i]) {
            return FALSE;
        }
        if (digest1[i + 1] != digest2[i + 1]) {
            return FALSE;
        }
        if (digest1[i + 2] != digest2[i + 2]) {
            return FALSE;
        }
        if (digest1[i + 3] != digest2[i + 3]) {
            return FALSE;
        }
    }
    return TRUE;
}

int CalculateStringHash(const char* string, apr_byte_t* digest)
{
    apr_status_t status = APR_SUCCESS;

    if (string == NULL) {
        CrtPrintf("NULL string passed");
        NewLine();
        return FALSE;
    }
    status = CalculateDigest(digest, string, strlen(string));
    if (status != APR_SUCCESS) {
        CrtPrintf("Failed to calculate " HASH_NAME " of string: %s", string);
        NewLine();
        PrintError(status);
        return FALSE;
    }
    return TRUE;
}

#ifdef CALC_DIGEST_NOT_IMPLEMETED
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    hash_context_t context = { 0 };

    InitContext(&context);
    UpdateHash(&context, input, inputLen);
    FinalHash(digest, &context);
    return APR_SUCCESS;
}
#endif
