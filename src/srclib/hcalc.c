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

#ifdef WIN32
#include "DebugHelplers.h"
#endif

#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND  // 1 megabyte
#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s" NEW_LINE NEW_LINE
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define PATTERN_SEPARATOR ";"
#define PATH_ELT_SEPARATOR '\\'
#define NUMBER_PARAM_FMT_STRING "%lu"
#define BIG_NUMBER_PARAM_FMT_STRING "%llu"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number" NEW_LINE
#define FILE_INFO_COLUMN_SEPARATOR " | "
#define INCOMPATIBLE_OPTIONS_HEAD "Incompatible options: "

#define HASH_FILE_COLUMN_SEPARATOR "   "

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

#define COMPOSITE_PATTERN_INIT_SZ 8 // composite pattern array init size
#define SUBDIRS_ARRAY_INIT_SZ 16 // subdirectories array init size
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

static char* alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

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
        OutputDigest(digest, &dataCtx, pool);
    }
    if ((string != NULL) && CalculateStringHash(string, digest)) {
        OutputDigest(digest, &dataCtx, pool);
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

        TraverseDirectory(HackRootPath(dir, pool), &dirContext, pool);
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

const char* HackRootPath(const char* path, apr_pool_t* pool)
{
    size_t len = strlen(path);
    return path[len - 1] == ':' ? apr_pstrcat(pool, path, "\\", NULL) : path;
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

void OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_pool_t* pool)
{
    OutputContext output = { 0 };
    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    output.StringToPrint = HashToString(digest, ctx->IsPrintLowCase, pool);
    ctx->PfnOutput(&output);
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

const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_pool_t* pool)
{
    int i = 0;
    char* str = apr_pcalloc(pool, DIGESTSIZE * BYTE_CHARS_SIZE + 1); // iteration ponter
    char* result = str; // result pointer

    for (; i < DIGESTSIZE; ++i) {
        apr_snprintf(str, BYTE_CHARS_SIZE + 1, isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
        str += BYTE_CHARS_SIZE;
    }
    return result;
}

const char* CopySizeToString(uint64_t size, apr_pool_t* pool)
{
    size_t sz = 64;
    char* str = apr_pcalloc(pool, sz);
    SizeToString(size, sz, str);
    return str;
}

const char* CopyTimeToString(Time time, apr_pool_t* pool)
{
    size_t sz = 48;
    char* str = apr_pcalloc(pool, sz);
    TimeToString(time, sz, str);
    return str;
}

void CheckHash(apr_byte_t* digest, const char* checkSum, DataContext* ctx)
{
    OutputContext output = { 0 };
    output.StringToPrint = "File is ";
    ctx->PfnOutput(&output);
    output.StringToPrint = CompareHash(digest, checkSum) ? "valid" : "invalid";
    ctx->PfnOutput(&output);
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
    char* maxTimeMsg = NULL;
    int maxTimeMsgSz = 63;
    const char* str1234 = NULL;
    apr_byte_t digest[DIGESTSIZE];
    uint64_t attempts = 0;
    Time time = { 0 };
    double ratio = 0;
    double maxAttepts = 0;
    Time maxTime = { 0 };

    CalculateStringHash("1234", digest);
    str1234 = HashToString(digest, FALSE, pool);
    
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
        passmax = passmax ? passmax : atoi(MAX_DEFAULT);
        maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
        maxTime = NormalizeTime(maxAttepts / ratio);
        maxTimeMsg = (char*)apr_pcalloc(pool, maxTimeMsgSz + 1);
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

int CompareHashAttempt(void* hash, const char* pass, uint32_t length)
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

apr_status_t CalculateFile(const char* fullPathToFile, DataContext* ctx, apr_pool_t* pool)
{
    apr_byte_t digest[DIGESTSIZE];
    size_t len = 0;
    apr_status_t status = APR_SUCCESS;

    if (!CalculateFileHash(fullPathToFile, digest, ctx->IsPrintCalcTime,
                           ctx->HashToSearch, ctx->Limit, ctx->Offset, ctx->PfnOutput, pool)) {
        return status;
    }

    OutputDigest(digest, ctx, pool);

    if (!(ctx->FileToSave)) {
        return status;
    }
    apr_file_printf(ctx->FileToSave, HashToString(digest, ctx->IsPrintLowCase, pool));

    len = strlen(fullPathToFile);

    while (len > 0 && *(fullPathToFile + (len - 1)) != PATH_ELT_SEPARATOR) {
        --len;
    }

    apr_file_printf(ctx->FileToSave,
                    HASH_FILE_COLUMN_SEPARATOR "%s" NEW_LINE,
                    fullPathToFile + len);
    return status;
}

void TraverseDirectory(const char* dir, TraverseContext* ctx, apr_pool_t* pool)
{
    apr_finfo_t info = { 0 };
    apr_dir_t* d = NULL;
    apr_status_t status = APR_SUCCESS;
    char* fullPath = NULL; // Full path to file or subdirectory
    apr_pool_t* iterPool = NULL;
    apr_array_header_t* subdirs = NULL;
    OutputContext output = { 0 };

    status = apr_dir_open(&d, dir, pool);
    if (status != APR_SUCCESS) {
        output.StringToPrint = FromUtf8ToAnsi(dir, pool);
        output.IsPrintSeparator = TRUE;
        ((DataContext*)ctx->DataCtx)->PfnOutput(&output);

        OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
        return;
    }

    if (ctx->IsScanDirRecursively) {
        subdirs = apr_array_make(pool, SUBDIRS_ARRAY_INIT_SZ, sizeof(const char*));
    }

    apr_pool_create(&iterPool, pool);
    for (;;) {
        apr_pool_clear(iterPool);  // cleanup file allocated memory
        status = apr_dir_read(&info, APR_FINFO_NAME | APR_FINFO_MIN, d);
        if (APR_STATUS_IS_ENOENT(status)) {
            break;
        }
        if (info.name == NULL) { // to avoid access violation
            OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
            continue;
        }
        // Subdirectory handling code
        if ((info.filetype == APR_DIR) && ctx->IsScanDirRecursively) {
            // skip current and parent dir
            if (((info.name[0] == '.') && (info.name[1] == '\0'))
                || ((info.name[0] == '.') && (info.name[1] == '.') && (info.name[2] == '\0'))) {
                continue;
            }

            status = apr_filepath_merge(&fullPath,
                                        dir,
                                        info.name,
                                        APR_FILEPATH_NATIVE,
                                        pool); // IMPORTANT: so as not to use strdup
            if (status != APR_SUCCESS) {
                OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
                continue;
            }
            *(const char**)apr_array_push(subdirs) = fullPath;
        } // End subdirectory handling code

        if ((status != APR_SUCCESS) || (info.filetype != APR_REG)) {
            continue;
        }

        if (!MatchToCompositePattern(info.name, ctx->IncludePattern)) {
            continue;
        }
        // IMPORTANT: check pointer here otherwise the logic will fail
        if (ctx->ExcludePattern &&
            MatchToCompositePattern(info.name, ctx->ExcludePattern)) {
            continue;
        }

        status = apr_filepath_merge(&fullPath,
                                    dir,
                                    info.name,
                                    APR_FILEPATH_NATIVE,
                                    iterPool);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
            continue;
        }

        if (ctx->PfnFileHandler(fullPath, ctx->DataCtx, iterPool) != APR_SUCCESS) {
            continue; // or break if you want to interrupt in case of any file handling error
        }
    }

    status = apr_dir_close(d);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, ((DataContext*)ctx->DataCtx)->PfnOutput, pool);
    }

    // scan subdirectories found
    if (ctx->IsScanDirRecursively) {
        int i = 0;
        for (; i < subdirs->nelts; ++i) {
            const char* path = ((const char**)subdirs->elts)[i];
            apr_pool_clear(iterPool);
            TraverseDirectory(path, ctx, iterPool);
        }
    }

    apr_pool_destroy(iterPool);
}


void CompilePattern(const char* pattern, apr_array_header_t** newpattern, apr_pool_t* pool)
{
    char* parts = NULL;
    char* last = NULL;
    char* p = NULL;

    if (!pattern) {
        return; // important
    }

    *newpattern = apr_array_make(pool, COMPOSITE_PATTERN_INIT_SZ, sizeof(const char*));

    parts = apr_pstrdup(pool, pattern);    /* strtok wants non-const data */
    p = apr_strtok(parts, PATTERN_SEPARATOR, &last);
    while (p) {
        *(const char**)apr_array_push(*newpattern) = p;
        p = apr_strtok(NULL, PATTERN_SEPARATOR, &last);
    }
}

int MatchToCompositePattern(const char* str, apr_array_header_t* pattern)
{
    int i = 0;

    if (!pattern) {
        return TRUE;    // important
    }
    if (!str) {
        return FALSE;   // important
    }

    for (; i < pattern->nelts; ++i) {
        const char* p = ((const char**)pattern->elts)[i];
        if (apr_fnmatch(p, str, APR_FNM_CASE_BLIND) == APR_SUCCESS) {
            return TRUE;
        }
    }

    return FALSE;
}

int CalculateFileHash(const char* filePath,
                      apr_byte_t* digest,
                      int         isPrintCalcTime,
                      const char* hashToSearch,
                      apr_off_t   limit,
                      apr_off_t   offset,
                      void        (* PfnOutput)(OutputContext* ctx),
                      apr_pool_t* pool)
{
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = { 0 };
    hash_context_t context = { 0 };
    apr_status_t status = APR_SUCCESS;
    int result = TRUE;
    apr_off_t pageSize = 0;
    apr_off_t filePartSize = 0;
    apr_off_t startOffset = offset;
    apr_mmap_t* mmap = NULL;
    char* fileAnsi = NULL;
    int isZeroSearchHash = FALSE;
    apr_byte_t digestToCompare[DIGESTSIZE];
    OutputContext output = { 0 };

    fileAnsi = FromUtf8ToAnsi(filePath, pool);
    if (!hashToSearch) {
        output.StringToPrint = fileAnsi == NULL ? filePath : fileAnsi;
        output.IsPrintSeparator = TRUE;
        PfnOutput(&output);
    }
    StartTimer();

    status = apr_file_open(&fileHandle, filePath, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        return FALSE;
    }
    status = InitContext(&context);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        result = FALSE;
        goto cleanup;
    }

    status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, fileHandle);

    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
        result = FALSE;
        goto cleanup;
    }

    if (!hashToSearch) {
        output.IsPrintSeparator = TRUE;
        output.IsFinishLine = FALSE;
        output.StringToPrint = CopySizeToString(info.size, pool);
        PfnOutput(&output);
    }

    if (hashToSearch) {
        ToDigest(hashToSearch, digestToCompare);
        status = CalculateDigest(digest, NULL, 0);
        if (CompareDigests(digest, digestToCompare)) { // Empty file optimization
            isZeroSearchHash = TRUE;
            goto endtiming;
        }
    }

    filePartSize = MIN(limit, info.size);

    if (filePartSize > FILE_BIG_BUFFER_SIZE) {
        pageSize = FILE_BIG_BUFFER_SIZE;
    } else if (filePartSize == 0) {
        status = CalculateDigest(digest, NULL, 0);
        goto endtiming;
    } else {
        pageSize = filePartSize;
    }

    if (offset >= info.size) {
        output.IsFinishLine = TRUE;
        output.IsPrintSeparator = FALSE;
        output.StringToPrint = "Offset is greater then file size";
        PfnOutput(&output);
        result = FALSE;
        goto endtiming;
    }

    do {
        apr_status_t hashCalcStatus = APR_SUCCESS;
        apr_size_t size = (apr_size_t)MIN(pageSize, (filePartSize + startOffset) - offset);

        if (size + offset > info.size) {
            size = info.size - offset;
        }

        status =
            apr_mmap_create(&mmap, fileHandle, offset, size, APR_MMAP_READ, pool);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, PfnOutput, pool);
            result = FALSE;
            mmap = NULL;
            goto cleanup;
        }
        hashCalcStatus = UpdateHash(&context, mmap->mm, mmap->size);
        if (hashCalcStatus != APR_SUCCESS) {
            OutputErrorMessage(hashCalcStatus, PfnOutput, pool);
            result = FALSE;
            goto cleanup;
        }
        offset += mmap->size;
        status = apr_mmap_delete(mmap);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, PfnOutput, pool);
            mmap = NULL;
            result = FALSE;
            goto cleanup;
        }
        mmap = NULL;
    } while (offset < filePartSize + startOffset && offset < info.size);
    status = FinalHash(digest, &context);
endtiming:
    StopTimer();

    if (!hashToSearch) {
        goto printtime;
    }

    result = FALSE;
    if (!((!isZeroSearchHash &&
           CompareDigests(digest, digestToCompare)) || (isZeroSearchHash && (info.size == 0) ))) {
        goto printtime;
    }

    output.IsFinishLine = FALSE;
    output.IsPrintSeparator = TRUE;

    // file name
    output.StringToPrint = fileAnsi == NULL ? filePath : fileAnsi;
    PfnOutput(&output);

    // file size
    output.StringToPrint = CopySizeToString(info.size, pool);

    if (isPrintCalcTime) {
        output.IsPrintSeparator = TRUE;
        PfnOutput(&output); // file size output before time

        // time
        output.StringToPrint = CopyTimeToString(ReadElapsedTime(), pool);
    }
    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    PfnOutput(&output); // file size or time output

printtime:
    if (isPrintCalcTime & !hashToSearch) {
        // time
        output.StringToPrint = CopyTimeToString(ReadElapsedTime(), pool);
        output.IsFinishLine = FALSE;
        output.IsPrintSeparator = TRUE;
        PfnOutput(&output);
    }
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
cleanup:
    if (mmap != NULL) {
        status = apr_mmap_delete(mmap);
        mmap = NULL;
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, PfnOutput, pool);
        }
    }
    status = apr_file_close(fileHandle);
    if (status != APR_SUCCESS) {
        OutputErrorMessage(status, PfnOutput, pool);
    }
    return result;
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
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen)
{
    hash_context_t context = { 0 };

    InitContext(&context);
    UpdateHash(&context, input, inputLen);
    FinalHash(digest, &context);
    return APR_SUCCESS;
}
#endif

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
