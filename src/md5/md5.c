/*
 * Copyright 2009 Alexander Egorov
 */

#include "targetver.h"

#include <stdio.h>
#include <locale.h>
#include <time.h>

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_md5.h"
#include "apr_file_io.h"
#include "apr_mmap.h"
#include "apr_fnmatch.h"

#include "pglib.h"
#ifdef WIN32
#include "DebugHelplers.h"
#endif

#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND  // 1 megabyte
#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s\n\n"
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define MIN(x, y) ((x)<(y) ? (x):(y))
#define PATTERN_SEPARATOR ";"
#define NUMBER_PARAM_FMT_STRING "%d"

#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number\n"
#define ALLOCATION_FAILURE_MESSAGE "Failed to allocate %i bytes in: %s:%d\n"
#define FILE_INFO_COLUMN_SEPARATOR " | "

#define OPT_FILE 'f'
#define OPT_DIR 'd'
#define OPT_EXCLUDE 'e'
#define OPT_INCLUDE 'i'
#define OPT_STRING 's'
#define OPT_MD5 'm'
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

static struct apr_getopt_option_t options[] = {
    {"file", OPT_FILE, TRUE, "input full file path to calculate MD5 sum for"},
    {"dir", OPT_DIR, TRUE, "full path to dir to calculate\n\t\t\t\tMD5 of all content"},
    {"exclude", OPT_EXCLUDE, TRUE,
     "exclude files that match the pattern specified\n\t\t\t\tit's possible to use several patterns\n\t\t\t\tseparated by ;"},
    {"include", OPT_INCLUDE, TRUE,
     "include only files that match\n\t\t\t\tthe pattern specified\n\t\t\t\tit's possible to use several patterns\n\t\t\t\tseparated by ;"},
    {"string", OPT_STRING, TRUE, "string to calculate MD5 sum for"},
    {"md5", OPT_MD5, TRUE, "MD5 hash to validate file or to find\n\t\t\t\tinitial string (crack)"},
    {"dict", OPT_DICT, TRUE,
     "initial string's dictionary by default all\n\t\t\t\tdigits and upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the string to\n\t\t\t\trestore using option crack (c). 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the string to\n\t\t\t\trestore  using option crack (c).\n\t\t\t\tThe length of the dictionary by default"},
    {"search", OPT_SEARCH, TRUE, "MD5 hash to search file that matches it"},
    {"crack", OPT_CRACK, FALSE, "crack MD5 hash specified\n\t\t\t\t(find initial string) by option md5 (m)"},
    {"lower", OPT_LOWER, FALSE, "whether to output sum using low case"},
    {"recursively", OPT_RECURSIVELY, FALSE, "scan directory recursively"},
    {"time", OPT_TIME, FALSE, "show MD5 calculation time (false by default)"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

static char *alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Forward declarations
void PrintUsage();
void PrintCopyright();
int CalculateFileMd5(apr_pool_t * pool, const char *file, apr_byte_t * digest, int isPrintCalcTime,
                     const char *pHashToSearch);
void CalculateDirContentMd5(apr_pool_t * pool, const char *dir, int isPrintLowCase, int isScanDirRecursively,
                            int isPrintCalcTime, const char *pExcludePattern, const char *pIncludePattern,
                            const char *pHashToSearch);
int CalculateStringMd5(const char *string, apr_byte_t * digest);
void PrintMd5(apr_byte_t * digest, int isPrintLowCase);
void CheckMd5(apr_byte_t * digest, const char *pCheckSum);
int CompareMd5(apr_byte_t * digest, const char *pCheckSum);
void PrintError(apr_status_t status);
void CrackMd5(apr_pool_t * pool, const char *pDict, const char *pCheckSum, int passmin, int passmax);
int CompareDigests(apr_byte_t * digest1, apr_byte_t * digest2);
void ToDigest(const char *pCheckSum, apr_byte_t * digest);
int MatchToCompositePattern(apr_pool_t * pool, const char *pStr, const char *pPattern);
char *BruteForce(int passmin, int passmax, apr_pool_t * pool, const char *pDict, apr_byte_t * desired,
                 unsigned long long *attemptsCount);
int MakeAttempt(int pos, int length, const char *pDict, int *indexes, char *pass, apr_byte_t * desired,
                       unsigned long long *attemptsCount, int maxIndex);

/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char *FromUtf8ToAnsi(const char *from, apr_pool_t * pool);
struct Version ReadVersion(apr_pool_t * pool, const char *pFile);

#ifdef WIN32
/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char *DecodeUtf8Ansi(const char *from, apr_pool_t * pool, UINT fromCodePage, UINT toCodePage);
#endif

int main(int argc, const char *const argv[])
{
    apr_pool_t *pool = NULL;
    apr_getopt_t *opt = NULL;
    int c = 0;
    const char *optarg = NULL;
    const char *pFile = NULL;
    const char *pDir = NULL;
    const char *pCheckSum = NULL;
    const char *pHashToSearch = NULL;
    const char *pString = NULL;
    const char *pExcludePattern = NULL;
    const char *pIncludePattern = NULL;
    const char *pDict = NULL;
    int isPrintLowCase = FALSE;
    int isScanDirRecursively = FALSE;
    int isPrintCalcTime = FALSE;
    int isCrack = FALSE;
    apr_byte_t digest[APR_MD5_DIGESTSIZE];
    apr_status_t status = APR_SUCCESS;
    int passmin = 1;    // important!
    int passmax = 0;

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

    if (argc < 2) {
        PrintUsage();
        goto cleanup;
    }

    while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
        switch (c) {
            case OPT_HELP:
                PrintUsage();
                goto cleanup;
            case OPT_FILE:
                pFile = apr_pstrdup(pool, optarg);
                break;
            case OPT_DIR:
                pDir = apr_pstrdup(pool, optarg);
                break;
            case OPT_MD5:
                pCheckSum = apr_pstrdup(pool, optarg);
                break;
            case OPT_SEARCH:
                pHashToSearch = apr_pstrdup(pool, optarg);
                break;
            case OPT_STRING:
                pString = apr_pstrdup(pool, optarg);
                break;
            case OPT_EXCLUDE:
                pExcludePattern = apr_pstrdup(pool, optarg);
                break;
            case OPT_INCLUDE:
                pIncludePattern = apr_pstrdup(pool, optarg);
                break;
            case OPT_DICT:
                pDict = apr_pstrdup(pool, optarg);
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
            case OPT_LOWER:
                isPrintLowCase = TRUE;
                break;
            case OPT_CRACK:
                isCrack = TRUE;
                break;
            case OPT_RECURSIVELY:
                isScanDirRecursively = TRUE;
                break;
            case OPT_TIME:
                isPrintCalcTime = TRUE;
                break;
        }
    }

    if (status != APR_EOF) {
        PrintUsage(pool);
        goto cleanup;
    }
    if (pDict == NULL) {
        pDict = alphabet;
    }

    if (pFile != NULL && pCheckSum == NULL && !isCrack && CalculateFileMd5(pool, pFile, digest, isPrintCalcTime, NULL)) {
        PrintMd5(digest, isPrintLowCase);
    }
    if (pString != NULL && CalculateStringMd5(pString, digest)) {
        PrintMd5(digest, isPrintLowCase);
    }
    if (pCheckSum != NULL && pFile != NULL && CalculateFileMd5(pool, pFile, digest, isPrintCalcTime, NULL)) {
        CheckMd5(digest, pCheckSum);
    }
    if (pDir != NULL) {
        CalculateDirContentMd5(pool, pDir, isPrintLowCase, isScanDirRecursively, isPrintCalcTime, pExcludePattern,
                               pIncludePattern, pHashToSearch);
    }
    if (pCheckSum != NULL && isCrack) {
        CrackMd5(pool, pDict, pCheckSum, passmin, passmax);
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

void PrintUsage()
{
    int i = 0;
    PrintCopyright();
    CrtPrintf("usage: md5 [OPTION] ...\n\nOptions:\n\n");
    for (; i < sizeof(options) / sizeof(apr_getopt_option_t); ++i) {
        CrtPrintf(options[i].has_arg ? HLP_ARG : HLP_NO_ARG,
                  (char)options[i].optch, options[i].name, options[i].description);
    }
}

void PrintCopyright()
{
    CrtPrintf("\nMD5 Calculator %s\nCopyright (C) 2009-2010 Alexander Egorov. All rights reserved.\n\n", PRODUCT_VERSION);
}

void PrintMd5(apr_byte_t * digest, int isPrintLowCase)
{
    int i = 0;
    for (; i < APR_MD5_DIGESTSIZE; ++i) {
        CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
    }
    CrtPrintf("\n");
}

void CheckMd5(apr_byte_t * digest, const char *pCheckSum)
{
    CrtPrintf("File is %s!\n", CompareMd5(digest, pCheckSum) ? "valid" : "invalid");
}

void ToDigest(const char *pCheckSum, apr_byte_t * digest)
{
    int i = 0;
    int to = MIN(APR_MD5_DIGESTSIZE, strlen(pCheckSum) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t) htoi(pCheckSum + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

int CompareMd5(apr_byte_t * digest, const char *pCheckSum)
{
    apr_byte_t bytes[APR_MD5_DIGESTSIZE];

    ToDigest(pCheckSum, bytes);
    return CompareDigests(bytes, digest);
}

void CrackMd5(apr_pool_t * pool, const char *pDict, const char *pCheckSum, int passmin, int passmax)
{
    char *pStr = NULL;
    apr_byte_t digest[APR_MD5_DIGESTSIZE];
    unsigned long long attemptsCount = 0;

    double span = 0;
    Time time = { 0 };

#ifdef WIN32
    LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER time1 = { 0 };
    LARGE_INTEGER time2 = { 0 };

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time1);

#else
    clock_t c0 = 0;
    clock_t c1 = 0;

    c0 = clock();
#endif

    // Empty string validation
    apr_md5(digest, NULL, 0);
    if (CompareMd5(digest, pCheckSum)) {
        pStr = "Empty string";
        goto exit;
    }

    ToDigest(pCheckSum, digest);

    pStr = BruteForce(passmin, passmax ? passmax : strlen(pDict), pool, pDict, digest, &attemptsCount);

exit:
#ifdef WIN32
    QueryPerformanceCounter(&time2);
    span = (double)(time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
#else
    c1 = clock();
    span = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
#endif
    time = NormalizeTime(span);
    CrtPrintf("\nAttempts: %llu Time " FULL_TIME_FMT, attemptsCount, time.hours, time.minutes, time.seconds);
    CrtPrintf("\n");
    if (pStr != NULL) {
        CrtPrintf("Initial string is: %s \n", pStr);
    } else {
        CrtPrintf("Nothing found\n");
    }
}

int MakeAttempt(int pos, int length, const char *pDict, int *indexes, char *pass, apr_byte_t * desired,
                       unsigned long long *attemptsCount, int maxIndex)
{
    int i = 0;
    int j = 0;
    apr_byte_t attempt[APR_MD5_DIGESTSIZE];

    for (; i <= maxIndex; ++i) {
        indexes[pos] = i;

        if (pos == length - 1) {
            for (j = 0; j < length; ++j) {
                pass[j] = pDict[indexes[j]];
            }
            ++*attemptsCount;
            apr_md5(attempt, pass, length);
            if (CompareDigests(attempt, desired)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, length, pDict, indexes, pass, desired, attemptsCount, maxIndex)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

char *BruteForce(int passmin, int passmax, apr_pool_t * pool, const char *pDict, apr_byte_t * desired,
                 unsigned long long *attemptsCount)
{
    char *pass = (char *)apr_pcalloc(pool, passmax + 1);
    int *indexes = (int *)apr_pcalloc(pool, passmax * sizeof(int));
    int passLength = passmin;
    int maxIndex = strlen(pDict) - 1;

    for (; passLength <= passmax; ++passLength) {
        if (MakeAttempt(0, passLength, pDict, indexes, pass, desired, attemptsCount, maxIndex)) {
            return pass;
        }
    }
    return NULL;
}

/**
* It's so ugly to improve performance
*/
int CompareDigests(apr_byte_t * digest1, apr_byte_t * digest2)
{
    int i = 0;

    for (; i <= APR_MD5_DIGESTSIZE - (APR_MD5_DIGESTSIZE >> 2); i += 4) {
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

void CalculateDirContentMd5(apr_pool_t * pool,
                            const char *dir,
                            int isPrintLowCase,
                            int isScanDirRecursively,
                            int isPrintCalcTime,
                            const char *pExcludePattern, const char *pIncludePattern, const char *pHashToSearch)
{
    apr_finfo_t info = { 0 };
    apr_dir_t *d = NULL;
    apr_status_t status = APR_SUCCESS;
    apr_byte_t digest[APR_MD5_DIGESTSIZE];
    char *fullPathToFile = NULL;
    apr_pool_t *filePool = NULL;
    apr_pool_t *dirPool = NULL;

    apr_pool_create(&filePool, pool);
    apr_pool_create(&dirPool, pool);

    status = apr_dir_open(&d, dir, dirPool);
    if (status != APR_SUCCESS) {
        PrintError(status);
        return;
    }

    for (;;) {
        apr_pool_clear(filePool);   // cleanup file allocated memory
        status = apr_dir_read(&info, APR_FINFO_NAME | APR_FINFO_MIN, d);
        if (APR_STATUS_IS_ENOENT(status)) {
            break;
        }
        if (info.filetype == APR_DIR && isScanDirRecursively) {
            if ((info.name[0] == '.' && info.name[1] == '\0')
                || (info.name[0] == '.' && info.name[1] == '.' && info.name[2] == '\0')) {
                continue;
            }

            status = apr_filepath_merge(&fullPathToFile, dir, info.name, APR_FILEPATH_NATIVE, filePool);
            if (status != APR_SUCCESS) {
                PrintError(status);
                goto cleanup;
            }
            CalculateDirContentMd5(pool, fullPathToFile, isPrintLowCase, isScanDirRecursively, isPrintCalcTime,
                                   pExcludePattern, pIncludePattern, pHashToSearch);
        }
        if (status != APR_SUCCESS || info.filetype != APR_REG) {
            continue;
        }

        if (!MatchToCompositePattern(filePool, info.name, pIncludePattern)) {
            continue;
        }
        // IMPORTANT: check pointer here otherwise the logic will fail
        if (pExcludePattern && MatchToCompositePattern(filePool, info.name, pExcludePattern)) {
            continue;
        }

        status = apr_filepath_merge(&fullPathToFile, dir, info.name, APR_FILEPATH_NATIVE, filePool);
        if (status != APR_SUCCESS) {
            PrintError(status);
            goto cleanup;
        }

        if (CalculateFileMd5(filePool, fullPathToFile, digest, isPrintCalcTime, pHashToSearch)) {
            PrintMd5(digest, isPrintLowCase);
        }
    }

cleanup:
    apr_pool_destroy(dirPool);
    apr_pool_destroy(filePool);
    status = apr_dir_close(d);
    if (status != APR_SUCCESS) {
        PrintError(status);
    }
}

int MatchToCompositePattern(apr_pool_t * pool, const char *pStr, const char *pPattern)
{
    char *parts = NULL;
    char *last = NULL;
    char *p = NULL;

    if (!pPattern) {
        return TRUE;    // important
    }
    if (!pStr) {
        return FALSE;   // important
    }

    parts = apr_pstrdup(pool, pPattern);    /* strtok wants non-const data */
    p = apr_strtok(parts, PATTERN_SEPARATOR, &last);
    while (p) {
        if (apr_fnmatch(p, pStr, APR_FNM_CASE_BLIND) == APR_SUCCESS) {
            return TRUE;
        }
        p = apr_strtok(NULL, PATTERN_SEPARATOR, &last);
    }
    return FALSE;
}

int CalculateFileMd5(apr_pool_t * pool, const char *pFile, apr_byte_t * digest, int isPrintCalcTime,
                     const char *pHashToSearch)
{
    apr_file_t *file = NULL;
    apr_finfo_t info = { 0 };
    apr_md5_ctx_t context = { 0 };
    apr_status_t status = APR_SUCCESS;
    apr_status_t md5CalcStatus = APR_SUCCESS;
    int result = TRUE;
    apr_off_t strSize = 0;
    apr_mmap_t *mmap = NULL;
    apr_off_t offset = 0;
    char *pFileAnsi = NULL;
    apr_byte_t digestToCompare[APR_MD5_DIGESTSIZE];

    double span = 0;
#ifdef WIN32
    LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER time1 = { 0 };
    LARGE_INTEGER time2 = { 0 };
#else
    clock_t c0 = 0;
    clock_t c1 = 0;
#endif

    pFileAnsi = FromUtf8ToAnsi(pFile, pool);
    if (!pHashToSearch) {
        CrtPrintf("%s", pFileAnsi == NULL ? pFile : pFileAnsi);
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
#ifdef WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time1);
#else
    c0 = clock();
#endif

    status = apr_file_open(&file, pFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        PrintError(status);
        return FALSE;
    }
    status = apr_md5_init(&context);
    if (status != APR_SUCCESS) {
        PrintError(status);
        result = FALSE;
        goto cleanup;
    }

    status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, file);

    if (status != APR_SUCCESS) {
        PrintError(status);
        result = FALSE;
        goto cleanup;
    }

    if (!pHashToSearch) {
        PrintSize(info.size);
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }

    if (info.size > FILE_BIG_BUFFER_SIZE) {
        strSize = FILE_BIG_BUFFER_SIZE;
    } else if (info.size == 0) {
        status = apr_md5(digest, NULL, 0);
        goto endtiming;
    } else {
        strSize = info.size;
    }

    do {
        status =
            apr_mmap_create(&mmap, file, offset, (apr_size_t) MIN(strSize, info.size - offset), APR_MMAP_READ, pool);
        if (status != APR_SUCCESS) {
            PrintError(status);
            result = FALSE;
            mmap = NULL;
            goto cleanup;
        }
        md5CalcStatus = apr_md5_update(&context, mmap->mm, mmap->size);
        if (md5CalcStatus != APR_SUCCESS) {
            PrintError(md5CalcStatus);
            result = FALSE;
            goto cleanup;
        }
        offset += mmap->size;
        status = apr_mmap_delete(mmap);
        if (status != APR_SUCCESS) {
            PrintError(status);
            mmap = NULL;
            result = FALSE;
            goto cleanup;
        }
        mmap = NULL;
    } while (offset < info.size);
    status = apr_md5_final(digest, &context);
endtiming:
#ifdef WIN32
    QueryPerformanceCounter(&time2);
    span = (double)(time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
#else
    c1 = clock();
    span = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
#endif

    if (pHashToSearch) {
        ToDigest(pHashToSearch, digestToCompare);
        if (CompareDigests(digest, digestToCompare)) {
            CrtPrintf("%s", pFileAnsi == NULL ? pFile : pFileAnsi);
            CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
            PrintSize(info.size);
            if (isPrintCalcTime) {
                CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
                PrintTime(span);
            }
            CrtPrintf("\n");
        }
        result = FALSE;
    }

    if (isPrintCalcTime & !pHashToSearch) {
        PrintTime(span);
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if (status != APR_SUCCESS) {
        PrintError(status);
    }
cleanup:
    if (mmap != NULL) {
        status = apr_mmap_delete(mmap);
        mmap = NULL;
        if (status != APR_SUCCESS) {
            PrintError(status);
        }
    }
    status = apr_file_close(file);
    if (status != APR_SUCCESS) {
        PrintError(status);
    }
    return result;
}

int CalculateStringMd5(const char *pString, apr_byte_t * digest)
{
    apr_status_t status = APR_SUCCESS;

    if (pString == NULL) {
        CrtPrintf("NULL string passed\n");
        return FALSE;
    }
    status = apr_md5(digest, pString, strlen(pString));
    if (status != APR_SUCCESS) {
        CrtPrintf("Failed to calculate MD5 of string: %s \n", pString);
        PrintError(status);
        return FALSE;
    }
    return TRUE;
}

/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char *FromUtf8ToAnsi(const char *from, apr_pool_t * pool)
{
#ifdef WIN32
    return DecodeUtf8Ansi(from, pool, CP_UTF8, CP_ACP);
#else
    return NULL;
#endif
}

#ifdef WIN32
/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char *DecodeUtf8Ansi(const char *from, apr_pool_t * pool, UINT fromCodePage, UINT toCodePage)
{
    int lengthWide = 0;
    int lengthAnsi = 0;
    size_t cbFrom = 0;
    wchar_t *wideStr = NULL;
    char *ansiStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1;  // IMPORTANT!!! including null terminator

    lengthWide = MultiByteToWideChar(fromCodePage, 0, from, cbFrom, NULL, 0);   // including null terminator
    wideBufferSize = sizeof(wchar_t) * lengthWide;
    wideStr = (wchar_t *) apr_pcalloc(pool, wideBufferSize);
    if (wideStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(fromCodePage, 0, from, cbFrom, wideStr, lengthWide);

    lengthAnsi = WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, 0, NULL, NULL);   // null terminator included
    ansiStr = (char *)apr_pcalloc(pool, lengthAnsi);

    if (ansiStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, lengthAnsi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, lengthAnsi, NULL, NULL);

    return ansiStr;
}
#endif
