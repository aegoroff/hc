/*!
 * \brief   The file contains common hash calculator implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#include "targetver.h"

#include <stdio.h>
#include <locale.h>

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_file_io.h"
#include "apr_mmap.h"
#include "apr_fnmatch.h"

#include "pglib.h"
#ifdef WIN32
#include "DebugHelplers.h"
#endif
#include "implementation.h"


#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND  // 1 megabyte
#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"

#define HLP_OPT_BEGIN "  -%c [ --%s ] "
#define HLP_OPT_END "\t\t%s\n\n"
#define HLP_ARG HLP_OPT_BEGIN "arg" HLP_OPT_END
#define HLP_NO_ARG HLP_OPT_BEGIN HLP_OPT_END

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define PATTERN_SEPARATOR ";"
#define NUMBER_PARAM_FMT_STRING "%lu"

#define ALLOCATION_FAILURE_MESSAGE ALLOCATION_FAIL_FMT " in: %s:%d\n"
#define INVALID_DIGIT_PARAMETER "Invalid parameter --%s %s. Must be number\n"
#define FILE_INFO_COLUMN_SEPARATOR " | "

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

static struct apr_getopt_option_t options[] = {
    {"file", OPT_FILE, TRUE, "input full file path to calculate " HASH_NAME " sum for"},
    {"dir", OPT_DIR, TRUE, "full path to dir to calculate\n\t\t\t\t" HASH_NAME " of all content"},
    {"exclude", OPT_EXCLUDE, TRUE,
     "exclude files that match the pattern specified\n\t\t\t\tit's possible to use several patterns\n\t\t\t\tseparated by ;"},
    {"include", OPT_INCLUDE, TRUE,
     "include only files that match\n\t\t\t\tthe pattern specified\n\t\t\t\tit's possible to use several patterns\n\t\t\t\tseparated by ;"},
    {"string", OPT_STRING, TRUE, "string to calculate " HASH_NAME " sum for"},
    {OPT_HASH_LONG, OPT_HASH, TRUE, HASH_NAME " hash to validate file or to find\n\t\t\t\tinitial string (crack)"},
    {"dict", OPT_DICT, TRUE,
     "initial string's dictionary by default all\n\t\t\t\tdigits and upper and lower case latin symbols"},
    {OPT_MIN_FULL, OPT_MIN, TRUE,
     "set minimum length of the string to\n\t\t\t\trestore using option crack (c). 1 by default"},
    {OPT_MAX_FULL, OPT_MAX, TRUE,
     "set maximum length of the string to\n\t\t\t\trestore  using option crack (c).\n\t\t\t\tThe length of the dictionary by default"},
    {"search", OPT_SEARCH, TRUE, HASH_NAME " hash to search file that matches it"},
    {"crack", OPT_CRACK, FALSE,
     "crack " HASH_NAME " hash specified\n\t\t\t\t(find initial string) by option " OPT_HASH_LONG " (m)"},
    {"lower", OPT_LOWER, FALSE, "whether to output sum using low case"},
    {"recursively", OPT_RECURSIVELY, FALSE, "scan directory recursively"},
    {"time", OPT_TIME, FALSE, "show " HASH_NAME " calculation time (false by default)"},
    {"help", OPT_HELP, FALSE, "show help message"}
};

static char* alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Forward declarations
void PrintUsage(void);
void PrintCopyright(void);
int  CalculateFileHash(apr_pool_t* pool, const char* file, apr_byte_t* digest, int isPrintCalcTime,
                      const char* pHashToSearch);
void CalculateDirContentHash(apr_pool_t* pool,
                            const char* dir,
                            int         isPrintLowCase,
                            int         isScanDirRecursively,
                            int         isPrintCalcTime,
                            const char* pExcludePattern,
                            const char* pIncludePattern,
                            const char* pHashToSearch);
int  CalculateStringHash(const char* string, apr_byte_t* digest);
void PrintHash(apr_byte_t* digest, int isPrintLowCase);
void CheckHash(apr_byte_t* digest, const char* pCheckSum);
int  CompareHash(apr_byte_t* digest, const char* pCheckSum);
void PrintError(apr_status_t status);
void CrackHash(apr_pool_t*  pool,
              const char*  pDict,
              const char*  pCheckSum,
              unsigned int passmin,
              unsigned int passmax);
int   CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);
void  ToDigest(const char* pCheckSum, apr_byte_t* digest);

/*!
 * \brief Try to match the string to the given pattern using apr_fnmatch function.
 *        Matching is case insensitive
 * PATTERN: Backslash followed by any character, including another
 *          backslash.<br/>
 * MATCHES: That character exactly.
 * 
 * <p>
 * PATTERN: ?<br/>
 * MATCHES: Any single character.
 * </p>
 * 
 * <p>
 * PATTERN: *<br/>
 * MATCHES: Any sequence of zero or more characters. (Note that multiple
 *          *s in a row are equivalent to one.)
 * 
 * PATTERN: Any character other than \?*[ or a \ at the end of the pattern<br/>
 * MATCHES: That character exactly. (Case sensitive.)
 * 
 * PATTERN: [ followed by a class description followed by ]<br/>
 * MATCHES: A single character described by the class description.
 *          (Never matches, if the class description reaches until the
 *          end of the string without a ].) If the first character of
 *          the class description is ^ or !, the sense of the description
 *          is reversed.  The rest of the class description is a list of
 *          single characters or pairs of characters separated by -. Any
 *          of those characters can have a backslash in front of them,
 *          which is ignored; this lets you use the characters ] and -
 *          in the character class, as well as ^ and ! at the
 *          beginning.  The pattern matches a single character if it
 *          is one of the listed characters or falls into one of the
 *          listed ranges (inclusive, case sensitive).  Ranges with
 *          the first character larger than the second are legal but
 *          never match. Edge cases: [] never matches, and [^] and [!]
 *          always match without consuming a character.
 * 
 * Note that these patterns attempt to match the entire string, not
 * just find a substring matching the pattern.
 *
 * \param pool Apache pool
 * \param pStr The string we are trying to match
 * \param pPattern The pattern to match to
 * \return non-zero if the string matches to the pattern specified
 */
int   MatchToCompositePattern(apr_pool_t* pool, const char* pStr, const char* pPattern);
char* BruteForce(unsigned int        passmin,
                 unsigned int        passmax,
                 apr_pool_t*         pool,
                 const char*         pDict,
                 apr_byte_t*         desired,
                 unsigned long long* attempts);
int MakeAttempt(unsigned int pos, unsigned int length, const char* pDict, int* indexes, char* pass,
                apr_byte_t* desired, unsigned long long* attempts, int maxIndex);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char*          FromUtf8ToAnsi(const char* from, apr_pool_t* pool);
struct Version ReadVersion(apr_pool_t* pool, const char* pFile);

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, apr_pool_t* pool, UINT fromCodePage, UINT toCodePage);
#endif

int main(int argc, const char* const argv[])
{
    apr_pool_t* pool = NULL;
    apr_getopt_t* opt = NULL;
    int c = 0;
    const char* optarg = NULL;
    const char* pFile = NULL;
    const char* pDir = NULL;
    const char* pCheckSum = NULL;
    const char* pHashToSearch = NULL;
    const char* pString = NULL;
    const char* pExcludePattern = NULL;
    const char* pIncludePattern = NULL;
    const char* pDict = NULL;
    int isPrintLowCase = FALSE;
    int isScanDirRecursively = FALSE;
    int isPrintCalcTime = FALSE;
    int isCrack = FALSE;
    apr_byte_t digest[DIGESTSIZE];
    apr_status_t status = APR_SUCCESS;
    unsigned int passmin = 1;   // important!
    unsigned int passmax = 0;

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
            case OPT_HASH:
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
        PrintUsage();
        goto cleanup;
    }
    if (pDict == NULL) {
        pDict = alphabet;
    }

    if ((pFile != NULL) && (pCheckSum == NULL) && !isCrack &&
        CalculateFileHash(pool, pFile, digest, isPrintCalcTime, NULL)) {
        PrintHash(digest, isPrintLowCase);
    }
    if ((pString != NULL) && CalculateStringHash(pString, digest)) {
        PrintHash(digest, isPrintLowCase);
    }
    if ((pCheckSum != NULL) && (pFile != NULL) &&
        CalculateFileHash(pool, pFile, digest, isPrintCalcTime, NULL)) {
        CheckHash(digest, pCheckSum);
    }
    if (pDir != NULL) {
        CalculateDirContentHash(pool,
                               pDir,
                               isPrintLowCase,
                               isScanDirRecursively,
                               isPrintCalcTime,
                               pExcludePattern,
                               pIncludePattern,
                               pHashToSearch);
    }
    if ((pCheckSum != NULL) && isCrack) {
        CrackHash(pool, pDict, pCheckSum, passmin, passmax);
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

void PrintHash(apr_byte_t* digest, int isPrintLowCase)
{
    int i = 0;
    for (; i < DIGESTSIZE; ++i) {
        CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
    }
    NewLine();
}

void CheckHash(apr_byte_t* digest, const char* pCheckSum)
{
    CrtPrintf("File is %s!\n", CompareHash(digest, pCheckSum) ? "valid" : "invalid");
}

void ToDigest(const char* pCheckSum, apr_byte_t* digest)
{
    int i = 0;
    int to = MIN(DIGESTSIZE, strlen(pCheckSum) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(pCheckSum + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

int CompareHash(apr_byte_t* digest, const char* pCheckSum)
{
    apr_byte_t bytes[DIGESTSIZE];

    ToDigest(pCheckSum, bytes);
    return CompareDigests(bytes, digest);
}

void CrackHash(apr_pool_t*  pool,
              const char*  pDict,
              const char*  pCheckSum,
              unsigned int passmin,
              unsigned int passmax)
{
    char* pStr = NULL;
    apr_byte_t digest[DIGESTSIZE];
    unsigned long long attempts = 0;
    Time time = { 0 };

    StartTimer();

    // Empty string validation
    CalculateDigest(digest, NULL, 0);
    if (CompareHash(digest, pCheckSum)) {
        pStr = "Empty string";
        goto exit;
    }

    ToDigest(pCheckSum, digest);

    pStr = BruteForce(passmin, passmax ? passmax : strlen(pDict), pool, pDict, digest, &attempts);

exit:
    StopTimer();
    time = ReadElapsedTime();
    CrtPrintf("\nAttempts: %llu Time " FULL_TIME_FMT,
              attempts,
              time.hours,
              time.minutes,
              time.seconds);
    NewLine();
    if (pStr != NULL) {
        CrtPrintf("Initial string is: %s\n", pStr);
    } else {
        CrtPrintf("Nothing found\n");
    }
}

int MakeAttempt(unsigned int pos, unsigned int length, const char* pDict, int* indexes, char* pass,
                apr_byte_t* desired, unsigned long long* attempts, int maxIndex)
{
    int i = 0;
    unsigned int j = 0;
    apr_byte_t attempt[DIGESTSIZE];

    for (; i <= maxIndex; ++i) {
        indexes[pos] = i;

        if (pos == length - 1) {
            for (j = 0; j < length; ++j) {
                pass[j] = pDict[indexes[j]];
            }
            ++*attempts;
            CalculateDigest(attempt, pass, length);

            if (CompareDigests(attempt, desired)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, length, pDict, indexes, pass, desired, attempts, maxIndex)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

char* BruteForce(unsigned int        passmin,
                 unsigned int        passmax,
                 apr_pool_t*         pool,
                 const char*         pDict,
                 apr_byte_t*         desired,
                 unsigned long long* attempts)
{
    char* pass = NULL;
    int* indexes = NULL;
    unsigned int passLength = passmin;
    int maxIndex = strlen(pDict) - 1;

    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max password length is too big: %lu", passmax);
        return NULL;
    }
    pass = (char*)apr_pcalloc(pool, passmax + 1);
    if (pass == NULL) {
        CrtPrintf(ALLOCATION_FAIL_FMT, passmax + 1);
        return NULL;
    }
    indexes = (int*)apr_pcalloc(pool, passmax * sizeof(int));
    if (indexes == NULL) {
        CrtPrintf(ALLOCATION_FAIL_FMT, passmax * sizeof(int));
        return NULL;
    }

    for (; passLength <= passmax; ++passLength) {
        if (MakeAttempt(0, passLength, pDict, indexes, pass, desired, attempts, maxIndex)) {
            return pass;
        }
    }
    return NULL;
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

void CalculateDirContentHash(apr_pool_t* pool,
                            const char* dir,
                            int         isPrintLowCase,
                            int         isScanDirRecursively,
                            int         isPrintCalcTime,
                            const char* pExcludePattern,
                            const char* pIncludePattern,
                            const char* pHashToSearch)
{
    apr_finfo_t info = { 0 };
    apr_dir_t* d = NULL;
    apr_status_t status = APR_SUCCESS;
    apr_byte_t digest[DIGESTSIZE];
    char* fullPathToFile = NULL;
    apr_pool_t* filePool = NULL;
    apr_pool_t* dirPool = NULL;

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
        if ((info.filetype == APR_DIR) && isScanDirRecursively) {
            if (((info.name[0] == '.') && (info.name[1] == '\0'))
                || ((info.name[0] == '.') && (info.name[1] == '.') && (info.name[2] == '\0'))) {
                continue;
            }

            status = apr_filepath_merge(&fullPathToFile,
                                        dir,
                                        info.name,
                                        APR_FILEPATH_NATIVE,
                                        filePool);
            if (status != APR_SUCCESS) {
                PrintError(status);
                goto cleanup;
            }
            CalculateDirContentHash(pool,
                                   fullPathToFile,
                                   isPrintLowCase,
                                   isScanDirRecursively,
                                   isPrintCalcTime,
                                   pExcludePattern,
                                   pIncludePattern,
                                   pHashToSearch);
        }
        if ((status != APR_SUCCESS) || (info.filetype != APR_REG)) {
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

        if (CalculateFileHash(filePool, fullPathToFile, digest, isPrintCalcTime, pHashToSearch)) {
            PrintHash(digest, isPrintLowCase);
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

int MatchToCompositePattern(apr_pool_t* pool, const char* pStr, const char* pPattern)
{
    char* parts = NULL;
    char* last = NULL;
    char* p = NULL;

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

int CalculateFileHash(apr_pool_t* pool, const char* pFile, apr_byte_t* digest, int isPrintCalcTime,
                     const char* pHashToSearch)
{
    apr_file_t* file = NULL;
    apr_finfo_t info = { 0 };
    hash_context_t context = { 0 };
    apr_status_t status = APR_SUCCESS;
    apr_status_t hashCalcStatus = APR_SUCCESS;
    int result = TRUE;
    apr_off_t strSize = 0;
    apr_mmap_t* mmap = NULL;
    apr_off_t offset = 0;
    char* pFileAnsi = NULL;
    apr_byte_t digestToCompare[DIGESTSIZE];

    pFileAnsi = FromUtf8ToAnsi(pFile, pool);
    if (!pHashToSearch) {
        CrtPrintf("%s", pFileAnsi == NULL ? pFile : pFileAnsi);
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    StartTimer();

    status = apr_file_open(&file, pFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
    if (status != APR_SUCCESS) {
        PrintError(status);
        return FALSE;
    }
    status = InitContext(&context);
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
        status = CalculateDigest(digest, NULL, 0);
        goto endtiming;
    } else {
        strSize = info.size;
    }

    do {
        status =
            apr_mmap_create(&mmap, file, offset, (apr_size_t)MIN(strSize,
                                                                 info.size - offset), APR_MMAP_READ,
                            pool);
        if (status != APR_SUCCESS) {
            PrintError(status);
            result = FALSE;
            mmap = NULL;
            goto cleanup;
        }
        hashCalcStatus = UpdateHash(&context, mmap->mm, mmap->size);
        if (hashCalcStatus != APR_SUCCESS) {
            PrintError(hashCalcStatus);
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
    status = FinalHash(digest, &context);
endtiming:
    StopTimer();

    if (pHashToSearch) {
        ToDigest(pHashToSearch, digestToCompare);
        if (CompareDigests(digest, digestToCompare)) {
            CrtPrintf("%s", pFileAnsi == NULL ? pFile : pFileAnsi);
            CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
            PrintSize(info.size);
            if (isPrintCalcTime) {
                CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
                PrintTime(ReadElapsedTime());
            }
            NewLine();
        }
        result = FALSE;
    }

    if (isPrintCalcTime & !pHashToSearch) {
        PrintTime(ReadElapsedTime());
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

int CalculateStringHash(const char* pString, apr_byte_t* digest)
{
    apr_status_t status = APR_SUCCESS;

    if (pString == NULL) {
        CrtPrintf("NULL string passed\n");
        return FALSE;
    }
    status = CalculateDigest(digest, pString, strlen(pString));
    if (status != APR_SUCCESS) {
        CrtPrintf("Failed to calculate " HASH_NAME " of string: %s\n", pString);
        PrintError(status);
        return FALSE;
    }
    return TRUE;
}

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool)
{
#ifdef WIN32
    return DecodeUtf8Ansi(from, pool, CP_UTF8, CP_ACP);
#else
    return NULL;
#endif
}

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, apr_pool_t* pool, UINT fromCodePage, UINT toCodePage)
{
    int lengthWide = 0;
    int lengthAnsi = 0;
    size_t cbFrom = 0;
    wchar_t* wideStr = NULL;
    char* ansiStr = NULL;
    apr_size_t wideBufferSize = 0;

    cbFrom = strlen(from) + 1;  // IMPORTANT!!! including null terminator

    lengthWide = MultiByteToWideChar(fromCodePage, 0, from, cbFrom, NULL, 0);   // including null terminator
    wideBufferSize = sizeof(wchar_t) * lengthWide;
    wideStr = (wchar_t*)apr_pcalloc(pool, wideBufferSize);
    if (wideStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, wideBufferSize, __FILE__, __LINE__);
        return NULL;
    }
    MultiByteToWideChar(fromCodePage, 0, from, cbFrom, wideStr, lengthWide);

    lengthAnsi = WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, 0, NULL, NULL);   // null terminator included
    ansiStr = (char*)apr_pcalloc(pool, lengthAnsi);

    if (ansiStr == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, lengthAnsi, __FILE__, __LINE__);
        return NULL;
    }
    WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, lengthAnsi, NULL, NULL);

    return ansiStr;
}
#endif
