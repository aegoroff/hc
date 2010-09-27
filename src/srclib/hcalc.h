/*!
 * \brief   The file contains common hash calculator definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-08-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef HC_HCALC_H_
#define HC_HCALC_H_

#include <stdio.h>
#include <locale.h>

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_file_io.h"
#include "apr_mmap.h"
#include "apr_fnmatch.h"
#include "apr_tables.h"
#include "lib.h"

typedef struct OutputContext {
    int         IsPrintSeparator;
    int         IsFinishLine;
    const char* StringToPrint;
} OutputContext;

typedef struct DataContext {
    int         IsPrintLowCase;
    int         IsPrintCalcTime;
    const char* HashToSearch;
    apr_off_t   Limit;
    apr_off_t   Offset;
    apr_file_t* FileToSave;
    void        (* PfnOutput)(OutputContext* ctx);
} DataContext;

typedef struct TraverseContext {
    int                 IsScanDirRecursively;
    apr_array_header_t* ExcludePattern;
    apr_array_header_t* IncludePattern;
    apr_status_t        (* PfnFileHandler)(const char* pathToFile, void* ctx, apr_pool_t* pool);
    void* DataCtx;
} TraverseContext;

void PrintUsage(void);
void PrintCopyright(void);
int  CalculateFileHash(const char* filePath,
                       apr_byte_t* digest,
                       int         isPrintCalcTime,
                       const char* hashToSearch,
                       apr_off_t   limit,
                       apr_off_t   offset,
                       apr_pool_t* pool);
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);
void         TraverseDirectory(const char* dir, TraverseContext* ctx, apr_pool_t* pool);

int  CalculateStringHash(const char* string, apr_byte_t* digest);
void PrintHash(apr_byte_t* digest, int isPrintLowCase);
void PrintFileName(const char* file, const char* fileAnsi);
void CheckHash(apr_byte_t* digest, const char* checkSum);
int  CompareHash(apr_byte_t* digest, const char* checkSum);
void PrintError(apr_status_t status);
void CrackHash(const char* dict,
               const char* checkSum,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool);
int  CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);
void ToDigest(const char* checkSum, apr_byte_t* digest);

// These functions must be defined in concrete calculator implementation
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen);
apr_status_t InitContext(hash_context_t* context);
apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context);
apr_status_t UpdateHash(hash_context_t* context, const void* input, apr_size_t inputLen);
const char*  HackRootPath(const char* path, apr_pool_t* pool);

/*!
 * \brief Try to match the string to the given pattern using apr_fnmatch function.
 *        Matching is case insensitive
 * \param str The string we are trying to match
 * \param pattern The pattern to match to
 * \return non-zero if the string matches to the pattern specified
 */
int MatchToCompositePattern(const char* str, apr_array_header_t* pattern);

/*!
 * \brief Compile composite pattern into patterns' table.
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
 * \param pattern The pattern to match to
 * \param newpattern The patterns' array
 * \param pool Apache pool
 */
void CompilePattern(const char* pattern, apr_array_header_t** newpattern, apr_pool_t* pool);

void OutputToConsole(OutputContext* ctx);

char* BruteForce(uint32_t    passmin,
                 uint32_t    passmax,
                 const char* dict,
                 apr_byte_t* desired,
                 uint64_t*   attempts,
                 apr_pool_t* pool);
int MakeAttempt(uint32_t pos, uint32_t length, const char* dict, int* indexes, char* pass,
                apr_byte_t* desired, uint64_t* attempts, int maxIndex);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool);

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, UINT fromCodePage, UINT toCodePage, apr_pool_t* pool);
#endif

#endif // HC_HCALC_H_
