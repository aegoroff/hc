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

#ifndef MD5_HCALC_H_
#define MD5_HCALC_H_

#include <stdio.h>
#include <locale.h>

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_file_io.h"
#include "apr_mmap.h"
#include "apr_fnmatch.h"

typedef struct DataContext {
    int         isPrintLowCase;
    int         isPrintCalcTime;
    const char* pHashToSearch;
    apr_file_t* fileToSave;
} DataContext;

typedef struct TraverseContext {
    int         isScanDirRecursively;
    const char* pExcludePattern;
    const char* pIncludePattern;
    void         (* pfnFileHandler)(apr_pool_t* pool, const char* pathToFile, DataContext* ctx);
    DataContext* dataContext;
} TraverseContext;

void PrintUsage(void);
void PrintCopyright(void);
int  CalculateFileHash(apr_pool_t* pool, const char* file, apr_byte_t* digest, int isPrintCalcTime,
                       const char* pHashToSearch);
void CalculateFile(apr_pool_t* pool, const char* pathToFile, DataContext* ctx);
void TraverseDirectory(apr_pool_t* pool, const char* dir, TraverseContext* ctx);

int  CalculateStringHash(const char* string, apr_byte_t* digest);
void PrintHash(apr_byte_t* digest, int isPrintLowCase);
void PrintFileName(const char* pFile, const char* pFileAnsi);
void CheckHash(apr_byte_t* digest, const char* pCheckSum);
int  CompareHash(apr_byte_t* digest, const char* pCheckSum);
void PrintError(apr_status_t status);
void CrackHash(apr_pool_t*  pool,
               const char*  pDict,
               const char*  pCheckSum,
               unsigned int passmin,
               unsigned int passmax);
int  CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);
void ToDigest(const char* pCheckSum, apr_byte_t* digest);

// These functions must be defined in concrete calculator implementation
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, apr_size_t inputLen);
apr_status_t InitContext(hash_context_t* context);
apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context);
apr_status_t UpdateHash(hash_context_t* context, const void* input, apr_size_t inputLen);

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
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool);

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, apr_pool_t* pool, UINT fromCodePage, UINT toCodePage);
#endif

#endif // MD5_HCALC_H_
