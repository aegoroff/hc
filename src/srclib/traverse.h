/*!
 * \brief   The file contains directory traverse interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef TRAVERSE_HCALC_H_
#define TRAVERSE_HCALC_H_

#include <stdio.h>
#include "apr_pools.h"
#include "apr_strings.h"
#include "apr_fnmatch.h"
#include "apr_tables.h"
#include "filehash.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TraverseContext {
    int                 IsScanDirRecursively;
    apr_array_header_t* ExcludePattern;
    apr_array_header_t* IncludePattern;
    apr_status_t        (* PfnFileHandler)(const char* pathToFile, void* ctx, apr_pool_t* pool);
    void* DataCtx;
} TraverseContext;

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

BOOL FilterByName(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool);

void TraverseDirectory(
    const char* dir, 
    TraverseContext* ctx, 
    BOOL (*filter)(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool),
    apr_pool_t* pool);



#ifdef __cplusplus
}
#endif

#endif // TRAVERSE_HCALC_H_
