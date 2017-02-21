// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains encoding functions interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_ENCODING_H_
#define LINQ2HASH_ENCODING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "apr_pools.h"

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_utf8_to_ansi(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_ansi_to_utf8(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
wchar_t* enc_from_ansi_to_unicode(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_from_unicode_to_ansi(const wchar_t* from, apr_pool_t* pool);

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* enc_decode_utf8_ansi(const char* from, UINT from_code_page, UINT to_code_page, apr_pool_t* pool);
#endif

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_ENCODING_H_
