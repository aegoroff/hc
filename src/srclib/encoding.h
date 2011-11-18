/*!
 * \brief   The file contains encoding functions interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef ENCODING_HCALC_H_
#define ENCODING_HCALC_H_

#include "apr_pools.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool);

/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* FromAnsiToUtf8(const char* from, apr_pool_t* pool);

#ifdef WIN32
/*!
 * IMPORTANT: Memory allocated for result must be freed up by caller
 */
char* DecodeUtf8Ansi(const char* from, UINT fromCodePage, UINT toCodePage, apr_pool_t* pool);
#endif

#ifdef __cplusplus
}
#endif

#endif // ENCODING_HCALC_H_
