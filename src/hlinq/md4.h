/*!
 * \brief   The file contains MD4 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef MD4_HCALC_H_
#define MD4_HCALC_H_

#include "apr_md4.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t MD4InitContext(void* context);
apr_status_t MD4FinalHash(apr_byte_t* digest, void* context);
apr_status_t MD4UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // MD4_HCALC_H_