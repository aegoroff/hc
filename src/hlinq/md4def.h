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

#include "apr.h"
#include "apr_errno.h"
#include "..\md4\sph_md4.h"

#ifdef __cplusplus
extern "C" {
#endif

void MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void MD4InitContext(void* context);
void MD4FinalHash(apr_byte_t* digest, void* context);
void MD4UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // MD4_HCALC_H_