/*!
 * \brief   The file contains SHA512 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA512DEF_HCALC_H_
#define SHA512DEF_HCALC_H_

#include "apr_errno.h"
#include "sph_sha2.h"

#ifdef __cplusplus
extern "C" {
#endif

void SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void SHA512InitContext(void* context);
void SHA512FinalHash(apr_byte_t* digest, void* context);
void SHA512UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // SHA512DEF_HCALC_H_
