/*!
 * \brief   The file contains SHA256 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA256DEF_HCALC_H_
#define SHA256DEF_HCALC_H_

#include "apr_errno.h"
#include "..\sha256\sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t SHA256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t SHA256InitContext(void* context);
apr_status_t SHA256FinalHash(apr_byte_t* digest, void* context);
apr_status_t SHA256UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // SHA256DEF_HCALC_H_