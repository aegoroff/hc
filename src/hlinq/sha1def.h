/*!
 * \brief   The file contains SHA1 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA1_HCALC_H_
#define SHA1_HCALC_H_

#include "apr.h"
#include "apr_errno.h"
#include "..\sha1\sph_sha1.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t SHA1InitContext(void* context);
apr_status_t SHA1FinalHash(apr_byte_t* digest, void* context);
apr_status_t SHA1UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // SHA1_HCALC_H_
