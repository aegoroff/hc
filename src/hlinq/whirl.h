/*!
 * \brief   The file contains WHIRLPOOL API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef WHIRL_HCALC_H_
#define WHIRL_HCALC_H_

#include "apr_errno.h"
#include "..\whirlpool\whrlpool.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t WHIRLPOOLCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t WHIRLPOOLInitContext(void* context);
apr_status_t WHIRLPOOLFinalHash(apr_byte_t* digest, void* context);
apr_status_t WHIRLPOOLUpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // WHIRL_HCALC_H_