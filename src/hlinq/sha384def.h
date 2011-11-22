/*!
 * \brief   The file contains SHA384 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#ifndef SHA384DEF_HCALC_H_
#define SHA384DEF_HCALC_H_

#include "apr_errno.h"
#include "..\sha384\sha384.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t SHA384InitContext(void* context);
apr_status_t SHA384FinalHash(apr_byte_t* digest, void* context);
apr_status_t SHA384UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // SHA384DEF_HCALC_H_