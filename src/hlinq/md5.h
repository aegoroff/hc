/*!
 * \brief   The file contains MD5 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef MD5_HCALC_H_
#define MD5_HCALC_H_

#include "apr_md5.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t MD5InitContext(void* context);
apr_status_t MD5FinalHash(apr_byte_t* digest, void* context);
apr_status_t MD5UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // MD5_HCALC_H_
