/*!
 * \brief   The file contains CRC32 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#ifndef CRC32DEF_HCALC_H_
#define CRC32DEF_HCALC_H_

#include "apr_errno.h"
#include "..\crc32\crc32.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t CRC32CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t CRC32InitContext(void* context);
apr_status_t CRC32FinalHash(apr_byte_t* digest, void* context);
apr_status_t CRC32UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // CRC32DEF_HCALC_H_