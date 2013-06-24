/*!
 * \brief   The file contains hashes from libtom lib API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef LIBTOM_HCALC_H_
#define LIBTOM_HCALC_H_

#include "apr.h"
#include "apr_errno.h"
#include "sph_md2.h"
#include "sph_ripemd.h"
#include "sph_sha2.h"
#include "sph_tiger.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t MD2InitContext(void* context);
apr_status_t MD2FinalHash(apr_byte_t* digest, void* context);
apr_status_t MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t TIGERInitContext(void* context);
apr_status_t TIGERFinalHash(apr_byte_t* digest, void* context);
apr_status_t TIGERUpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t TIGER2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t TIGER2InitContext(void* context);
apr_status_t TIGER2FinalHash(apr_byte_t* digest, void* context);
apr_status_t TIGER2UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t SHA224InitContext(void* context);
apr_status_t SHA224FinalHash(apr_byte_t* digest, void* context);
apr_status_t SHA224UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t RMD128InitContext(void* context);
apr_status_t RMD128FinalHash(apr_byte_t* digest, void* context);
apr_status_t RMD128UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t RMD160InitContext(void* context);
apr_status_t RMD160FinalHash(apr_byte_t* digest, void* context);
apr_status_t RMD160UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t RMD256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t RMD256InitContext(void* context);
apr_status_t RMD256FinalHash(apr_byte_t* digest, void* context);
apr_status_t RMD256UpdateHash(void* context, const void* input, const apr_size_t inputLen);

apr_status_t RMD320CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t RMD320InitContext(void* context);
apr_status_t RMD320FinalHash(apr_byte_t* digest, void* context);
apr_status_t RMD320UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // LIBTOM_HCALC_H_
