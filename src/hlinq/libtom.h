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

void MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void MD2InitContext(void* context);
void MD2FinalHash(apr_byte_t* digest, void* context);
void MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void TIGERInitContext(void* context);
void TIGERFinalHash(apr_byte_t* digest, void* context);
void TIGERUpdateHash(void* context, const void* input, const apr_size_t inputLen);

void TIGER2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void TIGER2InitContext(void* context);
void TIGER2FinalHash(apr_byte_t* digest, void* context);
void TIGER2UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void SHA224InitContext(void* context);
void SHA224FinalHash(apr_byte_t* digest, void* context);
void SHA224UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void RMD128InitContext(void* context);
void RMD128FinalHash(apr_byte_t* digest, void* context);
void RMD128UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void RMD160InitContext(void* context);
void RMD160FinalHash(apr_byte_t* digest, void* context);
void RMD160UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void RMD256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void RMD256InitContext(void* context);
void RMD256FinalHash(apr_byte_t* digest, void* context);
void RMD256UpdateHash(void* context, const void* input, const apr_size_t inputLen);

void RMD320CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void RMD320InitContext(void* context);
void RMD320FinalHash(apr_byte_t* digest, void* context);
void RMD320UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // LIBTOM_HCALC_H_
