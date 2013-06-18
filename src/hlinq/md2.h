/*!
 * \brief   The file contains MD2 API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef MD2_HCALC_H_
#define MD2_HCALC_H_

#include "apr.h"
#include "apr_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

apr_status_t MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t MD2InitContext(void* context);
apr_status_t MD2FinalHash(apr_byte_t* digest, void* context);
apr_status_t MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen);

#ifdef __cplusplus
}
#endif

#endif // MD2_HCALC_H_