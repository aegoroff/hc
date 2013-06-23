/*!
 * \brief   The file contains SHA512 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA512_IMPLEMENTATION_H_
#define SHA512_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "sph_sha2.h"

typedef sph_sha512_context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define SHA512_HASH_SIZE (SPH_SIZE_sha512/8)
#define DIGESTSIZE SHA512_HASH_SIZE
#define APP_NAME "SHA512 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA512"
#define OPT_HASH_LONG "sha512"

#endif // SHA512_IMPLEMENTATION_H_
