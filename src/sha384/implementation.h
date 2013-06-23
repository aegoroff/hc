/*!
 * \brief   The file contains SHA384 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA384_IMPLEMENTATION_H_
#define SHA384_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "sph_sha2.h"

typedef sph_sha384_context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define SHA384_HASH_SIZE (SPH_SIZE_sha384/8)
#define DIGESTSIZE SHA384_HASH_SIZE
#define APP_NAME "SHA384 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA384"
#define OPT_HASH_LONG "sha384"

#endif // SHA384_IMPLEMENTATION_H_
