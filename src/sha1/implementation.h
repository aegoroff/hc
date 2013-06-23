/*!
 * \brief   The file contains SHA1 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef SHA1_IMPLEMENTATION_H_
#define SHA1_IMPLEMENTATION_H_

#include "sph_sha1.h"

typedef sph_sha1_context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define DIGESTSIZE (SPH_SIZE_sha1 / 8)
#define APP_NAME "SHA1 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA1"
#define OPT_HASH_LONG "sha1"

#endif // SHA1_IMPLEMENTATION_H_
