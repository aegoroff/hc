/*!
 * \brief   The file contains WHIRLPOOL calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef WHIRLPOOL_IMPLEMENTATION_H_
#define WHIRLPOOL_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "sph_whirlpool.h"

#define WHIRLPOOL_DIGEST_LENGTH	(512/8)

typedef sph_whirlpool_context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define DIGESTSIZE WHIRLPOOL_DIGEST_LENGTH
#define APP_NAME "WHIRLPOOL Calculator " PRODUCT_VERSION
#define HASH_NAME "WHIRLPOOL"
#define OPT_HASH_LONG "whirlpool"

#endif // WHIRLPOOL_IMPLEMENTATION_H_
