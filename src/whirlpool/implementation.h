/*!
 * \brief   The file contains WHIRLPOOL calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef WHIRLPOOL_IMPLEMENTATION_H_
#define WHIRLPOOL_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "whrlpool.h"


typedef WHIRLPOOL_CTX hash_context_t;

#define DIGESTSIZE WHIRLPOOL_DIGEST_LENGTH
#define APP_NAME "WHIRLPOOL Calculator " PRODUCT_VERSION
#define HASH_NAME "WHIRLPOOL"
#define OPT_HASH_LONG "whirlpool"

#endif // WHIRLPOOL_IMPLEMENTATION_H_
