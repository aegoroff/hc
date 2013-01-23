/*!
 * \brief   The file contains MD4 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef MD4_IMPLEMENTATION_H_
#define MD4_IMPLEMENTATION_H_

#include "apr_md4.h"

typedef apr_md4_ctx_t hash_context_t;

#define DIGESTSIZE APR_MD4_DIGESTSIZE
#define APP_NAME "MD4 Calculator " PRODUCT_VERSION
#define HASH_NAME "MD4"
#define OPT_HASH_LONG "md4"

#endif // MD4_IMPLEMENTATION_H_
