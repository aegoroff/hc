/*!
 * \brief   The file contains SHA256 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef SHA256_IMPLEMENTATION_H_
#define SHA256_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "sha256.h"

typedef SHA256Context hash_context_t;

#define DIGESTSIZE SHA256_HASH_SIZE
#define APP_NAME "SHA256 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA256"
#define OPT_HASH_LONG "sha256"

#endif // SHA256_IMPLEMENTATION_H_
