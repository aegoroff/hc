/*!
 * \brief   The file contains SHA384 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#ifndef SHA384_IMPLEMENTATION_H_
#define SHA384_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "sha384.h"

typedef SHA384Context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define DIGESTSIZE SHA384_HASH_SIZE
#define APP_NAME "SHA384 Calculator " PRODUCT_VERSION
#define HASH_NAME "SHA384"
#define OPT_HASH_LONG "sha384"

#endif // SHA384_IMPLEMENTATION_H_
