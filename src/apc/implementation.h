/*!
 * \brief   The file contains CRC32 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef APC_IMPLEMENTATION_H_
#define APC_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"

#define CALC_DIGEST_NOT_IMPLEMETED
#define DIGESTSIZE CRC32_HASH_SIZE
#define APP_NAME "Apache password recovery " PRODUCT_VERSION

#endif // APC_IMPLEMENTATION_H_
