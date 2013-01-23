/*!
 * \brief   The file contains CRC32 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef CRC32_IMPLEMENTATION_H_
#define CRC32_IMPLEMENTATION_H_

#include "apr.h"
#include "apr_errno.h"
#include "crc32.h"

typedef Crc32Context hash_context_t;

#define CALC_DIGEST_NOT_IMPLEMETED
#define DIGESTSIZE CRC32_HASH_SIZE
#define APP_NAME "CRC32 Calculator " PRODUCT_VERSION
#define HASH_NAME "CRC32"
#define OPT_HASH_LONG "crc32"

#endif // CRC32_IMPLEMENTATION_H_
