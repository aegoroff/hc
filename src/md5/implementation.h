/*!
 * \brief   The file contains MD5 calculator implementation defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-07-21
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef MD5_IMPLEMENTATION_H_
#define MD5_IMPLEMENTATION_H_

#include "sph_md5.h"

typedef sph_md5_context hash_context_t;

#define DIGESTSIZE (SPH_SIZE_md5 / 8)
#define APP_NAME "MD5 Calculator " PRODUCT_VERSION
#define HASH_NAME "MD5"
#define OPT_HASH_LONG "md5"

#endif // MD5_IMPLEMENTATION_H_
