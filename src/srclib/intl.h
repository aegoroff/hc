/*!
 * \brief   The file contains necessary internationalization included and defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-29
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#ifndef LINQ2HASH_INTL_H_
#define LINQ2HASH_INTL_H_

#include <stdio.h>
#include "types.h"

/* internationalization support via gettext/libintl */
#ifdef USE_GETTEXT
# include <libgnuintl.h>
# define _(str) gettext(str)

# ifdef _WIN32
#  define LOCALEDIR "./"
# else /* _WIN32 */
#  define LOCALEDIR "/usr/share/locale"
# endif /* _WIN32 */

#else
# define _(str) (str)
#endif /* USE_GETTEXT */

#endif // LINQ2HASH_INTL_H_
