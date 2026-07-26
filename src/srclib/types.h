/*!
 * \brief   The file contains fundamental types definitions
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#ifndef LINQ2HASH_TYPES_H_
#define LINQ2HASH_TYPES_H_

#include <stdint.h>

#ifdef _MSC_VER
/* Win32 `BOOL` is typedef'd in windef.h (via windows.h). Translation units that
 * never include windows.h (notably the l2h parser chain: frontend.h -> types.h)
 * still need the type, so fall back to a compatible `int` definition when
 * windef.h hasn't run. _WINDEF_ is windef.h's include guard. */
#ifndef _WINDEF_
typedef int BOOL;
#endif
#else
#include <stdbool.h>
#include <stddef.h>
#ifndef BOOL
#define BOOL bool
#endif
#endif

#endif // LINQ2HASH_TYPES_H_
