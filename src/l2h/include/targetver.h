/*!
 * \brief   The file contains common project defines
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-02-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#ifndef L2H_TARGETVER_H_
#define L2H_TARGETVER_H_

// Including SDKDDKVer.h defines the highest available Windows platform.

// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.

#ifdef _MSC_VER
#include <SDKDDKVer.h>
#endif


#define PRODUCT_VERSION "1.0.0.1"
#define PROGRAM_NAME "l2h"
#define APP_NAME "LINQ to Hash tool " PRODUCT_VERSION
#define PROG_EXE PROGRAM_NAME ".exe"


#endif // L2H_TARGETVER_H_
