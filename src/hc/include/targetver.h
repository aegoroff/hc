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

#ifndef HC_TARGETVER_H_
#define HC_TARGETVER_H_

// Including SDKDDKVer.h defines the highest available Windows platform.

// If you wish to build your application for a previous Windows platform, include WinSDKVer.h and
// set the _WIN32_WINNT macro to the platform you wish to support before including SDKDDKVer.h.

#ifdef _MSC_VER
#include <SDKDDKVer.h>
#endif

#ifndef PRODUCT_VERSION
#define PRODUCT_VERSION "1.0.0.1"
#endif

#define PROGRAM_NAME_BASE "hc"

#ifdef _MSC_VER
#define PROGRAM_NAME PROGRAM_NAME_BASE ".exe"
#else
#define PROGRAM_NAME PROGRAM_NAME_BASE
#endif

#endif // HC_TARGETVER_H_
