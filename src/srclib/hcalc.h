/*!
 * \brief   The file contains common hash calculator definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-08-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef HC_HCALC_H_
#define HC_HCALC_H_

#include <stdio.h>
#include <locale.h>

#include "apr_getopt.h"
#include "lib.h"
#include "output.h"
#include "traverse.h"

void PrintUsage(void);
void PrintCopyright(void);

int CalculateStringHash(const char* string, apr_byte_t* digest);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool);

void OutputToConsole(OutputContext* ctx);

void* CreateDigest(const char* hash, apr_pool_t* pool);

#endif // HC_HCALC_H_
