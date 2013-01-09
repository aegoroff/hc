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

#include "lib.h"
#include <locale.h>

#include "apr_getopt.h"
#include "output.h"
#include "traverse.h"

void PrintUsage(void);
void PrintCopyright(void);
int CalculateStringHash(const char* string, apr_byte_t* digest, const apr_size_t inputLen);
void OutputToConsole(OutputContext* ctx);

#endif // HC_HCALC_H_
