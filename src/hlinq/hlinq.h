/*!
 * \brief   The file contains common HLINQ definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#ifndef HLINQ_HCALC_H_
#define HLINQ_HCALC_H_

#include <stdio.h>
#include <locale.h>
#include <assert.h>
#include "apr_getopt.h"

#include "HLINQLexer.h"
#include "HLINQParser.h"

#define APP_NAME "Hash LINQ " PRODUCT_VERSION


#ifdef __cplusplus
extern "C" {
#endif

void PrintUsage(void);
void PrintCopyright(void);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_HCALC_H_
