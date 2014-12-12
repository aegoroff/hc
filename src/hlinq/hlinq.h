/*!
 * \brief   The file contains common HLINQ definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef HLINQ_HCALC_H_
#define HLINQ_HCALC_H_

#include <stdio.h>
#include <locale.h>
#include <assert.h>


#include "HLINQLexer.h"
#include "HLINQParser.h"

#define APP_NAME "Hash Calculator " PRODUCT_VERSION


#ifdef __cplusplus
extern "C" {
#endif

void PrintCopyright(void);
void PrintSyntax(void* argtable, void* argtableA, void* argtableQ, void* argtableQC, void* argtableQF);
void RunQuery(pANTLR3_INPUT_STREAM input,
              ProgramOptions*      options,
              const char*          param,
              apr_pool_t*          pool);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_HCALC_H_
