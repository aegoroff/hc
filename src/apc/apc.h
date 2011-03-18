/*!
 * \brief   The file contains common hash calculator definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef APC_HCALC_H_
#define APC_HCALC_H_

#include <stdio.h>
#include <locale.h>

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_file_io.h"
#include "apr_fnmatch.h"
#include "apr_tables.h"
#include "..\srclib\lib.h"

typedef struct OutputContext {
    int         IsPrintSeparator;
    int         IsFinishLine;
    const char* StringToPrint;
} OutputContext;

void PrintUsage(void);
void PrintCopyright(void);
void        PrintError(apr_status_t status);
const char* CreateErrorMessage(apr_status_t status, apr_pool_t* pool);
void OutputErrorMessage(apr_status_t status, void (* PfnOutput)(
        OutputContext* ctx), apr_pool_t * pool);

void OutputToConsole(OutputContext* ctx);

#endif // APC_HCALC_H_
