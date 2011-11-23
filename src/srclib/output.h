/*!
 * \brief   The file contains output interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef OUTPUT_HCALC_H_
#define OUTPUT_HCALC_H_

#include <stdio.h>
#include "apr_pools.h"

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OutputContext {
    int         IsPrintSeparator;
    int         IsFinishLine;
    const char* StringToPrint;
} OutputContext;

void OutputErrorMessage(apr_status_t status, void (* PfnOutput)(
        OutputContext* ctx), apr_pool_t * pool);

void        PrintError(apr_status_t status);

#ifdef __cplusplus
}
#endif

#endif // OUTPUT_HCALC_H_
