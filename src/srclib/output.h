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

#include "lib.h"
#include "apr_pools.h"

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define BYTE_CHARS_SIZE 2   // byte representation string length

#define FILE_INFO_COLUMN_SEPARATOR " | "

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

const char* CopySizeToString(uint64_t size, apr_pool_t* pool);
const char* CopyTimeToString(Time time, apr_pool_t* pool);
const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_size_t sz, apr_pool_t* pool);
void OutputToConsole(OutputContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // OUTPUT_HCALC_H_
