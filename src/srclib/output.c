/*!
 * \brief   The file contains output implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include <assert.h>
#include "apr_strings.h"
#include "output.h"
#include "lib.h"

const char* CreateErrorMessage(apr_status_t status, apr_pool_t* pool)
{
    char* message = (char*)apr_pcalloc(pool, ERROR_BUFFER_SIZE);
    apr_strerror(status, message, ERROR_BUFFER_SIZE);
    return message;
}

void OutputErrorMessage(apr_status_t status, void (* PfnOutput)(
                            OutputContext* ctx), apr_pool_t* pool)
{
    OutputContext ctx = { 0 };
    ctx.StringToPrint = CreateErrorMessage(status, pool);
    ctx.IsPrintSeparator = FALSE;
    ctx.IsFinishLine = TRUE;
    PfnOutput(&ctx);
}

void PrintError(apr_status_t status)
{
    char errbuf[ERROR_BUFFER_SIZE];
    apr_strerror(status, errbuf, ERROR_BUFFER_SIZE);
    CrtPrintf("%s", errbuf); //-V111
    NewLine();
}

const char* CopySizeToString(uint64_t size, apr_pool_t* pool)
{
    size_t sz = 64;
    char* str = apr_pcalloc(pool, sz);
    SizeToString(size, sz, str);
    return str;
}

const char* CopyTimeToString(Time time, apr_pool_t* pool)
{
    size_t sz = 48;
    char* str = apr_pcalloc(pool, sz);
    TimeToString(time, sz, str);
    return str;
}

const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_size_t sz, apr_pool_t* pool)
{
    apr_size_t i = 0;
    char* str = apr_pcalloc(pool, sz * BYTE_CHARS_SIZE + 1); // iteration ponter
    char* result = str; // result pointer

    for (; i < sz; ++i) {
        apr_snprintf(str, BYTE_CHARS_SIZE + 1, isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
        str += BYTE_CHARS_SIZE;
    }
    return result;
}

void OutputToConsole(OutputContext* ctx)
{
    if (ctx == NULL) {
        assert(ctx != NULL);
        return;
    }
    CrtPrintf("%s", ctx->StringToPrint); //-V111
    if (ctx->IsPrintSeparator) {
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->IsFinishLine) {
        NewLine();
    }
}
