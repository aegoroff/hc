/*!
 * \brief   The file contains common Apache passwords cracker definitions and interface
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

typedef struct CrackContext {
    const char* Dict;
    uint32_t    Passmin;
    uint32_t    Passmax;
    const char* Login;
} CrackContext;

void        PrintUsage(void);
void        PrintCopyright(void);
void        PrintError(apr_status_t status);
const char* CreateErrorMessage(apr_status_t status, apr_pool_t* pool);
void OutputErrorMessage(apr_status_t status, void (* PfnOutput)(
        OutputContext* ctx), apr_pool_t * pool);

void OutputToConsole(OutputContext* ctx);

void CrackHash(const char* dict,
               const char* hash,
               const uint32_t    passmin,
               const uint32_t    passmax,
               apr_pool_t* pool);

void* PassThrough(const char* hash, apr_pool_t* pool);
void CrackFile(const char* file,
    void        (* PfnOutput)(OutputContext* ctx),
    const char* dict,
    const uint32_t    passmin,
    const uint32_t    passmax,
    const char* login,
    apr_pool_t * pool);

void ListAccounts(const char* file, void (* PfnOutput)(OutputContext* ctx), apr_pool_t* pool);

void ReadPasswdFile(
    const char* file,
    void (* PfnOutput)(OutputContext* ctx), 
    void (* PfnCallback)(OutputContext* ctx, void (* PfnOutput)(OutputContext* ctx), apr_file_t* fileHandle, void* context, apr_pool_t* pool),
    void* context,
    apr_pool_t * pool);

void ListAccountsCallback(
    OutputContext* ctx,
    void (* PfnOutput)(OutputContext* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool);

void CrackFileCallback(
    OutputContext* ctx,
    void (* PfnOutput)(OutputContext* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool);

int IsValidAsciiString(const char* string, int size);

#endif // APC_HCALC_H_
