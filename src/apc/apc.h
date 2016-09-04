/*!
 * \brief   The file contains common Apache passwords cracker definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-06
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef APC_HCALC_H_
#define APC_HCALC_H_

#include "apr_pools.h"
#include "apr_file_io.h"
#include "..\srclib\lib.h"
#include "..\srclib\output.h"

typedef struct CrackContext {
    const char* Dict;
    uint32_t    Passmin;
    uint32_t    Passmax;
    uint32_t    NumOfThreads;
    const char* Login;
} CrackContext;

void        PrintUsage(void);
void        PrintCopyright(void);

void CrackHtpasswdHash(const char* dict,
               const char* hash,
               const uint32_t    passmin,
               const uint32_t    passmax,
               const uint32_t    numOfThreads,
               apr_pool_t* pool);

void* PassThrough(const char* hash, apr_pool_t* pool);
void CrackFile(const char* file,
    void        (* PfnOutput)(out_context_t* ctx),
    const char* dict,
    const uint32_t    passmin,
    const uint32_t    passmax,
    const uint32_t    numOfThreads,
    const char* login,
    apr_pool_t * pool);

void ListAccounts(const char* file, void (* PfnOutput)(out_context_t* ctx), apr_pool_t* pool);

void ReadPasswdFile(
    const char* file,
    void (* PfnOutput)(out_context_t* ctx), 
    void (* PfnCallback)(out_context_t* ctx, void (* PfnOutput)(out_context_t* ctx), apr_file_t* fileHandle, void* context, apr_pool_t* pool),
    void* context,
    apr_pool_t * pool);

void ListAccountsCallback(
    out_context_t* ctx,
    void (* PfnOutput)(out_context_t* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool);

void CrackFileCallback(
    out_context_t* ctx,
    void (* PfnOutput)(out_context_t* ctx),
    apr_file_t* fileHandle,
    void* context,
    apr_pool_t* pool);

int IsValidAsciiString(const char* string, size_t size);

#endif // APC_HCALC_H_
