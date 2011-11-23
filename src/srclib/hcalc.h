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
#include "apr_mmap.h"
#include "lib.h"
#include "traverse.h"

void PrintUsage(void);
void PrintCopyright(void);
int CalculateFileHash(const char* filePath,
    apr_byte_t * digest,
    int         isPrintCalcTime,
    const char* hashToSearch,
    apr_off_t   limit,
    apr_off_t   offset,
    void (* PfnOutput)(OutputContext* ctx),
    apr_pool_t * pool);
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);

int         CalculateStringHash(const char* string, apr_byte_t* digest);
void        CheckHash(apr_byte_t* digest, const char* checkSum, DataContext* ctx);
int         CompareHash(apr_byte_t* digest, const char* checkSum);
void        PrintError(apr_status_t status);
const char* CreateErrorMessage(apr_status_t status, apr_pool_t* pool);

const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_pool_t* pool);
void        OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_pool_t* pool);
const char* CopySizeToString(uint64_t size, apr_pool_t* pool);
const char* CopyTimeToString(Time time, apr_pool_t* pool);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_pool_t* pool);
int  CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);
void ToDigest(const char* hash, apr_byte_t* digest);

// These functions must be defined in concrete calculator implementation
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t InitContext(hash_context_t* context);
apr_status_t FinalHash(apr_byte_t* digest, hash_context_t* context);
apr_status_t UpdateHash(hash_context_t* context, const void* input, const apr_size_t inputLen);

void OutputToConsole(OutputContext* ctx);

void* CreateDigest(const char* hash, apr_pool_t* pool);

#endif // HC_HCALC_H_
