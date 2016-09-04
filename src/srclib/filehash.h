/*!
 * \brief   The file contains file hash interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#ifndef FILEHASH_HCALC_H_
#define FILEHASH_HCALC_H_

#include <stdio.h>
#include <apr.h>
#include <apr_pools.h>
#include <apr_file_io.h>
#include "output.h"

#define HASH_FILE_COLUMN_SEPARATOR "   "
#define PATH_ELT_SEPARATOR '\\'

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FileHashResult {
    const char* File;
    const char* Size;
    const char* Hash;
    const char* CalculationTime;
    const char* ErrorMessage;
} FileHashResult;

typedef struct DataContext {
    int         IsPrintLowCase;
    int         IsPrintCalcTime;
    int         IsPrintSfv;
    int         IsPrintVerify;
    int         IsValidateFileByHash;
    int         IsPrintErrorOnFind;
    const char* HashToSearch;
    apr_off_t   Limit;
    apr_off_t   Offset;
    void        (* PfnOutput)(out_context_t* ctx);
} DataContext;


void CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);

int  CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);

void ToDigest(const char* hash, apr_byte_t* digest);

// These functions must be defined in concrete calculator implementation
void CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void InitContext(void* context);
void FinalHash(void* context, apr_byte_t* digest);
void UpdateHash(void* context, const void* input, const apr_size_t inputLen);
int         bf_compare_hash(apr_byte_t* digest, const char* checkSum);

const char* CalculateHash(apr_file_t* fileHandle,
                   apr_off_t fileSize,
                   apr_byte_t* digest,
                   apr_off_t   limit,
                   apr_off_t   offset,
                   apr_pool_t* pool);

void* AllocateContext(apr_pool_t* pool);
apr_size_t GetDigestSize();
int ComparisonFailure(int result);

#ifdef __cplusplus
}
#endif

#endif // FILEHASH_HCALC_H_
