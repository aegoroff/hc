/*!
 * \brief   The file contains file hash interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef FILEHASH_HCALC_H_
#define FILEHASH_HCALC_H_

#include <stdio.h>
#include "apr_pools.h"
#include "apr_file_io.h"
#include "output.h"

#define HASH_FILE_COLUMN_SEPARATOR "   "
#define PATH_ELT_SEPARATOR '\\'

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DataContext {
    int         IsPrintLowCase;
    int         IsPrintCalcTime;
    const char* HashToSearch;
    apr_off_t   Limit;
    apr_off_t   Offset;
    apr_file_t* FileToSave;
    void        (* PfnOutput)(OutputContext* ctx);
} DataContext;


int CalculateFileHash(const char* filePath,
    apr_byte_t * digest,
    int         isPrintCalcTime,
    const char* hashToSearch,
    apr_off_t   limit,
    apr_off_t   offset,
    void (* PfnOutput)(OutputContext* ctx),
    apr_pool_t * pool);

apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);

void OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_size_t sz, apr_pool_t* pool);

int  CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);

void ToDigest(const char* hash, apr_byte_t* digest);

// These functions must be defined in concrete calculator implementation
apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
apr_status_t InitContext(void* context);
apr_status_t FinalHash(apr_byte_t* digest, void* context);
apr_status_t UpdateHash(void* context, const void* input, const apr_size_t inputLen);
void        CheckHash(apr_byte_t* digest, const char* checkSum, DataContext* ctx);
int         CompareHash(apr_byte_t* digest, const char* checkSum);

void CalculateHash(apr_file_t* fileHandle,
                   apr_off_t fileSize,
                   apr_byte_t* digest,
                   apr_off_t   limit,
                   apr_off_t   offset,
                   void        (* PfnOutput)(OutputContext* ctx),
                   apr_pool_t* pool);

void* AllocateContext(apr_pool_t* pool);
apr_size_t GetDigestSize();
int ComparisonFailure(int result);

#ifdef __cplusplus
}
#endif

#endif // FILEHASH_HCALC_H_
