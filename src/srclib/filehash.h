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

#include <apr.h>
#include <apr_pools.h>
#include <apr_file_io.h>
#include "output.h"

#define HASH_FILE_COLUMN_SEPARATOR "   "
#define PATH_ELT_SEPARATOR '\\'

#ifdef __cplusplus
extern "C" {
#endif

typedef struct file_hash_result_t {
    const char* File;
    const char* Size;
    const char* Hash;
    const char* CalculationTime;
    const char* ErrorMessage;
} file_hash_result_t;

typedef struct data_ctx_t {
    int IsPrintLowCase;
    int IsPrintCalcTime;
    int IsPrintSfv;
    int IsPrintVerify;
    int IsValidateFileByHash;
    int IsPrintErrorOnFind;
    const char* HashToSearch;
    apr_off_t Limit;
    apr_off_t Offset;
    void (* PfnOutput)(out_context_t* ctx);
} data_ctx_t;


void fhash_calculate_file(const char* pathToFile, data_ctx_t* ctx, apr_pool_t* pool);

// These functions must be defined in concrete calculator implementation
int fhash_compare_digests(apr_byte_t* digest1, apr_byte_t* digest2);
void fhash_to_digest(const char* hash, apr_byte_t* digest);
void fhash_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t inputLen);
void fhash_final_hash(void* context, apr_byte_t* digest);
void fhash_update_hash(void* context, const void* input, const apr_size_t inputLen);

void fhash_init_hash_context(void* context);

const char* fhash_calculate_hash(apr_file_t* fileHandle,
                                 apr_off_t fileSize,
                                 apr_byte_t* digest,
                                 apr_off_t limit,
                                 apr_off_t offset,
                                 apr_pool_t* pool);

void* fhash_allocate_context(apr_pool_t* pool);
apr_size_t fhash_get_digest_size();
int fhash_comparison_failure(int result);

#ifdef __cplusplus
}
#endif

#endif // FILEHASH_HCALC_H_
