/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains file hash interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
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
    const char* file_;
    const char* size_;
    const char* hash_;
    const char* calculation_time_;
    const char* error_message_;
} file_hash_result_t;

typedef struct data_ctx_t {
    int is_print_low_case_;
    int is_print_calc_time_;
    int is_print_sfv_;
    int is_print_verify_;
    int is_validate_file_by_hash_;
    int is_print_error_on_find_;
    const char* hash_to_search_;
    apr_off_t limit_;
    apr_off_t offset_;
    BOOL is_base64_;
    void (* pfn_output_)(out_context_t* ctx);
} data_ctx_t;

void fhash_calculate_file(const char* path_to_file, data_ctx_t* ctx, apr_pool_t* pool);

// These functions must be defined in concrete calculator implementation
int fhash_compare_digests(apr_byte_t* digest1, apr_byte_t* digest2);
void fhash_to_digest(const char* hash, apr_byte_t* digest);
void fhash_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len);
void fhash_final_hash(void* context, apr_byte_t* digest);
void fhash_update_hash(void* context, const void* input, const apr_size_t input_len);

void fhash_init_hash_context(void* context);

/**
 * \brief Calculates file hash
 * \param file_handle file handle to calculate hash of
 * \param file_size file size
 * \param digest where to store the result
 * \param limit file limit in bytes
 * \param offset file offset in bytes
 * \param pool memory pool
 * \return null in case of the calculation success or error message otherwise
 */
const char* fhash_calculate_hash(apr_file_t* file_handle,
                                 apr_off_t file_size,
                                 apr_byte_t* digest,
                                 apr_off_t limit,
                                 apr_off_t offset,
                                 apr_pool_t* pool);

void* fhash_allocate_context(apr_pool_t* pool);
apr_size_t fhash_get_digest_size(void);

#ifdef __cplusplus
}
#endif

#endif // FILEHASH_HCALC_H_
