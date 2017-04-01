/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains output interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef OUTPUT_HCALC_H_
#define OUTPUT_HCALC_H_

#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"

#define FILE_INFO_COLUMN_SEPARATOR " | "

#ifdef __cplusplus

extern "C" {
#endif

#include <apr.h>
#include <apr_pools.h>
#include "lib.h"

typedef struct out_context_t {
    int         is_print_separator_;
    int         is_finish_line_;
    const char* string_to_print_;
} out_context_t;

void out_output_error_message(apr_status_t status, void (* pfn_output)(
        out_context_t* ctx), apr_pool_t * pool);

const char* out_create_error_message(apr_status_t status, apr_pool_t* pool);

void        out_print_error(apr_status_t status);

const char* out_copy_size_to_string(uint64_t size, apr_pool_t* pool);
const char* out_copy_time_to_string(lib_time_t time, apr_pool_t* pool);
const char* out_hash_to_string(apr_byte_t* digest, int is_print_low_case, apr_size_t sz, apr_pool_t* pool);
void out_output_to_console(out_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // OUTPUT_HCALC_H_
