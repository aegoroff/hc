/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains output implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#include <assert.h>
#include <apr_errno.h>
#include <apr_strings.h>
#include "output.h"
#include "tomcrypt.h"
#include "b64.h"

const char* out_create_error_message(apr_status_t status, apr_pool_t* pool) {
    char* message = (char*)apr_pcalloc(pool, ERROR_BUFFER_SIZE);
    apr_strerror(status, message, ERROR_BUFFER_SIZE);
    return message;
}

void out_output_error_message(apr_status_t status, void (* pfn_output)(out_context_t* ctx), apr_pool_t* pool) {
    // ReSharper disable once CppInitializedValueIsAlwaysRewritten
    out_context_t ctx = { 0 };
    ctx.string_to_print_ = out_create_error_message(status, pool);
    ctx.is_print_separator_ = FALSE;
    ctx.is_finish_line_ = TRUE;
    pfn_output(&ctx);
}

void out_print_error(apr_status_t status) {
    char errbuf[ERROR_BUFFER_SIZE];
    apr_strerror(status, errbuf, ERROR_BUFFER_SIZE);
    lib_printf("%s", errbuf); //-V111
    lib_new_line();
}

const char* out_copy_size_to_string(uint64_t size, apr_pool_t* pool) {
    size_t sz = 64;
    char* str = (char*)apr_pcalloc(pool, sz);
    lib_size_to_string(size, str);
    return str;
}

const char* out_copy_time_to_string(const lib_time_t* time, apr_pool_t* pool) {
    size_t sz = 48;
    char* str = (char*)apr_pcalloc(pool, sz);
    lib_time_to_string(time, str);
    return str;
}

const char* out_hash_to_string(apr_byte_t* digest, int is_print_low_case, apr_size_t sz, apr_pool_t* pool) {
    apr_size_t i = 0;
    char* str = (char*)apr_pcalloc(pool, sz * BYTE_CHARS_SIZE + 1); // iteration ponter
    char* result = str; // result pointer

    for(; i < sz; ++i) {
        apr_snprintf(str, BYTE_CHARS_SIZE + 1, is_print_low_case ? HEX_LOWER : HEX_UPPER, digest[i]);
        str += BYTE_CHARS_SIZE;
    }
    return result;
}

const char* out_hash_to_base64_string(apr_byte_t* digest, apr_size_t sz, apr_pool_t* pool) {
    size_t outlen = 0;
    char* result = b64_encode(digest, sz, &outlen, pool);
    return result;
}

void out_output_to_console(out_context_t* ctx) {
    if(ctx == NULL) {
        assert(ctx != NULL);
        return;
    }
    lib_printf("%s", ctx->string_to_print_); //-V111
    if(ctx->is_print_separator_) {
        lib_printf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if(ctx->is_finish_line_) {
        lib_new_line();
    }
}
