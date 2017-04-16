/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains directory traverse implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-23
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "traverse.h"
#include <apr_fnmatch.h>
#include <apr_tables.h>
#include <apr_file_info.h>
#include <apr_strings.h>
#include "encoding.h"

#define COMPOSITE_PATTERN_INIT_SZ 8 // composite pattern array init size
#define SUBDIRS_ARRAY_INIT_SZ 16 // subdirectories array init size
#define PATTERN_SEPARATOR ";"

int traverse_match_to_composite_pattern(const char* str, apr_array_header_t* pattern) {
    int i = 0;

    if(!pattern) {
        return TRUE; // important
    }
    if(!str) {
        return FALSE; // important
    }

    for(; i < pattern->nelts; ++i) {
        const char* p = ((const char**)pattern->elts)[i];
        if(apr_fnmatch(p, str, APR_FNM_CASE_BLIND) == APR_SUCCESS) {
            return TRUE;
        }
    }

    return FALSE;
}

const char* traverse_hack_root_path(const char* path, apr_pool_t* pool) {
    size_t len;

    if(path == NULL) {
        return path;
    }
    len = strlen(path);
    return path[len - 1] == ':' ? apr_pstrcat(pool, path, "\\", NULL) : path;
}

void traverse_compile_pattern(const char* pattern, apr_array_header_t** newpattern, apr_pool_t* pool) {
    char* parts;
    char* last = NULL;
    char* p;

    if(!pattern) {
        return; // important
    }

    *newpattern = apr_array_make(pool, COMPOSITE_PATTERN_INIT_SZ, sizeof(const char*));

    parts = apr_pstrdup(pool, pattern); /* strtok wants non-const data */
    p = apr_strtok(parts, PATTERN_SEPARATOR, &last);
    while(p) {
        *(const char**)apr_array_push(*newpattern) = p;
        p = apr_strtok(NULL, PATTERN_SEPARATOR, &last);
    }
}

void traverse_directory(
    const char* dir,
    traverse_ctx_t* ctx,
    BOOL (*filter)(apr_finfo_t* info, const char* d, traverse_ctx_t* c, apr_pool_t* pool),
    apr_pool_t* pool) {
    apr_finfo_t info = { 0 };
    apr_dir_t* d = NULL;
    apr_status_t status;
    char* full_path = NULL; // Full path to file or subdirectory
    apr_pool_t* iter_pool = NULL;
    apr_array_header_t* subdirs = NULL;
    out_context_t output = { 0 };

    if(ctx->pfn_file_handler == NULL || dir == NULL) {
        return;
    }

    status = apr_dir_open(&d, dir, pool);
    if(status != APR_SUCCESS) {
        if(((data_ctx_t*)ctx->data_ctx)->is_print_error_on_find_) {
            output.string_to_print_ = enc_from_utf8_to_ansi(dir, pool);
            output.is_print_separator_ = TRUE;
            ((data_ctx_t*)ctx->data_ctx)->pfn_output_(&output);
            out_output_error_message(status, ((data_ctx_t*)ctx->data_ctx)->pfn_output_, pool);
        }
        return;
    }

    if(ctx->is_scan_dir_recursively) {
        subdirs = apr_array_make(pool, SUBDIRS_ARRAY_INIT_SZ, sizeof(const char*));
    }

    apr_pool_create(&iter_pool, pool);
    for(;;) {
        apr_pool_clear(iter_pool); // cleanup file allocated memory
        status = apr_dir_read(&info, APR_FINFO_NAME | APR_FINFO_MIN, d);
        if(APR_STATUS_IS_ENOENT(status)) {
            break;
        }
        if(info.name == NULL) { // to avoid access violation
            if(((data_ctx_t*)ctx->data_ctx)->is_print_error_on_find_) {
                out_output_error_message(status, ((data_ctx_t*)ctx->data_ctx)->pfn_output_, pool);
            }
            continue;
        }
        // Subdirectory handling code
        if(info.filetype == APR_DIR && ctx->is_scan_dir_recursively) {
            // skip current and parent dir
            if(info.name[0] == '.' && info.name[1] == '\0' || info.name[0] == '.' && info.name[1] == '.' && info.name[2] == '\0') {
                continue;
            }

            status = apr_filepath_merge(&full_path,
                                        dir,
                                        info.name,
                                        APR_FILEPATH_NATIVE,
                                        pool); // IMPORTANT: so as not to use strdup
            if(status != APR_SUCCESS) {
                if(((data_ctx_t*)ctx->data_ctx)->is_print_error_on_find_) {
                    out_output_error_message(status, ((data_ctx_t*)ctx->data_ctx)->pfn_output_, pool);
                }
                continue;
            }
            *(const char**)apr_array_push(subdirs) = full_path;
        } // End subdirectory handling code

        if(status != APR_SUCCESS || info.filetype != APR_REG) {
            continue;
        }

        if(filter != NULL && !filter(&info, dir, ctx, iter_pool)) {
            continue;
        }

        status = apr_filepath_merge(&full_path,
                                    dir,
                                    info.name,
                                    APR_FILEPATH_NATIVE,
                                    iter_pool);
        if(status != APR_SUCCESS) {
            if(((data_ctx_t*)ctx->data_ctx)->is_print_error_on_find_) {
                out_output_error_message(status, ((data_ctx_t*)ctx->data_ctx)->pfn_output_, pool);
            }
            continue;
        }

        ctx->pfn_file_handler(full_path, ctx->data_ctx, iter_pool);
    }

    status = apr_dir_close(d);
    if(status != APR_SUCCESS) {
        out_output_error_message(status, ((data_ctx_t*)ctx->data_ctx)->pfn_output_, pool);
    }

    // scan subdirectories found
    if(ctx->is_scan_dir_recursively) {
        size_t i = 0;
        for(; i < subdirs->nelts; ++i) {
            const char* path = ((const char**)subdirs->elts)[i];
            apr_pool_clear(iter_pool);
            traverse_directory(path, ctx, filter, iter_pool);
        }
    }

    apr_pool_destroy(iter_pool);
}

BOOL traverse_filter_by_name(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool) {
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(dir);
    UNREFERENCED_PARAMETER(pool);
#endif
    if(!traverse_match_to_composite_pattern(info->name, ctx->include_pattern)) {
        return FALSE;
    }
    // IMPORTANT: check pointer here otherwise the logic will fail
    if(ctx->exclude_pattern &&
        traverse_match_to_composite_pattern(info->name, ctx->exclude_pattern)) {
        return FALSE;
    }
    return TRUE;
}
