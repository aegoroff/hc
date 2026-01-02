/*!
 * \brief   The file contains directory builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-11
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include "dir.h"
#include "traverse.h"
#include <apr_strings.h>
#include "encoding.h"
#include "intl.h"
#ifdef GTEST
#include "displayError.h"
#endif

static FILE* dir_output = NULL;
static apr_pool_t* dir_pool;
static dir_builtin_ctx_t* dir_ctx;

static void prdir_output_both_file_and_console(out_context_t* ctx);
static BOOL prdir_is_string_border(const char* str, size_t ix);
static const char* prdir_trim(const char* str);
static void prdir_print_file_info(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* p);
static BOOL prdir_filter_by_hash(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool);
static BOOL prdir_filter_by_hash_and_name(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool);

void dir_run(dir_builtin_ctx_t* ctx) {
    builtin_ctx_t* builtin_ctx = ctx->builtin_ctx_;
    dir_ctx = ctx;

    data_ctx_t data_ctx = { 0 };
    traverse_ctx_t traverse_ctx = { 0 };
    BOOL (*filter)(apr_finfo_t* info, const char* dir, traverse_ctx_t* c, apr_pool_t* pool) = NULL;
    const char* dir;
    dir_pool = builtin_get_pool();

    data_ctx.hash_to_search_ = ctx->hash_;
    data_ctx.is_print_calc_time_ = ctx->show_time_;
    data_ctx.is_print_low_case_ = builtin_ctx->is_print_low_case_;
    data_ctx.is_print_sfv_ = ctx->result_in_sfv_;
    data_ctx.is_validate_file_by_hash_ = ctx->is_verify_;
    data_ctx.is_print_verify_ = ctx->is_verify_;
    data_ctx.limit_ = ctx->limit_;
    data_ctx.offset_ = ctx->offset_;
    data_ctx.is_base64_ = ctx->is_base64_;

    if(ctx->search_hash_ != NULL) {
        data_ctx.hash_to_search_ = ctx->search_hash_;
    }

    if(!builtin_allow_sfv_option(ctx->result_in_sfv_)) {
        return;
    }

    if(ctx->search_hash_ != NULL) {
        traverse_ctx.pfn_file_handler = (void(*)(const char*, void*, apr_pool_t*))prdir_print_file_info;
        filter = prdir_filter_by_hash;
    } else {
        traverse_ctx.pfn_file_handler = (void(*)(const char*, void*, apr_pool_t*))fhash_calculate_file;
    }

#ifdef GTEST
    data_ctx.pfn_output_ = OutputToCppConsole;
#else
    if(ctx->save_result_path_ != NULL) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&dir_output, ctx->save_result_path_, "w+");
#else
        dir_output = fopen(ctx->save_result_path_, "w+");
#endif
        if(dir_output == NULL) {
            lib_printf(_("\nError opening file: %s Error message: "), ctx->save_result_path_);
            perror("");
            return;
        }
        data_ctx.pfn_output_ = prdir_output_both_file_and_console;
    } else {
        data_ctx.pfn_output_ = out_output_to_console;
    }

    traverse_ctx.data_ctx = &data_ctx;
    traverse_ctx.is_scan_dir_recursively = ctx->recursively_;

    if(ctx->include_pattern_ != NULL || ctx->exclude_pattern_ != NULL) {
        traverse_compile_pattern(ctx->include_pattern_, &traverse_ctx.include_pattern, dir_pool);
        traverse_compile_pattern(ctx->exclude_pattern_, &traverse_ctx.exclude_pattern, dir_pool);
        filter = ctx->search_hash_ == NULL ? traverse_filter_by_name : prdir_filter_by_hash_and_name;
    }

    dir = prdir_trim(ctx->dir_path_);

    traverse_directory(
                       traverse_hack_root_path(dir, dir_pool),
                       &traverse_ctx,
                       filter,
                       dir_pool);

#endif
}

void prdir_output_both_file_and_console(out_context_t* ctx) {
    builtin_output_both_file_and_console(dir_output, ctx);
}

BOOL prdir_is_string_border(const char* str, size_t ix) {
    return str[ix] == '\'' || str[ix] == '\"';
}

const char* prdir_trim(const char* str) {
    size_t len = 0;
    char* tmp = NULL;

    if(!str) {
        return NULL;
    }
    tmp = apr_pstrdup(dir_pool, (char*)str);

    if(prdir_is_string_border(str, 0)) {
        tmp = tmp + 1; // leading " or '
    }
    len = strlen(tmp);
    if(len > 0 && prdir_is_string_border((const char*)tmp, len - 1)) {
        tmp[len - 1] = '\0'; // trailing " or '
    }
    return tmp;
}

void prdir_print_file_info(const char* full_path_to_file, data_ctx_t* ctx, apr_pool_t* p) {
    out_context_t out_context = { 0 };
    apr_file_t* file_handle = NULL;
    apr_finfo_t info = { 0 };

    char* file_ansi = enc_from_utf8_to_ansi(full_path_to_file, p);

    apr_file_open(&file_handle, full_path_to_file, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, file_handle);

    out_context.is_finish_line_ = FALSE;
    out_context.is_print_separator_ = TRUE;

    // file name
    out_context.string_to_print_ = file_ansi == NULL ? full_path_to_file : file_ansi;
    ctx->pfn_output_(&out_context);

    // file size
    out_context.string_to_print_ = out_copy_size_to_string(info.size, p);

    out_context.is_finish_line_ = TRUE;
    out_context.is_print_separator_ = FALSE;
    ctx->pfn_output_(&out_context); // file size or time output
    apr_file_close(file_handle);
}

BOOL prdir_filter_by_hash_and_name(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool) {
    return traverse_filter_by_name(info, dir, ctx, pool) && prdir_filter_by_hash(info, dir, ctx, pool);
}

BOOL prdir_filter_by_hash(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool) {
    apr_status_t status = APR_SUCCESS;
    apr_file_t* file_handle = NULL;
    apr_byte_t* digest_to_compare = NULL;
    apr_byte_t* digest = NULL;

    char* full_path = NULL; // Full path to file or subdirectory

    digest = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * builtin_get_hash_definition()->hash_length_);
    digest_to_compare = (apr_byte_t*)
            apr_pcalloc(pool, sizeof(apr_byte_t) * builtin_get_hash_definition()->hash_length_);

    fhash_to_digest(dir_ctx->search_hash_, digest_to_compare);

    fhash_calculate_digest(digest, "", 0);

    // Empty file optimization
    if(fhash_compare_digests(digest, digest_to_compare) && info->size == 0) {
        return TRUE;
    }

    apr_filepath_merge(&full_path,
                       dir,
                       info->name,
                       APR_FILEPATH_NATIVE,
                       pool); // IMPORTANT: so as not to use strdup

    status = apr_file_open(&file_handle, full_path, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
    if(status != APR_SUCCESS) {
        return FALSE;
    }

    fhash_calculate_hash(file_handle, info->size, digest, dir_ctx->limit_, dir_ctx->offset_, pool);
    apr_file_close(file_handle);

    return fhash_compare_digests(digest, digest_to_compare);
}
