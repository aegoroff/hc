/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains file builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-11
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#include "file.h"
#include "filehash.h"
#include "intl.h"
#ifdef GTEST
#include "displayError.h"
#endif

static FILE* file_output = NULL;

static void prfile_output_both_file_and_console(out_context_t* ctx);

void file_run(file_builtin_ctx_t* ctx) {
    builtin_ctx_t* builtin_ctx = ctx->builtin_ctx_;

    data_ctx_t data_ctx = { 0 };
    data_ctx.hash_to_search_ = ctx->hash_;
    data_ctx.is_print_calc_time_ = ctx->show_time_;
    data_ctx.is_print_low_case_ = builtin_ctx->is_print_low_case_;
    data_ctx.is_print_sfv_ = ctx->result_in_sfv_;
    data_ctx.is_validate_file_by_hash_ = ctx->hash_ != NULL;
    data_ctx.is_print_verify_ = ctx->is_verify_;
    data_ctx.limit_ = ctx->limit_;
    data_ctx.offset_ = ctx->offset_;
    data_ctx.is_base64_ = ctx->is_base64_;

    if(ctx->result_in_sfv_ && (0 != strcmp(builtin_get_hash_definition()->name_, "crc32") && 0 != strcmp(builtin_get_hash_definition()->name_, "crc32c"))) {
        lib_printf(_("\n --sfv option doesn't support %s algorithm. Only crc32 or crc32c supported"
                   ), builtin_get_hash_definition()->name_);
        return;
    }

#ifdef GTEST
    data_ctx.pfn_output_ = OutputToCppConsole;
#else
    if(ctx->save_result_path_ != NULL) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&file_output, ctx->save_result_path_, "w+");
#else
        output = fopen(ctx->save_result_path_, "w+");
#endif
        if(file_output == NULL) {
            lib_printf(_("\nError opening file: %s Error message: "), ctx->save_result_path_);
            perror("");
            return;
        }
        data_ctx.pfn_output_ = prfile_output_both_file_and_console;
    } else {
        data_ctx.pfn_output_ = out_output_to_console;
    }

#endif

    fhash_calculate_file(ctx->file_path_, &data_ctx, builtin_get_pool());
}

void prfile_output_both_file_and_console(out_context_t* ctx) {
    builtin_output_both_file_and_console(file_output, ctx);
}
