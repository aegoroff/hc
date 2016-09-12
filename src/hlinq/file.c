/*!
 * \brief   The file contains file builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-11
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */


#include "file.h"
#include "filehash.h"

static FILE* file_output = NULL;

void prfile_output_both_file_and_console(out_context_t* ctx);

void file_run(file_builtin_ctx_t* ctx) {
    builtin_ctx_t* builtin_ctx = ctx->builtin_ctx_;
    
    data_ctx_t data_ctx = { 0 };
    data_ctx.HashToSearch = ctx->hash_;
    data_ctx.IsPrintCalcTime = ctx->show_time_;
    data_ctx.IsPrintLowCase = builtin_ctx->is_print_low_case_;
    data_ctx.IsPrintSfv = ctx->result_in_sfv_;
    data_ctx.IsValidateFileByHash = ctx->hash_ != NULL;
    data_ctx.IsPrintVerify = ctx->is_verify_;
    data_ctx.Limit = ctx->limit_;
    data_ctx.Offset = ctx->offset_;

    if (ctx->result_in_sfv_ && 0 != strcmp(builtin_get_hash_definition()->name_, "crc32")) {
        lib_printf("\n --sfv option doesn't support %s algorithm. Only crc32 supported", builtin_get_hash_definition()->name_);
        return;
    }
    
#ifdef GTEST
    data_ctx.PfnOutput = OutputToCppConsole;
#else
    if (ctx->save_result_path_ != NULL) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&file_output, ctx->save_result_path_, "w+");
#else
        output = fopen(ctx->save_result_path_, "w+");
#endif
        if (file_output == NULL) {
            lib_printf("\nError opening file: %s Error message: ", ctx->save_result_path_);
            perror("");
            return;
        }
        data_ctx.PfnOutput = prfile_output_both_file_and_console;
    }
    else {
        data_ctx.PfnOutput = out_output_to_console;
    }

#endif

    fhash_calculate_file(ctx->file_path_, &data_ctx, builtin_get_pool());
}

void prfile_output_both_file_and_console(out_context_t* ctx) {
    builtin_output_both_file_and_console(file_output, ctx);
}