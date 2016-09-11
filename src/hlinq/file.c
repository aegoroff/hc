
#include "file.h"
#include "filehash.h"

FILE* file_output = NULL;

void prfile_output_both_file_and_console(out_context_t* ctx);

void file_run(file_builtin_ctx_t* ctx) {
    builtin_ctx_t* builtin_ctx = ctx->builtin_ctx_;
    
    data_ctx_t data_ctx = { 0 };
    data_ctx.HashToSearch = ctx->hash_;
    data_ctx.IsPrintCalcTime = ctx->show_time_;
    data_ctx.IsPrintLowCase = builtin_ctx->is_print_low_case_;
    data_ctx.IsPrintSfv = ctx->result_in_sfv_;
    data_ctx.IsValidateFileByHash = ctx->is_verify_;
    data_ctx.Limit = ctx->limit_;
    data_ctx.Offset = ctx->offset_;
    
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
    out_output_to_console(ctx);

    lib_fprintf(file_output, "%s", ctx->string_to_print_); //-V111
    if (ctx->is_print_separator_) {
        lib_fprintf(file_output, FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->is_finish_line_) {
        lib_fprintf(file_output, NEW_LINE);
    }
}