/*!
 * \brief   The file contains HLINQ compiler API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include <math.h>
#include <io.h>
#include <stdint.h>
#include "compiler.h"
#include "apr_hash.h"
#include "apr_strings.h"
#include "../srclib/bf.h"
#include "../srclib/encoding.h"
#include "../linq2hash/hashes.h"
#ifdef GTEST
    #include "displayError.h"
#endif

#define ARRAY_INIT_SZ           32

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_pool_t* filePool = NULL;
apr_hash_t* htFileDigestCache = NULL;
const char* fileParameter = NULL;

statement_ctx_t* statement = NULL;
void* cpl_mode_context;

apr_size_t hashLength = 0;

// Forward declarations

void prcpl_print_file_info(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* p);
void prcpl_run_dir(data_ctx_t* dataCtx);

/**
 * \brief trims string by removing lead and trail ' or "
 * \param str string to trim
 * \return cleaned string
 */
const char* prcpl_trim(const char* str);

void* prcpl_get_context();

program_options_t* options = NULL;
FILE* output = NULL;

void cpl_init_program(program_options_t* po, const char* fileParam, apr_pool_t* root) {
    options = po;
    fileParameter = fileParam;
    apr_pool_create(&pool, root);
}

void cpl_open_statement() {
    apr_status_t status = apr_pool_create(&statementPool, pool);

    if(status != APR_SUCCESS) {
        statementPool = NULL;
        return;
    }

    statement = (statement_ctx_t*)apr_pcalloc(statementPool, sizeof(statement_ctx_t));
    if(statement == NULL) {
        apr_pool_destroy(statementPool);
        return;
    }
    statement->Type = CtxTypeUndefined;
    statement->HashAlgorithm = NULL;
}

void prcpl_output_both_file_and_console(out_context_t* ctx) {
    out_output_to_console(ctx);

    lib_fprintf(output, "%s", ctx->string_to_print_); //-V111
    if(ctx->is_print_separator_) {
        lib_fprintf(output, FILE_INFO_COLUMN_SEPARATOR);
    }
    if(ctx->is_finish_line_) {
        lib_fprintf(output, NEW_LINE);
    }
}

void cpl_close_statement(void) {
    data_ctx_t dataCtx = {0};

    if(statementPool == NULL) { // memory allocation error
        return;
    }

    if(options->OnlyValidate) {
        goto cleanup;
    }

#ifdef GTEST
    dataCtx.PfnOutput = OutputToCppConsole;
#else
    if(options->FileToSave != NULL) {
#ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&output, options->FileToSave, "w+");
#else
        output = fopen(options->FileToSave, "w+");
#endif
        if(output == NULL) {
            lib_printf("\nError opening file: %s Error message: ", options->FileToSave);
            perror("");
            goto cleanup;
        }
        dataCtx.PfnOutput = prcpl_output_both_file_and_console;
    }
    else {
        dataCtx.PfnOutput = out_output_to_console;
    }

#endif

    if(options->PrintSfv && 0 != strcmp(statement->HashAlgorithm->name_, "crc32")) {
        lib_printf("\n --sfv option doesn't support %s algorithm. Only crc32 supported", statement->HashAlgorithm->name_);
        goto cleanup;
    }

    dataCtx.IsPrintCalcTime = options->PrintCalcTime;
    dataCtx.IsPrintLowCase = options->PrintLowCase;
    dataCtx.IsPrintSfv = options->PrintSfv;
    dataCtx.IsPrintVerify = options->PrintVerify;
    dataCtx.IsPrintErrorOnFind = !(options->NoErrorOnFind);

    switch(statement->Type) {
        case CtxTypeDir:
            prcpl_run_dir(&dataCtx);
            break;
    }

cleanup:
    if(output != NULL) {
        fclose(output);
        output = NULL;
    }
    apr_pool_destroy(statementPool);
    statementPool = NULL;
    statement = NULL;
}

void prcpl_run_dir(data_ctx_t* dataCtx) {
    traverse_ctx_t dirContext = {0};
    dir_statement_ctx_t* ctx = cpl_get_dir_context();
    BOOL (* filter)(apr_finfo_t* info, const char* dir, traverse_ctx_t* c, apr_pool_t* pool) = NULL;

    if(NULL == ctx) {
        return;
    }

    dataCtx->Limit = ctx->limit_;
    dataCtx->Offset = ctx->offset_;
    if(ctx->hash_to_search_ != NULL) {
        dataCtx->HashToSearch = ctx->hash_to_search_;
    }

    if(ctx->find_files_) {
        dirContext.pfn_file_handler = prcpl_print_file_info;
    }
    else if(statement->HashAlgorithm == NULL) {
        return;
    }
    else {
        dirContext.pfn_file_handler = fhash_calculate_file;
    }

    dirContext.data_ctx = dataCtx;
    dirContext.is_scan_dir_recursively = ctx->recursively_;

    if(ctx->include_pattern_ != NULL || ctx->exclude_pattern_ != NULL) {
        traverse_compile_pattern(ctx->include_pattern_, &dirContext.include_pattern, statementPool);
        traverse_compile_pattern(ctx->exclude_pattern_, &dirContext.exclude_pattern, statementPool);
        filter = traverse_filter_by_name;
    }

    traverse_directory(traverse_hack_root_path(statement->Source,
                                   statementPool), &dirContext, filter, statementPool);
}

void cpl_set_recursively() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    cpl_get_dir_context()->recursively_ = TRUE;
}

void cpl_define_query_type(ctx_type_t type) {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    statement->Type = type;

    switch (statement->Type) {
    case CtxTypeDir:
    case CtxTypeFile:
        cpl_mode_context = apr_pcalloc(statementPool, sizeof(dir_statement_ctx_t));
        ((dir_statement_ctx_t*)cpl_mode_context)->limit_ = INT64_MAX;
        break;
    }
}

dir_statement_ctx_t* cpl_get_dir_context() {
    return (dir_statement_ctx_t*)cpl_mode_context;
}

void cpl_set_source(const char* str, void* token) {
    statement->Source = prcpl_trim(str);
}

void cpl_set_hash_algorithm_into_context(const char* str) {
    hash_definition_t* algorithm = NULL;
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    algorithm = hsh_get_hash((const char*)str);
    if(algorithm == NULL) {
        return;
    }

    statement->HashAlgorithm = algorithm;
    hashLength = algorithm->hash_length_;
}

BOOL prcpl_is_string_border(const char* str, size_t ix) {
    return str[ix] == '\'' || str[ix] == '\"';
}

const char* prcpl_trim(const char* str) {
    size_t len = 0;
    char* tmp = NULL;

    if(!str) {
        return NULL;
    }
    tmp = apr_pstrdup(statementPool, (char*)str);

    if(prcpl_is_string_border(str, 0)) {
        tmp = tmp + 1; // leading " or '
    }
    len = strlen(tmp);
    if(len > 0 && prcpl_is_string_border((const char*)tmp, len - 1)) {
        tmp[len - 1] = '\0'; // trailing " or '
    }
    return tmp;
}

void prcpl_print_file_info(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* p) {
    out_context_t outputCtx = {0};
    char* fileAnsi = NULL;
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = {0};

    fileAnsi = enc_from_utf8_to_ansi(fullPathToFile, p);

    apr_file_open(&fileHandle, fullPathToFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, fileHandle);

    outputCtx.is_finish_line_ = FALSE;
    outputCtx.is_print_separator_ = TRUE;

    // file name
    outputCtx.string_to_print_ = fileAnsi == NULL ? fullPathToFile : fileAnsi;
    ctx->PfnOutput(&outputCtx);

    // file size
    outputCtx.string_to_print_ = out_copy_size_to_string(info.size, p);

    outputCtx.is_finish_line_ = TRUE;
    outputCtx.is_print_separator_ = FALSE;
    ctx->PfnOutput(&outputCtx); // file size or time output
    apr_file_close(fileHandle);
}