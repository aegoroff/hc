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
#include "gost.h"
#include "../srclib/bf.h"
#include "../srclib/encoding.h"
#include "../linq2hash/hashes.h"
#ifdef GTEST
    #include "displayError.h"
#endif

#define MAX_ATTR "max"
#define MIN_ATTR "min"
#define DICT_ATTR "dict"
#define ARRAY_INIT_SZ           32
#define UNKNOWN_IDENTIFIER "unknown identifier"

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_pool_t* filePool = NULL;
apr_hash_t* ht = NULL;
apr_hash_t* htFileDigestCache = NULL;
const char* fileParameter = NULL;

statement_ctx_t* statement = NULL;

apr_size_t hashLength = 0;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

// Forward declarations
void* FileAlloc(size_t size);
void PrintFileInfo(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* p);
void RunString(data_ctx_t* dataCtx);
void RunDir(data_ctx_t* dataCtx);
void RunFile(data_ctx_t* dataCtx);
void RunHash();
void CalculateFile(const char* pathToFile, data_ctx_t* ctx, apr_pool_t* pool);
BOOL FilterFiles(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* p);
BOOL FilterFilesInternal(void* ctx, apr_pool_t* p);

const char* Trim(const char* str);
void* GetContext();

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

    ht = apr_hash_make(statementPool);

    if(ht == NULL) {
    destroyPool:
        apr_pool_destroy(statementPool);
        statementPool = NULL;
        return;
    }
    statement = (statement_ctx_t*)apr_pcalloc(statementPool, sizeof(statement_ctx_t));
    if(statement == NULL) {
        goto destroyPool;
    }
    statement->Type = CtxTypeUndefined;
    statement->HashAlgorithm = NULL;
}

void OutputBothFileAndConsole(out_context_t* ctx) {
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
        dataCtx.PfnOutput = OutputBothFileAndConsole;
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
        case CtxTypeString:
            RunString(&dataCtx);
            break;
        case CtxTypeHash:
            RunHash();
            break;
        case CtxTypeDir:
            RunDir(&dataCtx);
            break;
        case CtxTypeFile:
            RunFile(&dataCtx);
            break;
    }

cleanup:
    if(output != NULL) {
        fclose(output);
        output = NULL;
    }
    apr_pool_destroy(statementPool);
    statementPool = NULL;
    ht = NULL;
    statement = NULL;
}

void RunHash() {
    string_statement_ctx_t* ctx = cpl_get_string_context();

    if((NULL == ctx) || (statement->HashAlgorithm == NULL) || !(ctx->BruteForce)) {
        return;
    }

    hashLength = statement->HashAlgorithm->hash_length_;

    bf_crack_hash(ctx->Dictionary,
              statement->Source,
              ctx->Min,
              ctx->Max,
              hashLength,
              statement->HashAlgorithm->pfn_digest_,
              options->NoProbe,
              options->NumOfThreads,
              statement->HashAlgorithm->use_wide_string_,
              statementPool);
}

void RunString(data_ctx_t* dataCtx) {
    apr_byte_t* digest = NULL;
    apr_size_t sz = 0;
    out_context_t o = {0};

    if(statement->HashAlgorithm == NULL) {
        return;
    }
    sz = statement->HashAlgorithm->hash_length_;
    digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * sz);

    if(statement->HashAlgorithm->use_wide_string_) {
        wchar_t* str = enc_from_ansi_to_unicode(statement->Source, statementPool);
        statement->HashAlgorithm->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
    }
    else {
        statement->HashAlgorithm->pfn_digest_(digest, statement->Source, strlen(statement->Source));
    }

    o.is_finish_line_ = TRUE;
    o.is_print_separator_ = FALSE;
    o.string_to_print_ = out_hash_to_string(digest, dataCtx->IsPrintLowCase, sz, pool);
    dataCtx->PfnOutput(&o);
}

void RunDir(data_ctx_t* dataCtx) {
    traverse_ctx_t dirContext = {0};
    dir_statement_ctx_t* ctx = cpl_get_dir_context();
    BOOL (* filter)(apr_finfo_t* info, const char* dir, traverse_ctx_t* c, apr_pool_t* pool) = FilterFiles;

    if(NULL == ctx) {
        return;
    }

    dataCtx->Limit = ctx->limit_;
    dataCtx->Offset = ctx->offset_;
    if(ctx->hash_to_search_ != NULL) {
        dataCtx->HashToSearch = ctx->hash_to_search_;
    }

    if(ctx->find_files_) {
        dirContext.pfn_file_handler = PrintFileInfo;
    }
    else if(statement->HashAlgorithm == NULL) {
        return;
    }
    else {
        dirContext.pfn_file_handler = CalculateFile;
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

void RunFile(data_ctx_t* dataCtx) {
    dir_statement_ctx_t* ctx = cpl_get_dir_context();

    dataCtx->Limit = ctx->limit_;
    dataCtx->Offset = ctx->offset_;
    dataCtx->HashToSearch = ctx->hash_to_search_;
    dataCtx->IsValidateFileByHash = TRUE;
    if(fileParameter != NULL) {
        apr_file_t* fileHandle = NULL;
        apr_status_t status = APR_SUCCESS;
        apr_finfo_t info = {0};
        file_ctx_t fileCtx = {0};
        char* dir = NULL;
        out_context_t outputCtx = {0};
        char* fileAnsi = NULL;

        statement->Source = fileParameter;
        /*
            1. Extract dir from path
            2. Open file
            3. Make necessary context
            4. Run filtering files internal function
            5. Output result
         */

        status = apr_file_open(&fileHandle,
                               statement->Source,
                               APR_READ | APR_BINARY,
                               APR_FPROT_WREAD,
                               pool);
        if(status != APR_SUCCESS) {
            if(dataCtx->IsPrintErrorOnFind) {
                out_output_error_message(status, dataCtx->PfnOutput, statementPool);
            }
            return;
        }
        status = apr_file_info_get(
            &info,
            APR_FINFO_NAME | APR_FINFO_SIZE | APR_FINFO_IDENT |
            APR_FINFO_TYPE,
            fileHandle);
        if(status != APR_SUCCESS) {
            if(dataCtx->IsPrintErrorOnFind) {
                out_output_error_message(status, dataCtx->PfnOutput, statementPool);
            }
            goto cleanup;
        }
        status = apr_filepath_root(&dir, &fileParameter, APR_FILEPATH_NATIVE, statementPool);

        if(status == APR_ERELATIVE) {
            status = apr_filepath_get(&dir, APR_FILEPATH_NATIVE, statementPool);
            if(status != APR_SUCCESS) {
                if(dataCtx->IsPrintErrorOnFind) {
                    out_output_error_message(status, dataCtx->PfnOutput, statementPool);
                }
                goto cleanup;
            }
        }
        else if(status != APR_SUCCESS) {
            if(dataCtx->IsPrintErrorOnFind) {
                out_output_error_message(status, dataCtx->PfnOutput, statementPool);
            }
            goto cleanup;
        }

        fileCtx.Dir = dir;
        info.name = info.fname;
        fileCtx.Info = &info;
        fileCtx.PfnOutput = dataCtx->PfnOutput;

        fileAnsi = enc_from_utf8_to_ansi(statement->Source, statementPool);
        outputCtx.string_to_print_ = fileAnsi == NULL ? statement->Source : fileAnsi;
        outputCtx.is_print_separator_ = TRUE;
        dataCtx->PfnOutput(&outputCtx);

        outputCtx.is_print_separator_ = TRUE;
        outputCtx.is_finish_line_ = FALSE;
        outputCtx.string_to_print_ = out_copy_size_to_string(info.size, statementPool);
        dataCtx->PfnOutput(&outputCtx);

        outputCtx.string_to_print_ = "File is ";
        outputCtx.is_print_separator_ = FALSE;
        dataCtx->PfnOutput(&outputCtx);

        if(FilterFilesInternal(&fileCtx, statementPool)) {
            outputCtx.string_to_print_ = "valid";
        }
        else {
            outputCtx.string_to_print_ = "invalid";
        }
        dataCtx->PfnOutput(&outputCtx);
    cleanup:
        status = apr_file_close(fileHandle);
        if(status != APR_SUCCESS && dataCtx->IsPrintErrorOnFind) {
            out_output_error_message(status, dataCtx->PfnOutput, statementPool);
        }
        return;
    }
    if(statement->HashAlgorithm == NULL) {
        return;
    }
    CalculateFile(statement->Source, dataCtx, statementPool);
}

BOOL FilterFilesInternal(void* ctx, apr_pool_t* p) {
    return TRUE;
}

void cpl_set_recursively() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    cpl_get_dir_context()->recursively_ = TRUE;
}

void cpl_set_brute_force() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    if(statement->Type != CtxTypeHash) {
        return;
    }
    cpl_get_string_context()->BruteForce = TRUE;
    if(cpl_get_string_context()->Min == 0) {
        cpl_get_string_context()->Min = 1;
    }
    if(cpl_get_string_context()->Max == 0) {
        cpl_get_string_context()->Max = MAX_DEFAULT;
    }
    if(cpl_get_string_context()->Dictionary == NULL) {
        cpl_get_string_context()->Dictionary = alphabet;
    }
}

void cpl_register_identifier(const char* identifier) {
    void* ctx = NULL;

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    switch (statement->Type) {
    case CtxTypeDir:
    case CtxTypeFile:
        ctx = apr_pcalloc(statementPool, sizeof(dir_statement_ctx_t));
        ((dir_statement_ctx_t*)ctx)->limit_ = INT64_MAX;
        break;
    case CtxTypeString:
    case CtxTypeHash:
        ctx = apr_pcalloc(statementPool, sizeof(string_statement_ctx_t));
        ((string_statement_ctx_t*)ctx)->BruteForce = FALSE;
        break;
    }
    statement->Id = (const char*)identifier;
    apr_hash_set(ht, statement->Id, APR_HASH_KEY_STRING, ctx);
}

void cpl_define_query_type(ctx_type_t type) {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    statement->Type = type;
}

void* GetContext() {
    if(NULL == statement->Id) {
        return NULL;
    }
    return apr_hash_get(ht, statement->Id, APR_HASH_KEY_STRING);
}

dir_statement_ctx_t* cpl_get_dir_context() {
    return (dir_statement_ctx_t*)GetContext();
}

string_statement_ctx_t* cpl_get_string_context() {
    return (string_statement_ctx_t*)GetContext();
}

void cpl_set_source(const char* str, void* token) {
    statement->Source = Trim(str);
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

BOOL IsStringBorder(const char* str, size_t ix) {
    return str[ix] == '\'' || str[ix] == '\"';
}

const char* Trim(const char* str) {
    size_t len = 0;
    char* tmp = NULL;

    if(!str) {
        return NULL;
    }
    tmp = apr_pstrdup(statementPool, (char*)str);

    if(IsStringBorder(str, 0)) {
        tmp = tmp + 1; // leading " or '
    }
    len = strlen(tmp);
    if((len > 0) && IsStringBorder((const char*)tmp, len - 1)) {
        tmp[len - 1] = '\0'; // trailing " or '
    }
    return tmp;
}

/*!
 * It's so ugly to improve performance
 */
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2) {
    return memcmp(digest1, digest2, hashLength) == 0;
}

int ComparisonFailure(int result) {
    return cpl_get_dir_context()->operation_ == CondOpEq ? !result : result;
}

int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length) {
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    statement->HashAlgorithm->pfn_digest_(attempt, pass, (apr_size_t)length);
    return CompareDigests(attempt, hash);
}

void ToDigest(const char* hash, apr_byte_t* digest) {
    lib_hex_str_2_byte_array(hash, digest, hashLength);
}

void* bf_create_digest(const char* hash, apr_pool_t* p) {
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(p, hashLength);
    ToDigest(hash, result);
    return result;
}

void CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen) {
    statement->HashAlgorithm->pfn_digest_(digest, input, inputLen);
}

void InitContext(void* context) {
    statement->HashAlgorithm->pfn_init_(context);
}

void FinalHash(apr_byte_t* digest, void* context) {
    statement->HashAlgorithm->pfn_final_(digest, context);
}

void UpdateHash(void* context, const void* input, const apr_size_t inputLen) {
    statement->HashAlgorithm->pfn_update_(context, input, inputLen);
}

void* AllocateContext(apr_pool_t* p) {
    return apr_pcalloc(p, statement->HashAlgorithm->context_size_);
}

apr_size_t GetDigestSize() {
    return statement->HashAlgorithm->hash_length_;
}

int bf_compare_hash(apr_byte_t* digest, const char* checkSum) {
    apr_byte_t* bytes = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * GetDigestSize());

    ToDigest(checkSum, bytes);
    return CompareDigests(bytes, digest);
}

BOOL FilterFiles(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* p) {
    file_ctx_t fileCtx = {0};
    fileCtx.Dir = dir;
    fileCtx.Info = info;
    fileCtx.PfnOutput = ((data_ctx_t*)ctx->data_ctx)->PfnOutput;
    return FilterFilesInternal(&fileCtx, p);
}

void PrintFileInfo(const char* fullPathToFile, data_ctx_t* ctx, apr_pool_t* p) {
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