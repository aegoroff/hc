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

#define PCRE2_CODE_UNIT_WIDTH 8

#include <math.h>
#include <io.h>
#include "compiler.h"
#include "apr_hash.h"
#include "apr_strings.h"
#include "gost.h"
#include "pcre2.h"
#include "..\srclib\bf.h"
#include "..\srclib\encoding.h"
#include <basetsd.h>
#include "../linq2hash/hashes.h"
#include "../linq2hash/backend.h"
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

StatementCtx* statement = NULL;

apr_size_t hashLength = 0;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

// Forward declarations
void* FileAlloc(size_t size);
void PrintFileInfo(const char* fullPathToFile, DataContext* ctx, apr_pool_t* p);
void RunString(DataContext* dataCtx);
void RunDir(DataContext* dataCtx);
void RunFile(DataContext* dataCtx);
void RunHash();
void CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);
BOOL FilterFiles(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* p);
BOOL FilterFilesInternal(void* ctx, apr_pool_t* p);

BOOL SetMin(const char* value, const char* attr);
BOOL SetMax(const char* value, const char* attr);
BOOL SetLimit(const char* value, const char* attr);
BOOL SetOffset(const char* value, const char* attr);
BOOL SetDictionary(const char* value, const char* attr);
BOOL SetName(const char* value, const char* attr);
BOOL SetHashToSearch(const char* value, const char* attr);


BOOL CompareName(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareSize(BoolOperation* op, void* context, apr_pool_t* p);
BOOL ComparePath(BoolOperation* op, void* context, apr_pool_t* p);

BOOL CompareStr(const char* value, CondOp operation, const char* str, apr_pool_t* p);
BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer);

BOOL Compare(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareLimit(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareOffset(BoolOperation* op, void* context, apr_pool_t* p);

const char* Trim(const char* str);
void* GetContext();


static BOOL (*strOperations[])(const char*, const char*) = {
    SetName,
    NULL,
    SetDictionary,
    NULL,
    SetLimit,
    SetOffset,
    SetMin,
    SetMax,
    SetHashToSearch
};

static BOOL (*comparators[])(BoolOperation*, void*, apr_pool_t*) = {
    CompareName,
    ComparePath,
    NULL,
    CompareSize,
    CompareLimit /* limit */,
    CompareOffset /* offset */,
    NULL,
    NULL,
    Compare /* hash */
};

static int opWeights[] = {
    0, /* == */
    0, /* != */
    1, /* ~ */
    1 /* !~ */,
    0 /* > */,
    0 /* < */,
    0 /* >= */,
    0 /* <= */,
    0 /* or */,
    0 /* and */,
    0 /* not */
};

ProgramOptions* options = NULL;
FILE* output = NULL;

void InitProgram(ProgramOptions* po, const char* fileParam, apr_pool_t* root) {
    options = po;
    fileParameter = fileParam;
    apr_pool_create(&pool, root);
}

void OpenStatement() {
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
    statement = (StatementCtx*)apr_pcalloc(statementPool, sizeof(StatementCtx));
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

void CloseStatement(void) {
    DataContext dataCtx = {0};

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
    StringStatementContext* ctx = GetStringContext();

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

void RunString(DataContext* dataCtx) {
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

void RunDir(DataContext* dataCtx) {
    traverse_ctx_t dirContext = {0};
    DirStatementContext* ctx = GetDirContext();
    BOOL (* filter)(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* pool) = FilterFiles;

    if(NULL == ctx) {
        return;
    }

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
    if(ctx->HashToSearch != NULL) {
        dataCtx->HashToSearch = ctx->HashToSearch;
    }

    if(ctx->FindFiles) {
        dirContext.pfn_file_handler = PrintFileInfo;
    }
    else if(statement->HashAlgorithm == NULL) {
        return;
    }
    else {
        dirContext.pfn_file_handler = CalculateFile;
    }

    dirContext.data_ctx = dataCtx;
    dirContext.is_scan_dir_recursively = ctx->Recursively;

    if(ctx->IncludePattern != NULL || ctx->ExcludePattern != NULL) {
        traverse_compile_pattern(ctx->IncludePattern, &dirContext.include_pattern, statementPool);
        traverse_compile_pattern(ctx->ExcludePattern, &dirContext.exclude_pattern, statementPool);
        filter = traverse_filter_by_name;
    }

    traverse_directory(traverse_hack_root_path(statement->Source,
                                   statementPool), &dirContext, filter, statementPool);
}

void RunFile(DataContext* dataCtx) {
    DirStatementContext* ctx = GetDirContext();

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
    dataCtx->HashToSearch = ctx->HashToSearch;
    dataCtx->IsValidateFileByHash = TRUE;
    if(fileParameter != NULL) {
        apr_file_t* fileHandle = NULL;
        apr_status_t status = APR_SUCCESS;
        apr_finfo_t info = {0};
        FileCtx fileCtx = {0};
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

void SetRecursively() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    GetDirContext()->Recursively = TRUE;
}

void SetFindFiles() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    GetDirContext()->FindFiles = TRUE;
}

void SetBruteForce() {
    if(statementPool == NULL) { // memory allocation error
        return;
    }
    if(statement->Type != CtxTypeHash) {
        return;
    }
    GetStringContext()->BruteForce = TRUE;
    if(GetStringContext()->Min == 0) {
        GetStringContext()->Min = 1;
    }
    if(GetStringContext()->Max == 0) {
        GetStringContext()->Max = MAX_DEFAULT;
    }
    if(GetStringContext()->Dictionary == NULL) {
        GetStringContext()->Dictionary = alphabet;
    }
}

BOOL SetMin(const char* value, const char* attr) {
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if(statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Min = atoi(value);
    return TRUE;
}

BOOL SetMax(const char* value, const char* attr) {
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if(statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Max = atoi(value);
    return TRUE;
}

BOOL SetDictionary(const char* value, const char* attr) {
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if(statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Dictionary = Trim(value);
    return TRUE;
}

BOOL SetName(const char* value, const char* attr) {
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if(statement->Type != CtxTypeDir) {
        return FALSE;
    }
    GetDirContext()->NameFilter = Trim(value);
    return TRUE;
}

BOOL SetHashToSearch(const char* value, const char* attr) {
    DirStatementContext* ctx = NULL;

    if((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    ctx = GetDirContext();
    ctx->HashToSearch = Trim(value);
    SetHashAlgorithmIntoContext(attr);
    return TRUE;
}

BOOL SetLimit(const char* value, const char* attr) {
    apr_status_t status = APR_SUCCESS;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Limit, value, NULL, 0);
    return status == APR_SUCCESS;
}

BOOL SetOffset(const char* value, const char* attr) {
    apr_status_t status = APR_SUCCESS;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Offset, value, NULL, 0);
    return status == APR_SUCCESS;
}

void RegisterIdentifier(const char* identifier) {
    void* ctx = NULL;

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    switch (statement->Type) {
    case CtxTypeDir:
    case CtxTypeFile:
        ctx = apr_pcalloc(statementPool, sizeof(DirStatementContext));
        ((DirStatementContext*)ctx)->Limit = MAXLONG64;
        break;
    case CtxTypeString:
    case CtxTypeHash:
        ctx = apr_pcalloc(statementPool, sizeof(StringStatementContext));
        ((StringStatementContext*)ctx)->BruteForce = FALSE;
        break;
    }
    statement->Id = (const char*)identifier;
    apr_hash_set(ht, statement->Id, APR_HASH_KEY_STRING, ctx);
}

void DefineQueryType(CtxType type) {
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

DirStatementContext* GetDirContext() {
    return (DirStatementContext*)GetContext();
}

StringStatementContext* GetStringContext() {
    return (StringStatementContext*)GetContext();
}

void SetSource(const char* str, void* token) {
    statement->Source = Trim(str);
}

void SetHashAlgorithmIntoContext(const char* str) {
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
    return GetDirContext()->Operation == CondOpEq ? !result : result;
}

int CompareHashAttempt(void* hash, const void* pass, const uint32_t length) {
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    statement->HashAlgorithm->pfn_digest_(attempt, pass, (apr_size_t)length);
    return CompareDigests(attempt, hash);
}

void ToDigest(const char* hash, apr_byte_t* digest) {
    lib_hex_str_2_byte_array(hash, digest, hashLength);
}

void* CreateDigest(const char* hash, apr_pool_t* p) {
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

int CompareHash(apr_byte_t* digest, const char* checkSum) {
    apr_byte_t* bytes = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * GetDigestSize());

    ToDigest(checkSum, bytes);
    return CompareDigests(bytes, digest);
}

BOOL FilterFiles(apr_finfo_t* info, const char* dir, traverse_ctx_t* ctx, apr_pool_t* p) {
    FileCtx fileCtx = {0};
    fileCtx.Dir = dir;
    fileCtx.Info = info;
    fileCtx.PfnOutput = ((DataContext*)ctx->data_ctx)->PfnOutput;
    return FilterFilesInternal(&fileCtx, p);
}

BOOL MatchStr(const char* value, CondOp operation, const char* str, apr_pool_t* p) {
    BOOL result = bend_match_re(value, str);

    switch (operation) {
    case CondOpMatch:
        return result;
    case CondOpNotMatch:
        return !result;
    }

    return FALSE;
}

BOOL CompareStr(const char* value, CondOp operation, const char* str, apr_pool_t* p) {
    switch(operation) {
        case CondOpMatch:
        case CondOpNotMatch:
            return MatchStr(value, operation, str, p);
        case CondOpEq:
            return strcmp(value, str) == 0;
        case CondOpNotEq:
            return strcmp(value, str) != 0;
    }

    return FALSE;
}

BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer) {
    apr_off_t size = 0;
    apr_strtoff(&size, integer, NULL, 0);

    switch(operation) {
        case CondOpGe:
            return value > size;
        case CondOpLe:
            return value < size;
        case CondOpEq:
            return value == size;
        case CondOpNotEq:
            return value != size;
        case CondOpGeEq:
            return value >= size;
        case CondOpLeEq:
            return value <= size;
    }

    return FALSE;
}

BOOL CompareName(BoolOperation* op, void* context, apr_pool_t* p) {
    FileCtx* ctx = (FileCtx*)context;
    return CompareStr(op->Value, op->Operation, ctx->Info->name, p);
}

BOOL ComparePath(BoolOperation* op, void* context, apr_pool_t* p) {
    FileCtx* ctx = (FileCtx*)context;
    char* fullPath = NULL; // Full path to file or subdirectory

    apr_filepath_merge(&fullPath,
                       ctx->Dir,
                       ctx->Info->name,
                       APR_FILEPATH_NATIVE,
                       p); // IMPORTANT: so as not to use strdup

    return CompareStr(op->Value, op->Operation, fullPath, p);
}

BOOL CompareSize(BoolOperation* op, void* context, apr_pool_t* p) {
    FileCtx* ctx = (FileCtx*)context;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
#endif
    return CompareInt(ctx->Info->size, op->Operation, op->Value);
}

void PrintFileInfo(const char* fullPathToFile, DataContext* ctx, apr_pool_t* p) {
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

BOOL Compare(BoolOperation* op, void* context, apr_pool_t* p) {
    apr_status_t status = APR_SUCCESS;
    apr_file_t* fileHandle = NULL;
    FileCtx* ctx = (FileCtx*)context;
    apr_byte_t* digestToCompare = NULL;
    apr_byte_t* digest = NULL;
    char* cacheKey = NULL;
    const char* cachedDigest = NULL;

    char* fullPath = NULL; // Full path to file or subdirectory
    BOOL result = FALSE;

    SetHashAlgorithmIntoContext(op->AttributeName);

    digest = (apr_byte_t*)apr_pcalloc(p, sizeof(apr_byte_t) * hashLength);
    digestToCompare = (apr_byte_t*)apr_pcalloc(p, sizeof(apr_byte_t) * hashLength);

    ToDigest(op->Value, digestToCompare);

    CalculateDigest(digest, NULL, 0);
    if(CompareDigests(digest, digestToCompare) && (ctx->Info->size == 0)) { // Empty file optimization
        result = TRUE;
        goto ret;
    }

    apr_filepath_merge(&fullPath,
                       ctx->Dir,
                       ctx->Info->name,
                       APR_FILEPATH_NATIVE,
                       p); // IMPORTANT: so as not to use strdup

    if(htFileDigestCache != NULL) {
        cacheKey = apr_psprintf(p, "%s_%ld_%ld_%ld", op->AttributeName, GetDirContext()->Offset, GetDirContext()->Limit, ctx->Info->size);
        cachedDigest = apr_hash_get(htFileDigestCache, (const char*)cacheKey, APR_HASH_KEY_STRING);
    }

    if(cachedDigest != NULL) {
        ToDigest(cachedDigest, digest);
    }
    else {
        status = apr_file_open(&fileHandle, fullPath, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
        if(status != APR_SUCCESS) {
            result = FALSE;
            goto ret;
        }

        CalculateHash(fileHandle, ctx->Info->size, digest, GetDirContext()->Limit, GetDirContext()->Offset, p);
        apr_file_close(fileHandle);

        if(htFileDigestCache != NULL) {
            apr_hash_set(htFileDigestCache, cacheKey, APR_HASH_KEY_STRING, out_hash_to_string(digest, FALSE, hashLength, p));
        }
    }
    result = CompareDigests(digest, digestToCompare);
ret:
    return op->Operation == CondOpEq ? result : !result;
}


BOOL CompareLimit(BoolOperation* op, void* context, apr_pool_t* p) {
    apr_off_t limit = 0;
    apr_status_t status = apr_strtoff(&limit, op->Value, NULL, 0);
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
    UNREFERENCED_PARAMETER(context);
#endif
    GetDirContext()->Limit = limit;
    return status == APR_SUCCESS;
}

BOOL CompareOffset(BoolOperation* op, void* context, apr_pool_t* p) {
    apr_off_t offset = 0;
    FileCtx* ctx = (FileCtx*)context;
    apr_status_t status = apr_strtoff(&offset, op->Value, NULL, 0);

#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
#endif

    if(ctx->Info->size < offset) {
        return FALSE;
    }
    GetDirContext()->Offset = offset;
    return status == APR_SUCCESS;
}
