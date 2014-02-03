/*!
 * \brief   The file contains HLINQ compiler API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include <math.h>
#include <io.h>
#include "compiler.h"
#include "apr_hash.h"
#include "sph_md2.h"
#include "sph_md5.h"
#include "sph_md4.h"
#include "crc32.h"
#include "sph_sha1.h"
#include "sph_whirlpool.h"
#include "sph_ripemd.h"
#include "sph_sha2.h"
#include "sph_tiger.h"
#include "gost.h"
#include "pcre.h"
#include "..\srclib\encoding.h"
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
apr_hash_t* htVars = NULL;
apr_array_header_t* whereStack;
const char* fileParameter = NULL;
pANTLR3_RECOGNIZER_SHARED_STATE parserState = NULL;

StatementCtx* statement = NULL;

apr_size_t hashLength = 0;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

// Forward declarations
void*        FileAlloc(size_t size);
apr_status_t FindFile(const char* fullPathToFile, DataContext* ctx, apr_pool_t* p);
void         RunString(DataContext* dataCtx);
void         RunDir(DataContext* dataCtx);
void         RunFile(DataContext* dataCtx);
void         RunHash();
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);
BOOL         FilterFiles(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* p);

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

const char* Trim(pANTLR3_UINT8 str);
void*       GetContext();


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

static BOOL (*comparators[])(BoolOperation *, void*, apr_pool_t*) = {
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

void InitProgram(ProgramOptions* po, const char* fileParam, apr_pool_t* root)
{
    options = po;
    fileParameter = fileParam;
    apr_pool_create(&pool, root);
    htVars = apr_hash_make(pool);
}

void OpenStatement(pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    apr_status_t status = apr_pool_create(&statementPool, pool);
    parserState = state;

    if (status != APR_SUCCESS) {
        statementPool = NULL;
        return;
    }

    ht = apr_hash_make(statementPool);

    if (ht == NULL) {
destroyPool:
        apr_pool_destroy(statementPool);
        statementPool = NULL;
        return;
    }

    whereStack = apr_array_make(statementPool, ARRAY_INIT_SZ, sizeof(BoolOperation*));
    if (whereStack == NULL) {
        goto destroyPool;
    }
    statement = (StatementCtx*)apr_pcalloc(statementPool, sizeof(StatementCtx));
    if (statement == NULL) {
        goto destroyPool;
    }
    statement->Type = CtxTypeUndefined;
    statement->HashAlgorithm = NULL;
}

void OutputBothFileAndConsole(OutputContext* ctx)
{
    OutputToConsole(ctx);

    CrtFprintf(output, "%s", ctx->StringToPrint); //-V111
    if (ctx->IsPrintSeparator) {
        CrtFprintf(output, FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->IsFinishLine) {
        CrtFprintf(output, NEW_LINE);
    }
}

void CloseStatement(void)
{
    DataContext dataCtx = { 0 };

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    if (options->OnlyValidate || ((parserState != NULL) && (parserState->errorCount > 0))) {
        goto cleanup;
    }

#ifdef GTEST
    dataCtx.PfnOutput = OutputToCppConsole;
#else
    if (options->FileToSave != NULL) {
        #ifdef __STDC_WANT_SECURE_LIB__
        fopen_s(&output, options->FileToSave, "w+");
        #else
        output = fopen(options->FileToSave, "w+");
        #endif
        if (output == NULL) {
            CrtPrintf("\nError opening file: %s Error message: ", options->FileToSave);
            perror("");
            goto cleanup;
        }
        dataCtx.PfnOutput = OutputBothFileAndConsole;
    } else {
        dataCtx.PfnOutput = OutputToConsole;
    }

#endif
    dataCtx.IsPrintCalcTime = options->PrintCalcTime;
    dataCtx.IsPrintLowCase = options->PrintLowCase;
    dataCtx.IsPrintSfv = options->PrintSfv;

    pcre_malloc = FileAlloc;

    switch (statement->Type) {
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
        default:
            goto cleanup;
            break;
    }

cleanup:
    if (output != NULL) {
        fclose(output);
        output = NULL;
    }
    apr_pool_destroy(statementPool);
    statementPool = NULL;
    ht = NULL;
    whereStack = NULL;
    statement = NULL;
}

void RunHash()
{
    StringStatementContext* ctx = GetStringContext();

    if ((NULL == ctx) || (statement->HashAlgorithm == NULL) || !(ctx->BruteForce)) {
        return;
    }

    hashLength = statement->HashAlgorithm->HashLength;

    CrackHash(ctx->Dictionary,
              statement->Source,
              ctx->Min,
              ctx->Max,
              hashLength,
              statement->HashAlgorithm->PfnDigest,
              options->NoProbe,
              options->NumOfThreads,
              statement->HashAlgorithm->UseWideString,
              statementPool);
}

void RunString(DataContext* dataCtx)
{
    apr_byte_t* digest = NULL;
    apr_size_t sz = 0;

    if (statement->HashAlgorithm == NULL) {
        return;
    }
    sz = statement->HashAlgorithm->HashLength;
    digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * sz);

    if (statement->HashAlgorithm->UseWideString) {
        wchar_t* str = FromAnsiToUnicode(statement->Source, statementPool);
        statement->HashAlgorithm->PfnDigest(digest, str, wcslen(str) * sizeof(wchar_t));
    } else {
        statement->HashAlgorithm->PfnDigest(digest, statement->Source, strlen(statement->Source));
    }

    OutputDigest(digest, dataCtx, sz, statementPool);
}

void RunDir(DataContext* dataCtx)
{
    TraverseContext dirContext = { 0 };
    DirStatementContext* ctx = GetDirContext();
    BOOL (* filter)(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool) = FilterFiles;

    if (NULL == ctx) {
        return;
    }

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
    if (ctx->HashToSearch != NULL) {
        dataCtx->HashToSearch = ctx->HashToSearch;
    }

    if (ctx->FindFiles) {
        dirContext.PfnFileHandler = FindFile;
    } else if (statement->HashAlgorithm == NULL) {
        return;
    } else {
        dirContext.PfnFileHandler = CalculateFile;
    }

    dirContext.DataCtx = dataCtx;
    dirContext.IsScanDirRecursively = ctx->Recursively;

    if ((ctx->IncludePattern != NULL) || (ctx->ExcludePattern != NULL)) {
        CompilePattern(ctx->IncludePattern, &dirContext.IncludePattern, statementPool);
        CompilePattern(ctx->ExcludePattern, &dirContext.ExcludePattern, statementPool);
        filter = FilterByName;
    }

    TraverseDirectory(HackRootPath(statement->Source,
                                   statementPool), &dirContext, filter, statementPool);
}

void RunFile(DataContext* dataCtx)
{
    DirStatementContext* ctx = GetDirContext();

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
    if (fileParameter != NULL) {
        apr_file_t* fileHandle = NULL;
        apr_status_t status = APR_SUCCESS;
        apr_finfo_t info = { 0 };
        FileCtx fileCtx = { 0 };
        char* dir = NULL;
        OutputContext output = { 0 };
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
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
            return;
        }
        status = apr_file_info_get(
            &info,
            APR_FINFO_NAME | APR_FINFO_SIZE | APR_FINFO_IDENT |
            APR_FINFO_TYPE,
            fileHandle);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
            goto cleanup;
        }
        status = apr_filepath_root(&dir, &fileParameter, APR_FILEPATH_NATIVE, statementPool);

        if (status == APR_ERELATIVE) {
            status = apr_filepath_get(&dir, APR_FILEPATH_NATIVE, statementPool);
            if (status != APR_SUCCESS) {
                OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
                goto cleanup;
            }
        } else if (status != APR_SUCCESS) {
            OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
            goto cleanup;
        }

        fileCtx.Dir = dir;
        info.name = info.fname;
        fileCtx.Info = &info;
        fileCtx.PfnOutput = dataCtx->PfnOutput;

        fileAnsi = FromUtf8ToAnsi(statement->Source, statementPool);
        output.StringToPrint = fileAnsi == NULL ? statement->Source : fileAnsi;
        output.IsPrintSeparator = TRUE;
        dataCtx->PfnOutput(&output);

        output.IsPrintSeparator = TRUE;
        output.IsFinishLine = FALSE;
        output.StringToPrint = CopySizeToString(info.size, statementPool);
        dataCtx->PfnOutput(&output);

        output.StringToPrint = "File is ";
        output.IsPrintSeparator = FALSE;
        dataCtx->PfnOutput(&output);

        if (FilterFilesInternal(&fileCtx, statementPool)) {
            output.StringToPrint = "valid";
        } else {
            output.StringToPrint = "invalid";
        }
        dataCtx->PfnOutput(&output);
cleanup:
        status = apr_file_close(fileHandle);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
        }
        return;
    }
    if (statement->HashAlgorithm == NULL) {
        return;
    }
    if (ctx->HashToSearch) {
        apr_byte_t* digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * GetDigestSize());
        
        CalculateFileHash(statement->Source, digest, dataCtx->IsPrintCalcTime, options->PrintSfv, NULL, dataCtx->Limit,
                          dataCtx->Offset, dataCtx->PfnOutput, statementPool);
        CheckHash(digest, ctx->HashToSearch, dataCtx);
    } else {
        CalculateFile(statement->Source, dataCtx, statementPool);
    }
}

void SetRecursively()
{
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    GetDirContext()->Recursively = TRUE;
}

void SetFindFiles()
{
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    GetDirContext()->FindFiles = TRUE;
}

void SetBruteForce()
{
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    if (statement->Type != CtxTypeHash) {
        return;
    }
    GetStringContext()->BruteForce = TRUE;
    if (GetStringContext()->Min == 0) {
        GetStringContext()->Min = 1;
    }
    if (GetStringContext()->Max == 0) {
        GetStringContext()->Max = MAX_DEFAULT;
    }
    if (GetStringContext()->Dictionary == NULL) {
        GetStringContext()->Dictionary = alphabet;
    }
}

BOOL SetMin(const char* value, const char* attr)
{
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Min = atoi(value);
    return TRUE;
}

BOOL SetMax(const char* value, const char* attr)
{
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Max = atoi(value);
    return TRUE;
}

BOOL SetDictionary(const char* value, const char* attr)
{
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Dictionary = Trim(value);
    return TRUE;
}

BOOL SetName(const char* value, const char* attr)
{
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if (statement->Type != CtxTypeDir) {
        return FALSE;
    }
    GetDirContext()->NameFilter = Trim(value);
    return TRUE;
}

BOOL SetHashToSearch(const char* value, const char* attr)
{
    DirStatementContext* ctx = NULL;

    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    ctx = GetDirContext();
    ctx->HashToSearch = Trim(value);
    SetHashAlgorithmIntoContext(attr);
    return TRUE;
}

BOOL SetLimit(const char* value, const char* attr)
{
    apr_status_t status = APR_SUCCESS;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Limit, value, NULL, 0);
    return status == APR_SUCCESS;
}

BOOL SetOffset(const char* value, const char* attr)
{
    apr_status_t status = APR_SUCCESS;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(attr);
#endif
    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Offset, value, NULL, 0);
    return status == APR_SUCCESS;
}

void AssignAttribute(Attr code, pANTLR3_UINT8 value, void* valueToken, pANTLR3_UINT8 attrubute)
{
    BOOL (* op)(const char*, const char*) = NULL;

    if (code == AttrUndefined) {
        return;
    }
    op = strOperations[code];
    if (!op) {
        return;
    }
    if (!op((const char*)value, (const char*)attrubute)) {
        parserState->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION,
                                                    "invalid value",
                                                    "error: value is invalid",
                                                    ANTLR3_FALSE);
        parserState->exception->token = valueToken;
        parserState->error = ANTLR3_RECOGNITION_EXCEPTION;
    }
}

void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode, void* token, pANTLR3_UINT8 attrubute)
{
    BoolOperation* op = NULL;
    int weight = 0;

    if (statementPool == NULL) { // memory allocation error
        return;
    }
    switch (code) {
        case AttrName:
            weight = 1;
            break;
        case AttrPath:
            weight = 1;
            break;
        case AttrDict:
            weight = 1;
            break;
        case AttrHash:
            weight = GetHash((const char*)attrubute)->Weight;
            break;
        default:
            break;
    }

    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));
    op->Attribute = code;
    op->AttributeName = (const char*)attrubute;
    op->Operation = opcode;
    op->Value =  Trim(value);
    op->Token = token;
    op->Weight = weight;
    *(BoolOperation**)apr_array_push(whereStack) = op;
}

void WhereClauseCond(CondOp opcode, void* token)
{
    WhereClauseCall(AttrUndefined, NULL, opcode, token, NULL);
}

void DefineQueryType(CtxType type)
{
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    statement->Type = type;
}

void  RegisterVariable(pANTLR3_UINT8 var, pANTLR3_UINT8 value)
{
    apr_hash_set(htVars, (const char*)var, APR_HASH_KEY_STRING, value);
}

void RegisterIdentifier(pANTLR3_UINT8 identifier)
{
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

BOOL CallAttiribute(pANTLR3_UINT8 identifier, void* token)
{
    if (statementPool == NULL) { // memory allocation error
        return FALSE;
    }

    if (apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING) != NULL) {
        return TRUE;
    }
    parserState->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION,
                                                UNKNOWN_IDENTIFIER,
                                                "error: " UNKNOWN_IDENTIFIER,
                                                ANTLR3_FALSE);
    parserState->exception->token = token;
    parserState->error = ANTLR3_RECOGNITION_EXCEPTION;
    return FALSE;
}

void* GetContext()
{
    if (NULL == statement->Id) {
        return NULL;
    }
    return apr_hash_get(ht, statement->Id, APR_HASH_KEY_STRING);
}

DirStatementContext* GetDirContext()
{
    return (DirStatementContext*)GetContext();
}

StringStatementContext* GetStringContext()
{
    return (StringStatementContext*)GetContext();
}

const char* GetValue(pANTLR3_UINT8 variable, void* token)
{
    const char* result = apr_hash_get(htVars, (const char*)variable, APR_HASH_KEY_STRING);
    if (result == NULL) {
        parserState->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION,
                                                    UNKNOWN_IDENTIFIER,
                                                    "error: " UNKNOWN_IDENTIFIER,
                                                    ANTLR3_FALSE);
        parserState->exception->token = token;
        parserState->error = ANTLR3_RECOGNITION_EXCEPTION;
        return NULL;
    }
    return Trim(result);
}

void SetSource(pANTLR3_UINT8 str, void* token)
{
    char* tmp = Trim(str);

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    if (NULL == tmp) {
        return;
    }
    if (token == NULL) {
        statement->Source = tmp;
        return;
    }
    statement->Source = GetValue(str, token);
}

void SetHashAlgorithmIntoContext(pANTLR3_UINT8 str)
{
    HashDefinition* algorithm = NULL;
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    algorithm = GetHash((const char*)str);
    if (algorithm == NULL) {
        return;
    }

    statement->HashAlgorithm = algorithm;
    hashLength = algorithm->HashLength;
}

BOOL IsStringBorder(pANTLR3_UINT8 str, size_t ix)
{
    return str[ix] == '\'' || str[ix] == '\"';
}

const char* Trim(pANTLR3_UINT8 str)
{
    size_t len = 0;
    char* tmp = NULL;

    if (!str) {
        return NULL;
    }
    tmp = apr_pstrdup(statementPool, (char*)str);

    if (IsStringBorder(str, 0)) {
        tmp = tmp + 1; // leading " or '
    }
    len = strlen(tmp);
    if ((len > 0) && IsStringBorder((pANTLR3_UINT8)tmp, len - 1)) {
        tmp[len - 1] = '\0'; // trailing " or '
    }
    return tmp;
}

/*!
 * It's so ugly to improve performance
 */
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2)
{
    return memcmp(digest1, digest2, hashLength) == 0;
}

int ComparisonFailure(int result)
{
    return GetDirContext()->Operation == CondOpEq ? !result : result;
}

int CompareHashAttempt(void* hash, const void* pass, const uint32_t length)
{
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    statement->HashAlgorithm->PfnDigest(attempt, pass, (apr_size_t)length);
    return CompareDigests(attempt, hash);
}

void ToDigest(const char* hash, apr_byte_t* digest)
{
    HexStrintToByteArray(hash, digest, hashLength);
}

void* CreateDigest(const char* hash, apr_pool_t* p)
{
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(p, hashLength);
    ToDigest(hash, result);
    return result;
}

void CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    statement->HashAlgorithm->PfnDigest(digest, input, inputLen);
}

void InitContext(void* context)
{
    statement->HashAlgorithm->PfnInit(context);
}

void FinalHash(apr_byte_t* digest, void* context)
{
    statement->HashAlgorithm->PfnFinal(digest, context);
}

void UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    statement->HashAlgorithm->PfnUpdate(context, input, inputLen);
}

void* AllocateContext(apr_pool_t* p)
{
    return apr_pcalloc(p, statement->HashAlgorithm->ContextSize);
}

apr_size_t GetDigestSize()
{
    return statement->HashAlgorithm->HashLength;
}

int CompareHash(apr_byte_t* digest, const char* checkSum)
{
    apr_byte_t* bytes = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * GetDigestSize());

    ToDigest(checkSum, bytes);
    return CompareDigests(bytes, digest);
}

BOOL Skip(CondOp op)
{
    switch (op) {
        case CondOpAnd:
        case CondOpOr:
        case CondOpNot:
        case CondOpUndefined:
            return TRUE;
    }
    return FALSE;
}

BOOL FilterFilesInternal(void* ctx, apr_pool_t* p)
{
    int i;
    apr_array_header_t* stack = NULL;

    if (apr_is_empty_array(whereStack)) {
        return TRUE;
    }

    stack = apr_array_make(p, ARRAY_INIT_SZ, sizeof(BOOL));
    // optimization conditions
    for (i = 0; i < whereStack->nelts - 1; i++) {
        BoolOperation* op1 = ((BoolOperation**)whereStack->elts)[i];
        BoolOperation* op2 = NULL;

        if (i + 1 >= whereStack->nelts) {
            break;
        }
        op2 = ((BoolOperation**)whereStack->elts)[i + 1];
        if (Skip(op1->Operation) || Skip(op2->Operation)) {
            continue;
        }
        if (op1->Weight < op2->Weight) {
            continue;
        } else if ((op1->Weight == op2->Weight) && (opWeights[op1->Operation] <= opWeights[op2->Operation])) {
            continue;
        }
        ((BoolOperation**)whereStack->elts)[i] = op2;
        ((BoolOperation**)whereStack->elts)[i + 1] = op1;
    }

    for (i = 0; i < whereStack->nelts; i++) {
        BOOL left;
        BOOL right;
        BoolOperation* op = ((BoolOperation**)whereStack->elts)[i];

        switch (op->Operation) {
            case CondOpAnd:
            case CondOpOr:
            {
                left = *((BOOL*)apr_array_pop(stack));
                right = *((BOOL*)apr_array_pop(stack));

                if (op->Operation == CondOpAnd) {
                    *(BOOL*)apr_array_push(stack) = left && right;
                } else {
                    *(BOOL*)apr_array_push(stack) = left || right;
                }
                break;
            }
            case CondOpNot:
                left = *((BOOL*)apr_array_pop(stack));
                *(BOOL*)apr_array_push(stack) = !left;
                break;
            default:
            {
                BOOL (* comparator)(BoolOperation*, void*,
                                    apr_pool_t*) = comparators[op->Attribute];

                if (comparator == NULL) {
                    *(BOOL*)apr_array_push(stack) = TRUE;
                } else {
                    // optimization
                    if (i + 1 < whereStack->nelts) {
                        BoolOperation* ahead = ((BoolOperation**)whereStack->elts)[i + 1];
                        if ((ahead->Operation == CondOpAnd) || (ahead->Operation == CondOpOr) ) {
                            left = *((BOOL*)apr_array_pop(stack));

                            if ((ahead->Operation == CondOpAnd) && !left ||
                                (ahead->Operation == CondOpOr) && left) {
                                *(BOOL*)apr_array_push(stack) = left;
                                *(BOOL*)apr_array_push(stack) = FALSE;
                            } else {
                                *(BOOL*)apr_array_push(stack) = left;
                                goto run;
                            }
                        } else {
                            goto run;
                        }
                    } else {
run:
                        *(BOOL*)apr_array_push(stack) = comparator(op, ctx, p);
                    }
                }
                break;
            }
        }
    }
    return *((BOOL*)apr_array_pop(stack));
}

BOOL FilterFiles(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* p)
{
    FileCtx fileCtx = { 0 };
    fileCtx.Dir = dir;
    fileCtx.Info = info;
    fileCtx.PfnOutput = ((DataContext*)ctx->DataCtx)->PfnOutput;
    return FilterFilesInternal(&fileCtx, p);
}

void* FileAlloc(size_t size)
{
    return apr_palloc(filePool, size);
}

BOOL MatchStr(const char* value, CondOp operation, const char* str, apr_pool_t* p)
{
    pcre* re = NULL;
    const char* error = NULL;
    int erroffset = 0;
    int rc = 0;
    int flags  = PCRE_NOTEMPTY;

    filePool = p; // needed for pcre_alloc (FileAlloc) function

    re = pcre_compile(value,           /* the pattern */
                      PCRE_UTF8,
                      &error,          /* for error message */
                      &erroffset,      /* for error offset */
                      0);              /* use default character tables */

    if (!re) {
        return FALSE;
    }

    if (!strstr(value, "^")) {
        flags |= PCRE_NOTBOL;
    }
    if (!strstr(value, "$")) {
        flags |= PCRE_NOTEOL;
    }

    rc = pcre_exec(
        re,                   /* the compiled pattern */
        0,                    /* no extra data - pattern was not studied */
        str,                  /* the string to match */
        (int)strlen(str),     /* the length of the string */
        0,                    /* start at offset 0 in the subject */
        flags,
        NULL,              /* output vector for substring information */
        0);           /* number of elements in the output vector */

    switch (operation) {
        case CondOpMatch:
            return rc >= 0;
        case CondOpNotMatch:
            return rc < 0;
    }

    return FALSE;
}

BOOL CompareStr(const char* value, CondOp operation, const char* str, apr_pool_t* p)
{
    switch (operation) {
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

BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer)
{
    apr_off_t size = 0;
    apr_strtoff(&size, integer, NULL, 0);

    switch (operation) {
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

BOOL CompareName(BoolOperation* op, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
    return CompareStr(op->Value, op->Operation, ctx->Info->name, p);
}

BOOL ComparePath(BoolOperation* op, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
    char* fullPath = NULL; // Full path to file or subdirectory

    apr_filepath_merge(&fullPath,
                       ctx->Dir,
                       ctx->Info->name,
                       APR_FILEPATH_NATIVE,
                       p);  // IMPORTANT: so as not to use strdup

    return CompareStr(op->Value, op->Operation, fullPath, p);
}

BOOL CompareSize(BoolOperation* op, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
#endif
    return CompareInt(ctx->Info->size, op->Operation, op->Value);
}

apr_status_t FindFile(const char* fullPathToFile, DataContext* ctx, apr_pool_t* p)
{
    OutputContext output = { 0 };
    char* fileAnsi = NULL;
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = { 0 };

    fileAnsi = FromUtf8ToAnsi(fullPathToFile, p);

    apr_file_open(&fileHandle, fullPathToFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, fileHandle);

    output.IsFinishLine = FALSE;
    output.IsPrintSeparator = TRUE;

    // file name
    output.StringToPrint = fileAnsi == NULL ? fullPathToFile : fileAnsi;
    ctx->PfnOutput(&output);

    // file size
    output.StringToPrint = CopySizeToString(info.size, p);

    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    ctx->PfnOutput(&output); // file size or time output
    apr_file_close(fileHandle);
    return APR_SUCCESS;
}

BOOL Compare(BoolOperation* op, void* context, apr_pool_t* p)
{
    apr_status_t status = APR_SUCCESS;
    apr_file_t* fileHandle = NULL;
    FileCtx* ctx = (FileCtx*)context; 
    apr_byte_t* digestToCompare = NULL;
    apr_byte_t* digest = NULL;
    
    char* fullPath = NULL; // Full path to file or subdirectory
    BOOL result = FALSE;

    SetHashAlgorithmIntoContext(op->AttributeName);
    
    digest = (apr_byte_t*)apr_pcalloc(p, sizeof(apr_byte_t) * hashLength);
    digestToCompare = (apr_byte_t*)apr_pcalloc(p, sizeof(apr_byte_t) * hashLength);
    
    ToDigest(op->Value, digestToCompare);

    CalculateDigest(digest, NULL, 0);
    if (CompareDigests(digest, digestToCompare) && (ctx->Info->size == 0)) { // Empty file optimization
        result = TRUE;
        goto ret;
    }

    apr_filepath_merge(&fullPath,
                       ctx->Dir,
                       ctx->Info->name,
                       APR_FILEPATH_NATIVE,
                       p);  // IMPORTANT: so as not to use strdup

    status = apr_file_open(&fileHandle, fullPath, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    if (status != APR_SUCCESS) {
        result = FALSE;
        goto ret;
    }

    CalculateHash(fileHandle, ctx->Info->size, digest, GetDirContext()->Limit,
                  GetDirContext()->Offset, ctx->PfnOutput, p);

    result = CompareDigests(digest, digestToCompare);
    apr_file_close(fileHandle);
ret:
    return op->Operation == CondOpEq ? result : !result;
}


BOOL CompareLimit(BoolOperation* op, void* context, apr_pool_t* p)
{
    apr_off_t limit = 0;
    apr_status_t status = apr_strtoff(&limit, op->Value, NULL, 0);
#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
    UNREFERENCED_PARAMETER(context);
#endif
    GetDirContext()->Limit = limit;
    return status == APR_SUCCESS;
}

BOOL CompareOffset(BoolOperation* op, void* context, apr_pool_t* p)
{
    apr_off_t offset = 0;
    FileCtx* ctx = (FileCtx*)context;
    apr_status_t status = apr_strtoff(&offset, op->Value, NULL, 0);

#ifdef _MSC_VER
    UNREFERENCED_PARAMETER(p);
#endif

    if (ctx->Info->size < offset) {
        return FALSE;
    }
    GetDirContext()->Offset = offset;
    return status == APR_SUCCESS;
}
