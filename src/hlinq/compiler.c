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
#include "compiler.h"
#include "md4.h"
#include "md5.h"
#include "sha1.h"
#include "sha256def.h"
#include "sha384def.h"
#include "sha512def.h"
#include "whirl.h"
#include "crc32def.h"
#include "libtom.h"
#include "pcre.h"
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
BOOL dontRunActions = FALSE;
const char* fileParameter = NULL;
pANTLR3_RECOGNIZER_SHARED_STATE parserState = NULL;

StatementCtx* statement = NULL;

apr_status_t (* digestFunction)(apr_byte_t* digest, const void* input,
                                const apr_size_t inputLen) = NULL;

apr_size_t hashLength = 0;
static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

/*
Hash sizes:

WHIRLPOOL     64
SHA-512       64
SHA-384       48
RIPEMD-320    40
SHA-256       32
RIPEMD-256    32
SHA-224       28
TIGER-192     24
SHA-1         20
RIPEMD-160    20
RIPEMD-128    16
MD5           16
MD4           16
MD2           16

*/

#define SZ_WHIRLPOOL    64
#define SZ_SHA512       64
#define SZ_SHA384       48
#define SZ_RIPEMD320    40
#define SZ_SHA256       32
#define SZ_RIPEMD256    32
#define SZ_SHA224       28
#define SZ_TIGER192     24
#define SZ_SHA1         20
#define SZ_RIPEMD160    20
#define SZ_RIPEMD128    16
#define SZ_MD5          16
#define SZ_MD4          16
#define SZ_MD2          16


static apr_size_t hashLengths[] = {
    SZ_MD5,
    SZ_SHA1,
    SZ_MD4,
    SZ_SHA256,
    SZ_SHA384,
    SZ_SHA512,
    SZ_WHIRLPOOL,
    CRC32_HASH_SIZE,
    SZ_MD2,
    SZ_TIGER192,
    SZ_RIPEMD128,
    SZ_RIPEMD160,
    SZ_RIPEMD256,
    SZ_RIPEMD320,
    SZ_SHA224
};

static apr_status_t (*digestFunctions[])(apr_byte_t * digest, const void* input,
                                         const apr_size_t inputLen) = {
    MD5CalculateDigest,
    SHA1CalculateDigest,
    MD4CalculateDigest,
    SHA256CalculateDigest,
    SHA384CalculateDigest,
    SHA512CalculateDigest,
    WHIRLPOOLCalculateDigest,
    CRC32CalculateDigest,
    MD2CalculateDigest,
    TIGERCalculateDigest,
    RMD128CalculateDigest,
    RMD160CalculateDigest,
    RMD256CalculateDigest,
    RMD320CalculateDigest,
    SHA224CalculateDigest
};

static apr_status_t (*initCtxFuncs[])(void* context) = {
    MD5InitContext,
    SHA1InitContext,
    MD4InitContext,
    SHA256InitContext,
    SHA384InitContext,
    SHA512InitContext,
    WHIRLPOOLInitContext,
    CRC32InitContext,
    MD2InitContext,
    TIGERInitContext,
    RMD128InitContext,
    RMD160InitContext,
    RMD256InitContext,
    RMD320InitContext,
    SHA224InitContext
};

static apr_status_t (*finalHashFuncs[])(apr_byte_t * digest, void* context) = {
    MD5FinalHash,
    SHA1FinalHash,
    MD4FinalHash,
    SHA256FinalHash,
    SHA384FinalHash,
    SHA512FinalHash,
    WHIRLPOOLFinalHash,
    CRC32FinalHash,
    MD2FinalHash,
    TIGERFinalHash,
    RMD128FinalHash,
    RMD160FinalHash,
    RMD256FinalHash,
    RMD320FinalHash,
    SHA224FinalHash
};

static apr_status_t (*updateHashFuncs[])(void* context, const void* input,
                                         const apr_size_t inputLen) = {
    MD5UpdateHash,
    SHA1UpdateHash,
    MD4UpdateHash,
    SHA256UpdateHash,
    SHA384UpdateHash,
    SHA512UpdateHash,
    WHIRLPOOLUpdateHash,
    CRC32UpdateHash,
    MD2UpdateHash,
    TIGERUpdateHash,
    RMD128UpdateHash,
    RMD160UpdateHash,
    RMD256UpdateHash,
    RMD320UpdateHash,
    SHA224UpdateHash
};

static size_t contextSizes[] = {
    sizeof(apr_md5_ctx_t),
    sizeof(apr_sha1_ctx_t),
    sizeof(apr_md4_ctx_t),
    sizeof(SHA256Context),
    sizeof(SHA384Context),
    sizeof(SHA512Context),
    sizeof(WHIRLPOOL_CTX),
    sizeof(Crc32Context),
    sizeof(hash_state),
    sizeof(hash_state),
    sizeof(hash_state),
    sizeof(hash_state),
    sizeof(hash_state),
    sizeof(hash_state),
    sizeof(hash_state)
};

static BOOL (*strOperations[])(const char*) = {
    SetName,
    NULL,
    SetDictionary,
    SetMd5ToSearch,
    SetSha1ToSearch,
    SetSha256ToSearch,
    SetSha384ToSearch,
    SetSha512ToSearch,
    SetShaMd4ToSearch,
    SetShaCrc32ToSearch,
    SetShaWhirlpoolToSearch,
    NULL,
    SetLimit,
    SetOffset,
    SetMin,
    SetMax,
    SetMd2ToSearch
};

static BOOL (*comparators[])(BoolOperation *, void*, apr_pool_t*) = {
    CompareName,
    ComparePath,
    NULL,
    CompareMd5 /* md5 */,
    CompareSha1 /* sha1 */,
    CompareSha256 /* sha256 */,
    CompareSha384 /* sha384 */,
    CompareSha512 /* sha512 */,
    CompareMd4 /* md4 */,
    CompareCrc32 /* crc32 */,
    CompareWhirlpool /* whirlpool */,
    CompareSize,
    CompareLimit /* limit */,
    CompareOffset /* offset */,
    NULL,
    NULL,
    CompareMd2 /* md2 */
};

static int attrWeights[] = {
    1, /* name */
    1, /* path */
    1, /* dict */
    4 /* md5 */,
    5 /* sha1 */,
    6 /* sha256 */,
    7 /* sha384 */,
    8 /* sha512 */,
    3 /* md4 */,
    2 /* crc32 */,
    8 /* whirlpool */,
    0, /* size */
    0 /* limit */,
    0 /* offset */,
    0, /* min */
    0, /* max */
    3 /* md2 */
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

void InitProgram(BOOL onlyValidate, const char* fileParam, apr_pool_t* root)
{
    dontRunActions = onlyValidate;
    fileParameter = fileParam;
    apr_pool_create(&pool, root);
    htVars = apr_hash_make(pool);
}

void OpenStatement(pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    apr_status_t status = APR_SUCCESS;

    parserState = state;
    status = apr_pool_create(&statementPool, pool);

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
    statement->HashAlgorithm = AlgUndefined;
    statement->Type = CtxTypeUndefined;
}

void CloseStatement(BOOL isPrintCalcTime, BOOL isPrintLowCase)
{
    DataContext dataCtx = { 0 };

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    if (dontRunActions || (parserState->errorCount > 0)) {
        goto cleanup;
    }

#ifdef GTEST
    dataCtx.PfnOutput = OutputToCppConsole;
#else
    dataCtx.PfnOutput = OutputToConsole;
#endif
    dataCtx.IsPrintCalcTime = isPrintCalcTime;
    dataCtx.IsPrintLowCase = isPrintLowCase;

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
    apr_pool_destroy(statementPool);
    statementPool = NULL;
    ht = NULL;
    whereStack = NULL;
    statement = NULL;
}

void RunHash()
{
    StringStatementContext* ctx = GetStringContext();

    if ((NULL == ctx) || (statement->HashAlgorithm == AlgUndefined) || !(ctx->BruteForce)) {
        return;
    }

    digestFunction = digestFunctions[statement->HashAlgorithm];
    hashLength = statement->HashLength;

    CrackHash(ctx->Dictionary, statement->Source, ctx->Min, ctx->Max, hashLength, digestFunction, statementPool);
}

void RunString(DataContext* dataCtx)
{
    apr_byte_t* digest = NULL;
    apr_size_t sz = 0;

    if (statement->HashAlgorithm == AlgUndefined) {
        return;
    }
    sz = hashLengths[statement->HashAlgorithm];
    digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * sz);
    digestFunctions[statement->HashAlgorithm] (digest, statement->Source, strlen(statement->Source));
    OutputDigest(digest, dataCtx, sz, statementPool);
}

void RunDir(DataContext* dataCtx)
{
    TraverseContext dirContext = { 0 };
    DirStatementContext* ctx = GetDirContext();

    if (NULL == ctx) {
        return;
    }

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;

    if (ctx->FindFiles) {
        dirContext.PfnFileHandler = FindFile;
    } else if (statement->HashAlgorithm == AlgUndefined) {
        return;
    } else {
        digestFunction = digestFunctions[statement->HashAlgorithm];
        dirContext.PfnFileHandler = CalculateFile;
    }

    dirContext.DataCtx = dataCtx;
    dirContext.IsScanDirRecursively = ctx->Recursively;

    TraverseDirectory(HackRootPath(statement->Source,
                                   statementPool), &dirContext, FilterFiles, statementPool);
}

void RunFile(DataContext* dataCtx)
{
    apr_byte_t digest[SHA512_HASH_SIZE];
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

        status = apr_file_open(&fileHandle, statement->Source, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
        if (status != APR_SUCCESS) {
            OutputErrorMessage(status, dataCtx->PfnOutput, statementPool);
            return;
        }
        status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_SIZE | APR_FINFO_IDENT | APR_FINFO_TYPE, fileHandle);
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

        if(FilterFilesInternal(&fileCtx, statementPool)) {
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
    if (statement->HashAlgorithm == AlgUndefined) {
        return;
    }
    digestFunction = digestFunctions[statement->HashAlgorithm];
    if (ctx->HashToSearch) {
        CalculateFileHash(statement->Source, digest, dataCtx->IsPrintCalcTime, NULL, dataCtx->Limit,
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

BOOL SetMin(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Min = atoi(value);
    return TRUE;
}

BOOL SetMax(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Max = atoi(value);
    return TRUE;
}

BOOL SetDictionary(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return FALSE;
    }
    GetStringContext()->Dictionary = Trim(value);
    return TRUE;
}

BOOL SetName(const char* value)
{
    if (statement->Type != CtxTypeDir) {
        return FALSE;
    }
    GetDirContext()->NameFilter = Trim(value);
    return TRUE;
}

void SetHashToSearch(const char* value, Alg algorithm)
{
    DirStatementContext* ctx = NULL;

    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return;
    }
    ctx = GetDirContext();
    ctx->HashToSearch = Trim(value);
    SetHashAlgorithm(algorithm);
}

BOOL SetMd5ToSearch(const char* value)
{
    SetHashToSearch(value, AlgMd5);
    return TRUE;
}

BOOL SetSha1ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha1);
    return TRUE;
}

BOOL SetSha256ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha256);
    return TRUE;
}

BOOL SetSha384ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha384);
    return TRUE;
}

BOOL SetSha512ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha512);
    return TRUE;
}

BOOL SetShaMd4ToSearch(const char* value)
{
    SetHashToSearch(value, AlgMd4);
    return TRUE;
}

BOOL SetShaCrc32ToSearch(const char* value)
{
    SetHashToSearch(value, AlgCrc32);
    return TRUE;
}

BOOL SetShaWhirlpoolToSearch(const char* value)
{
    SetHashToSearch(value, AlgWhirlpool);
    return TRUE;
}

BOOL SetMd2ToSearch(const char* value)
{
    SetHashToSearch(value, AlgMd2);
    return TRUE;
}

BOOL SetTigerToSearch(const char* value)
{
    SetHashToSearch(value, AlgTiger);
    return TRUE;
}

BOOL SetSha224ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha224);
    return TRUE;
}

BOOL SetRmd128ToSearch(const char* value)
{
    SetHashToSearch(value, AlgRmd128);
    return TRUE;
}

BOOL SetRmd160ToSearch(const char* value)
{
    SetHashToSearch(value, AlgRmd160);
    return TRUE;
}

BOOL SetRmd256ToSearch(const char* value)
{
    SetHashToSearch(value, AlgRmd256);
    return TRUE;
}

BOOL SetRmd320ToSearch(const char* value)
{
    SetHashToSearch(value, AlgRmd320);
    return TRUE;
}

BOOL SetLimit(const char* value)
{
    apr_status_t status = APR_SUCCESS;
    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Limit, value, NULL, 0);
    return status == APR_SUCCESS;
}

BOOL SetOffset(const char* value)
{
    apr_status_t status = APR_SUCCESS;
    if ((statement->Type != CtxTypeDir) && (statement->Type != CtxTypeFile)) {
        return FALSE;
    }
    status = apr_strtoff(&GetDirContext()->Offset, value, NULL, 0);
    return status == APR_SUCCESS;
}

void AssignAttribute(Attr code, pANTLR3_UINT8 value, void* valueToken)
{
    BOOL (* op)(const char*) = NULL;

    if (code == AttrUndefined) {
        return;
    }
    op = strOperations[code];
    if (!op) {
        return;
    }
    if(!op((const char*)value)) {
        parserState->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION,
                                                "invalid value",
                                                "error: value is invalid",
                                                ANTLR3_FALSE);
        parserState->exception->token = valueToken;
        parserState->error = ANTLR3_RECOGNITION_EXCEPTION;
    }
}

void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode, void* token)
{
    BoolOperation* op = NULL;

    if (statementPool == NULL) { // memory allocation error
        return;
    }

    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));
    op->Attribute = code;
    op->Operation = opcode;
    op->Value =  Trim(value);
    op->Token = token;
    *(BoolOperation**)apr_array_push(whereStack) = op;
}

void WhereClauseCond(CondOp opcode, void* token)
{
    WhereClauseCall(AttrUndefined, NULL, opcode, token);
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

void SetHashAlgorithm(Alg algorithm)
{
    if (statementPool == NULL) { // memory allocation error
        return;
    }
    statement->HashAlgorithm = algorithm;
    hashLength = GetDigestSize();
    statement->HashLength = hashLength;
    digestFunction = digestFunctions[algorithm];
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
    apr_size_t i = 0;

    for (; i <= hashLength - (hashLength >> 2); i += 4) {
        if (digest1[i] != digest2[i]) {
            return FALSE;
        }
        if (digest1[i + 1] != digest2[i + 1]) {
            return FALSE;
        }
        if (digest1[i + 2] != digest2[i + 2]) {
            return FALSE;
        }
        if (digest1[i + 3] != digest2[i + 3]) {
            return FALSE;
        }
    }
    return TRUE;
}

int ComparisonFailure(int result)
{
    return GetDirContext()->Operation == CondOpEq ? !result : result;
}

int CompareHashAttempt(void* hash, const char* pass, const uint32_t length)
{
    apr_byte_t attempt[SHA512_HASH_SIZE]; // hack to improve performance

    digestFunction(attempt, pass, (apr_size_t)length);
    return CompareDigests(attempt, hash);
}

void ToDigest(const char* hash, apr_byte_t* digest)
{
    size_t i = 0;
    size_t to = MIN(hashLength, strlen(hash) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(hash + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

void* CreateDigest(const char* hash, apr_pool_t* p)
{
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(p, hashLength);
    ToDigest(hash, result);
    return result;
}

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return digestFunction(digest, input, inputLen);
}

apr_status_t InitContext(void* context)
{
    return initCtxFuncs[statement->HashAlgorithm] (context);
}

apr_status_t FinalHash(apr_byte_t* digest, void* context)
{
    return finalHashFuncs[statement->HashAlgorithm] (digest, context);
}

apr_status_t UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return updateHashFuncs[statement->HashAlgorithm] (context, input, inputLen);
}

void* AllocateContext(apr_pool_t* p)
{
    return apr_pcalloc(p, contextSizes[statement->HashAlgorithm]);
}

apr_size_t GetDigestSize()
{
    return hashLengths[statement->HashAlgorithm];
}

int CompareHash(apr_byte_t* digest, const char* checkSum)
{
    apr_byte_t bytes[SHA512_HASH_SIZE]; // HACK

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
        int w1;
        int w2;

        if (i + 1 >= whereStack->nelts) {
            break;
        }
        op2 = ((BoolOperation**)whereStack->elts)[i + 1];
        if (Skip(op1->Operation) || Skip(op2->Operation)) {
            continue;
        }
        w1 = attrWeights[op1->Attribute];
        w2 = attrWeights[op2->Attribute];
        if (w1 < w2) {
            continue;
        } else if ((w1 == w2) && (opWeights[op1->Operation] <= opWeights[op2->Operation])) {
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
    apr_status_t status = APR_SUCCESS;
    apr_file_t* fileHandle = NULL;
    apr_finfo_t info = { 0 };

    fileAnsi = FromUtf8ToAnsi(fullPathToFile, p);

    status = apr_file_open(&fileHandle, fullPathToFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, fileHandle);

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
    status = apr_file_close(fileHandle);
    return APR_SUCCESS;
}

BOOL Compare(BoolOperation* op, void* context, Alg algorithm, apr_pool_t* p)
{
    apr_status_t status = APR_SUCCESS;
    apr_file_t* fileHandle = NULL;
    FileCtx* ctx = (FileCtx*)context;
    apr_byte_t digestToCompare[SHA512_HASH_SIZE];
    apr_byte_t digest[SHA512_HASH_SIZE];
    char* fullPath = NULL; // Full path to file or subdirectory
    BOOL result = FALSE;

    SetHashAlgorithm(algorithm);
    ToDigest(op->Value, digestToCompare);

    status = CalculateDigest(digest, NULL, 0);
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

BOOL CompareMd5(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgMd5, p);
}

BOOL CompareMd4(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgMd4, p);
}

BOOL CompareSha1(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgSha1, p);
}

BOOL CompareSha256(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgSha256, p);
}

BOOL CompareSha384(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgSha384, p);
}
BOOL CompareSha512(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgSha512, p);
}

BOOL CompareWhirlpool(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgWhirlpool, p);
}

BOOL CompareCrc32(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgCrc32, p);
}

BOOL CompareMd2(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgMd2, p);
}

BOOL CompareTiger(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgTiger, p);
}

BOOL CompareSha224(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgSha224, p);
}

BOOL CompareRmd128(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgRmd128, p);
}

BOOL CompareRmd160(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgRmd160, p);
}

BOOL CompareRmd256(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgRmd256, p);
}

BOOL CompareRmd320(BoolOperation* op, void* context, apr_pool_t* p)
{
    return Compare(op, context, AlgRmd320, p);
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
