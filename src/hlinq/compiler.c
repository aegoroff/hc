/*!
 * \brief   The file contains HLINQ compiler API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
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
#include "pcre.h"
#ifdef GTEST
    #include "displayError.h"
#endif

#define MAX_DEFAULT 10
#define MAX_ATTR "max"
#define MIN_ATTR "min"
#define DICT_ATTR "dict"
#define ARRAY_INIT_SZ           32
#define UNKNOWN_IDENTIFIER "unknown identifier"

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_pool_t* filePool = NULL;
apr_hash_t* ht = NULL;
apr_array_header_t* whereStack;
BOOL dontRunActions = FALSE;
pANTLR3_RECOGNIZER_SHARED_STATE parserState = NULL;

StatementCtx* statement = NULL;

apr_status_t (*digestFunction)(apr_byte_t* digest, const void* input, const apr_size_t inputLen) = NULL;
apr_size_t hashLength = 0;

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

static apr_size_t hashLengths[] = {
    APR_MD5_DIGESTSIZE,
    APR_SHA1_DIGESTSIZE,
    APR_MD4_DIGESTSIZE,
    SHA256_HASH_SIZE,
    SHA384_HASH_SIZE,
    SHA512_HASH_SIZE,
    WHIRLPOOL_DIGEST_LENGTH,
    CRC32_HASH_SIZE
};

static apr_status_t (*digestFunctions[])(apr_byte_t* digest, const void* input, const apr_size_t inputLen) = {
    MD5CalculateDigest,
    SHA1CalculateDigest,
    MD4CalculateDigest,
    SHA256CalculateDigest,
    SHA384CalculateDigest,
    SHA512CalculateDigest,
    WHIRLPOOLCalculateDigest,
    CRC32CalculateDigest
};

static apr_status_t (*initCtxFuncs[])(void* context) = {
    MD5InitContext,
    SHA1InitContext,
    MD4InitContext,
    SHA256InitContext,
    SHA384InitContext,
    SHA512InitContext,
    WHIRLPOOLInitContext,
    CRC32InitContext
};

static apr_status_t (*finalHashFuncs[])(apr_byte_t* digest, void* context) = {
    MD5FinalHash,
    SHA1FinalHash,
    MD4FinalHash,
    SHA256FinalHash,
    SHA384FinalHash,
    SHA512FinalHash,
    WHIRLPOOLFinalHash,
    CRC32FinalHash
};

static apr_status_t (*updateHashFuncs[])(void* context, const void* input, const apr_size_t inputLen) = {
    MD5UpdateHash,
    SHA1UpdateHash,
    MD4UpdateHash,
    SHA256UpdateHash,
    SHA384UpdateHash,
    SHA512UpdateHash,
    WHIRLPOOLUpdateHash,
    CRC32UpdateHash
};

static size_t contextSizes[] = {
    sizeof(apr_md5_ctx_t),
    sizeof(apr_sha1_ctx_t),
    sizeof(apr_md4_ctx_t),
    sizeof(SHA256Context),
    sizeof(SHA384Context),
    sizeof(SHA512Context),
    sizeof(WHIRLPOOL_CTX),
    sizeof(Crc32Context)
};

static void (*strOperations[])(const char*) = {
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
    SetMax
};

static BOOL (*comparators[])(BoolOperation*, void*, apr_pool_t*) = {
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
    NULL
};

void InitProgram(BOOL onlyValidate, apr_pool_t* root)
{
    dontRunActions = onlyValidate;
    apr_pool_create(&pool, root);
}

void OpenStatement(pANTLR3_RECOGNIZER_SHARED_STATE state)
{
    parserState = state;
    apr_pool_create(&statementPool, pool);
    ht = apr_hash_make(statementPool);
    whereStack = apr_array_make(statementPool, ARRAY_INIT_SZ, sizeof(BoolOperation*));
    statement = (StatementCtx*)apr_pcalloc(statementPool, sizeof(StatementCtx));
    statement->HashAlgorithm = AlgUndefined;
    statement->Type = CtxTypeUndefined;
}

void CloseStatement(BOOL isPrintCalcTime, BOOL isPrintLowCase)
{
    DataContext dataCtx = { 0 };
#ifdef GTEST
    dataCtx.PfnOutput = OutputToCppConsole;
#else
    dataCtx.PfnOutput = OutputToConsole;
#endif
    dataCtx.IsPrintCalcTime = isPrintCalcTime;
    dataCtx.IsPrintLowCase = isPrintLowCase;

    pcre_malloc = FileAlloc;

    if (dontRunActions || parserState->errorCount > 0) {
        goto cleanup;
    }
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
        default:
            goto cleanup;
            break;
    }

cleanup:
    if (statementPool) {
        apr_pool_destroy(statementPool);
        statementPool = NULL;
    }
    ht = NULL;
    whereStack = NULL;
    statement = NULL;
}

void RunHash()
{
    StringStatementContext* ctx = GetStringContext();

    if (NULL == ctx || statement->HashAlgorithm == AlgUndefined || !(ctx->BruteForce)) {
        return;
    }
    
    digestFunction = digestFunctions[statement->HashAlgorithm];
    hashLength = statement->HashLength;
    
    CrackHash(ctx->Dictionary, statement->Source, ctx->Min, ctx->Max);
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
    digestFunctions[statement->HashAlgorithm](digest, statement->Source, strlen(statement->Source));
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

    TraverseDirectory(HackRootPath(statement->Source, statementPool), &dirContext, FilterFiles, statementPool);
}

void RunFile(DataContext* dataCtx)
{
    apr_byte_t digest[SHA512_HASH_SIZE];
    DirStatementContext* ctx = GetDirContext();

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
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
    GetDirContext()->Recursively = TRUE;
}

void SetFindFiles()
{
    GetDirContext()->FindFiles = TRUE;
}

void SetBruteForce()
{
    if (statement->Type != CtxTypeHash) {
        return;
    }
    GetStringContext()->BruteForce = TRUE;
    if ( GetStringContext()->Min == 0) {
         GetStringContext()->Min = 1;
    }
    if ( GetStringContext()->Max == 0) {
         GetStringContext()->Max = MAX_DEFAULT;
    }
    if ( GetStringContext()->Dictionary == NULL) {
         GetStringContext()->Dictionary = alphabet;
    }
}

void SetMin(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return;
    }
    GetStringContext()->Min = atoi(value);
}

void SetMax(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return;
    }
     GetStringContext()->Max = atoi(value);
}

void SetDictionary(const char* value)
{
    if (statement->Type != CtxTypeHash) {
        return;
    }
     GetStringContext()->Dictionary = Trim(value);
}

void SetName(const char* value)
{
    if (statement->Type != CtxTypeDir) {
        return;
    }
    GetDirContext()->NameFilter = Trim(value);
}

void SetHashToSearch(const char* value, Alg algorithm)
{
    DirStatementContext* ctx = NULL;
    
    if (statement->Type != CtxTypeDir && statement->Type != CtxTypeFile) {
        return;
    }
    ctx = GetDirContext();
    ctx->HashToSearch = Trim(value);
    SetHashAlgorithm(algorithm);
}

void SetMd5ToSearch(const char* value)
{
    SetHashToSearch(value, AlgMd5);
}

void SetSha1ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha1);
}

void SetSha256ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha256);
}

void SetSha384ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha384);
}

void SetSha512ToSearch(const char* value)
{
    SetHashToSearch(value, AlgSha512);
}

void SetShaMd4ToSearch(const char* value)
{
    SetHashToSearch(value, AlgMd4);
}

void SetShaCrc32ToSearch(const char* value)
{
    SetHashToSearch(value, AlgCrc32);
}

void SetShaWhirlpoolToSearch(const char* value)
{
    SetHashToSearch(value, AlgWhirlpool);
}

void SetLimit(const char* value)
{
    if (statement->Type != CtxTypeDir && statement->Type != CtxTypeFile) {
        return;
    }
    GetDirContext()->Limit = atoi(value);
}

void SetOffset(const char* value)
{
    if (statement->Type != CtxTypeDir && statement->Type != CtxTypeFile) {
        return;
    }
    GetDirContext()->Offset = atoi(value);
}

void AssignAttribute(Attr code, pANTLR3_UINT8 value)
{
    void (*op)(const char*) = NULL;
    
    if (code == AttrUndefined) {
        return;
    }
    op = strOperations[code];
    if (!op) {
        return;
    }
    op((const char*) value);
}

void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode, void* token)
{
    BoolOperation* op = NULL;

    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));

    op->Attribute = code;
    op->Operation = opcode;
    op->Value =  Trim(value);
    op->Token = token;

    *(BoolOperation**)apr_array_push(whereStack) = op;
}

void WhereClauseCond(CondOp opcode, void* token)
{
    BoolOperation* op = NULL;
    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));
    op->Operation = opcode;
    op->Attribute = AttrUndefined;
    op->Token = token;
    *(BoolOperation**)apr_array_push(whereStack) = op;
}

void DefineQueryType(CtxType type)
{
    statement->Type = type;
}

void RegisterIdentifier(pANTLR3_UINT8 identifier)
{
    void* ctx = NULL;

    switch(statement->Type) {
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
    if (apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING) != NULL) {
        return TRUE;
    }
    parserState->exception = antlr3ExceptionNew(ANTLR3_RECOGNITION_EXCEPTION, UNKNOWN_IDENTIFIER, "error: " UNKNOWN_IDENTIFIER, ANTLR3_FALSE);
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

void SetSource(pANTLR3_UINT8 str)
{
    char* tmp = Trim(str);
   
    if (NULL == tmp) {
        return;
    }
    statement->Source = tmp;
}

void SetHashAlgorithm(Alg algorithm)
{
    statement->HashAlgorithm = algorithm;
    hashLength = GetDigestSize();
    statement->HashLength = hashLength;
    digestFunction = digestFunctions[algorithm];
}

BOOL IsStringBorder(pANTLR3_UINT8 str, size_t ix)
{
    return str[ix] == '\'' || str[ix] == '\"';
}

char* Trim(pANTLR3_UINT8 str)
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
    if (len > 0 && IsStringBorder((pANTLR3_UINT8)tmp, len - 1)) {
        tmp[len - 1] = '\0'; // trailing " or '
    }
    return tmp;
}

/*!
 * It's so ugly to improve performance
 */
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2)
{
    int i = 0;

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
    
    digestFunction(attempt, pass, length);
    return CompareDigests(attempt, hash);
}

void ToDigest(const char* hash, apr_byte_t* digest)
{
    int i = 0;
    int to = MIN(hashLength, strlen(hash) / BYTE_CHARS_SIZE);

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
    return initCtxFuncs[statement->HashAlgorithm](context);
}

apr_status_t FinalHash(apr_byte_t* digest, void* context)
{
    return finalHashFuncs[statement->HashAlgorithm](digest, context);
}

apr_status_t UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return updateHashFuncs[statement->HashAlgorithm](context, input, inputLen);
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

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax)
{
    char* str = NULL;

    apr_byte_t digest[SHA512_HASH_SIZE]; // HACK!
    uint64_t attempts = 0;
    Time time = { 0 };
    

    // Empty string validation
    CalculateDigest(digest, NULL, 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if (CompareHash(digest, hash)) { 
        str = "Empty string";
    } else {
        char* maxTimeMsg = NULL;
        int maxTimeMsgSz = 63;
        double ratio = 0;
        double maxAttepts = 0;
        Time maxTime = { 0 };
        const char* str1234 = NULL;
        
        digestFunction(digest, "1234", 4);
        str1234 = HashToString(digest, FALSE, hashLength, statementPool);
    
        StartTimer();

        BruteForce(1,
                    MAX_DEFAULT,
                    alphabet,
                    str1234,
                    &attempts,
                    CreateDigest,
                    statementPool);

        StopTimer();
        time = ReadElapsedTime();
        ratio = attempts / time.seconds;

        attempts = 0;

        maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
        maxTime = NormalizeTime(maxAttepts / ratio);
        maxTimeMsg = (char*)apr_pcalloc(statementPool, maxTimeMsgSz + 1);
        TimeToString(maxTime, maxTimeMsgSz, maxTimeMsg);
        CrtPrintf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        StartTimer();
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, statementPool);
    }

    StopTimer();
    time = ReadElapsedTime();
    CrtPrintf(NEW_LINE "Attempts: %llu Time " FULL_TIME_FMT,
              attempts,
              time.hours,
              time.minutes,
              time.seconds);
    NewLine();
    if (str != NULL) {
        CrtPrintf("Initial string is: %s", str);
    } else {
        CrtPrintf("Nothing found");
    }
    NewLine();
}

BOOL FilterFiles(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* p)
{
    int i;
    apr_array_header_t* stack = NULL;
    
    if (whereStack->nelts > 0) {
        stack = apr_array_make(p, ARRAY_INIT_SZ, sizeof(BOOL));
    }

    for (i = 0; i < whereStack->nelts; i++) {
        BOOL left;
        BOOL right;
        FileCtx fileCtx = { 0 };
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
                }
                break;
            case CondOpNot:
                left = *((BOOL*)apr_array_pop(stack));
                *(BOOL*)apr_array_push(stack) = !left;
                break;
            default:
                {
                    BOOL (*comparator)(BoolOperation*, void*, apr_pool_t*) = comparators[op->Attribute];

                    if (comparator == NULL) {
                        *(BOOL*)apr_array_push(stack) = TRUE;
                    } else {
                        // optimization
                        if (i+1 < whereStack->nelts) {
                            BoolOperation* ahead = ((BoolOperation**)whereStack->elts)[i+1];
                            if (ahead->Operation == CondOpAnd || ahead->Operation == CondOpOr) {
                                left = *((BOOL*)apr_array_pop(stack));
                        
                                if (ahead->Operation == CondOpAnd && !left || ahead->Operation == CondOpOr && left) {
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
                            fileCtx.Dir = dir;
                            fileCtx.Info = info;
                            fileCtx.PfnOutput = ((DataContext*)ctx->DataCtx)->PfnOutput;
                            *(BOOL*)apr_array_push(stack) = comparator(op, &fileCtx, p);
                        }
                    }
                }
                break;
        }
    }
    return i == 0 || *((BOOL*)apr_array_pop(stack));
}

void* FileAlloc(size_t size)
{
    return apr_palloc(filePool, size);
}

BOOL MatchStr(const char* value, CondOp operation, const char* str, apr_pool_t* p)
{
    pcre* re = NULL;
    const char* error = NULL;
    int   erroffset = 0;
    int   rc = 0;
    int   flags  = PCRE_NOTEMPTY;

    filePool = p;
    
    re = pcre_compile (value,          /* the pattern */
                       PCRE_UTF8,
                       &error,         /* for error message */
                       &erroffset,     /* for error offset */
                       0);             /* use default character tables */

    if (!re) {
        return FALSE;
    }

    if (!strstr(value, "^")) {
        flags |= PCRE_NOTBOL;
    }
    if (!strstr(value, "$")) {
        flags |= PCRE_NOTEOL;
    }

    rc = pcre_exec (
        re,                   /* the compiled pattern */
        0,                    /* no extra data - pattern was not studied */
        str,                  /* the string to match */
        strlen(str),          /* the length of the string */
        0,                    /* start at offset 0 in the subject */
        flags,
        NULL,              /* output vector for substring information */
        0);           /* number of elements in the output vector */
    
    switch(operation) {
        case CondOpMatch:
            return rc >= 0;
        case CondOpNotMatch:
            return rc < 0;
    }

    return FALSE;
}

BOOL CompareStr(const char* value, CondOp operation, const char* str, apr_pool_t* p)
{
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

BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer)
{
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
                        p); // IMPORTANT: so as not to use strdup

    return CompareStr(op->Value, op->Operation, fullPath, p);
}

BOOL CompareSize(BoolOperation* op, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
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
    if (CompareDigests(digest, digestToCompare) && ctx->Info->size == 0) { // Empty file optimization
        result = TRUE;
        goto ret;
    }

    apr_filepath_merge(&fullPath,
                        ctx->Dir,
                        ctx->Info->name,
                        APR_FILEPATH_NATIVE,
                        p); // IMPORTANT: so as not to use strdup

    status = apr_file_open(&fileHandle, fullPath, APR_READ | APR_BINARY, APR_FPROT_WREAD, p);
    if (status != APR_SUCCESS) {
        result = FALSE;
        goto ret;
    }

    CalculateHash(fileHandle, ctx->Info->size, digest, GetDirContext()->Limit, GetDirContext()->Offset, ctx->PfnOutput, p);

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

BOOL CompareLimit(BoolOperation* op, void* context, apr_pool_t* p)
{
    apr_off_t limit = 0;
    apr_strtoff(&limit, op->Value, NULL, 0);
    GetDirContext()->Limit = limit;
    return TRUE;
}

BOOL CompareOffset(BoolOperation* op, void* context, apr_pool_t* p)
{
    apr_off_t offset = 0;
    FileCtx* ctx = (FileCtx*)context;

    apr_strtoff(&offset, op->Value, NULL, 0);
    if (ctx->Info->size < offset) {
        return FALSE;
    }
    GetDirContext()->Offset = offset;
    return TRUE;
}