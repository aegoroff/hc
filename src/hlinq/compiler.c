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
#ifdef GTEST
    #include "displayError.h"
#endif

#define MAX_DEFAULT 10
#define MAX_ATTR "max"
#define MIN_ATTR "min"
#define DICT_ATTR "dict"
#define ARRAY_INIT_SZ           32

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_hash_t* ht = NULL;
apr_array_header_t* whereStack;
BOOL dontRunActions = FALSE;

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

static BOOL (*comparators[])(const char*, CondOp, void*, apr_pool_t*) = {
    CompareName,
    ComparePath,
    NULL,
    NULL /* md5 */,
    NULL /* sha1 */,
    NULL /* sha256 */,
    NULL /* sha384 */,
    NULL /* sha512 */,
    NULL /* md4 */,
    NULL /* crc32 */,
    NULL /* whirlpool */,
    CompareSize,
    NULL /* limit */,
    NULL /* offset */,
    NULL,
    NULL
};

void InitProgram(BOOL onlyValidate, apr_pool_t* root)
{
    dontRunActions = onlyValidate;
    apr_pool_create(&pool, root);
}

void OpenStatement()
{
    apr_pool_create(&statementPool, pool);
    ht = apr_hash_make(statementPool);
    whereStack = apr_array_make(statementPool, ARRAY_INIT_SZ, sizeof(BoolOperation*));
    statement = (StatementCtx*)apr_pcalloc(statementPool, sizeof(StatementCtx));
    statement->HashAlgorithm = AlgUndefined;
    statement->Type = CtxTypeUndefined;
}

void CloseStatement(ANTLR3_UINT32 errors, BOOL isPrintCalcTime)
{
    DataContext dataCtx = { 0 };
#ifdef GTEST
    dataCtx.PfnOutput = OutputToCppConsole;
#else
    dataCtx.PfnOutput = OutputToConsole;
#endif
    dataCtx.IsPrintCalcTime = isPrintCalcTime;

    if (dontRunActions || errors > 0) {
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

    if (statement->HashAlgorithm == AlgUndefined) {
        return;
    }
    digestFunction = digestFunctions[statement->HashAlgorithm];

    dataCtx->Limit = ctx->Limit;
    dataCtx->Offset = ctx->Offset;
    dataCtx->HashToSearch = ctx->HashToSearch;
    dirContext.DataCtx = dataCtx;
    dirContext.PfnFileHandler = CalculateFile;
    dirContext.IsScanDirRecursively = ctx->Recursively;

    CompilePattern(ctx->NameFilter, &dirContext.IncludePattern, pool);
    TraverseDirectory(HackRootPath(statement->Source, statementPool), &dirContext, FilterFiles, statementPool);
}

void SetRecursively()
{
    GetDirContext()->Recursively = TRUE;
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
    
    if (statement->Type != CtxTypeDir) {
        return;
    }
    ctx = GetDirContext();
    ctx->HashToSearch = value;
    statement->HashAlgorithm = algorithm;
    hashLength = GetDigestSize();
    statement->HashLength = hashLength;
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
    if (statement->Type != CtxTypeDir) {
        return;
    }
    GetDirContext()->Limit = atoi(value);
}

void SetOffset(const char* value)
{
    if (statement->Type != CtxTypeDir) {
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

void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode)
{
    BoolOperation* op = NULL;
    const char* v = NULL;

    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));

    op->Attribute = code;
    op->Operation = opcode;
    v = value != NULL && value[0] != '\'' && value[0] != '\"' ? value : Trim(value);
    op->Value =  apr_pstrdup(statementPool, v);

    *(BoolOperation**)apr_array_push(whereStack) = op;
}

void WhereClauseCond(CondOp opcode)
{
    BoolOperation* op = NULL;
    op = (BoolOperation*)apr_pcalloc(statementPool, sizeof(BoolOperation));
    op->Operation = opcode;
    op->Attribute = AttrUndefined;
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

BOOL CallAttiribute(pANTLR3_UINT8 identifier)
{
    return apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING) != NULL;
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
    statement->HashLength = hashLengths[algorithm];
}

char* Trim(pANTLR3_UINT8 str)
{
    size_t len = 0;
    char* tmp = NULL;
    
    if (str) {
        tmp = (char*)str+1; // leading " or '
        len = strlen(tmp);
        tmp[len - 1] = '\0';
    }
    return tmp == NULL ? tmp : apr_pstrdup(statementPool, tmp);
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
    char* maxTimeMsg = NULL;
    int maxTimeMsgSz = 63;
    const char* str1234 = NULL;
    apr_byte_t digest[SHA512_HASH_SIZE]; // HACK!
    uint64_t attempts = 0;
    Time time = { 0 };
    double ratio = 0;
    double maxAttepts = 0;
    Time maxTime = { 0 };

    // Empty string validation
    CalculateDigest(digest, NULL, 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if (CompareHash(digest, hash)) { 
        str = "Empty string";
    } else {
        
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
    BOOL (*comparator)(const char*, CondOp, void*, apr_pool_t*) = NULL;
    BOOL left = FALSE;
    BOOL right = FALSE;
    FileCtx fileCtx = { 0 };
    
    if (whereStack->nelts > 0) {
        stack = apr_array_make(p, ARRAY_INIT_SZ, sizeof(BOOL));
    }

    for (i = 0; i < whereStack->nelts; i++) {
        BoolOperation* op = ((BoolOperation**)whereStack->elts)[i];
        
        if (op->Operation == CondOpAnd || op->Operation == CondOpOr) {
            left = *((BOOL*)apr_array_pop(stack));
            right = *((BOOL*)apr_array_pop(stack));

            if (op->Operation == CondOpAnd) {
                *(BOOL*)apr_array_push(stack) = left && right;
            } else {
                *(BOOL*)apr_array_push(stack) = left || right;
            }

        } else if (op->Operation == CondOpNot) {
            left = *((BOOL*)apr_array_pop(stack));
            *(BOOL*)apr_array_push(stack) = !left;
        } else {
            comparator = comparators[op->Attribute];
            if (comparator == NULL) {
                *(BOOL*)apr_array_push(stack) = TRUE;
            } else {
                fileCtx.Dir = dir;
                fileCtx.Info = info;
                *(BOOL*)apr_array_push(stack) = comparator(op->Value, op->Operation, &fileCtx, p);
            }
        }
    }
    return i == 0 || *((BOOL*)apr_array_pop(stack));
}

BOOL CompareStr(const char* value, CondOp operation, const char* str)
{
    switch(operation) {
        case CondOpMatch:
            return apr_fnmatch(value, str, APR_FNM_CASE_BLIND) == APR_SUCCESS;
        case CondOpNotMatch:
            return apr_fnmatch(value, str, APR_FNM_CASE_BLIND) != APR_SUCCESS;
        case CondOpEq:
            return strcmp(value, str) == 0;
        case CondOpNotEq:
            return strcmp(value, str) != 0;
    }

    return FALSE;
}

BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer)
{
    int size = atoi(integer);

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

BOOL CompareName(const char* value, CondOp operation, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
    return CompareStr(value, operation, ctx->Info->name);
}

BOOL ComparePath(const char* value, CondOp operation, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
    char* fullPath = NULL; // Full path to file or subdirectory

    apr_filepath_merge(&fullPath,
                        ctx->Dir,
                        ctx->Info->name,
                        APR_FILEPATH_NATIVE,
                        p); // IMPORTANT: so as not to use strdup

    return CompareStr(value, operation, fullPath);
}

BOOL CompareSize(const char* value, CondOp operation, void* context, apr_pool_t* p)
{
    FileCtx* ctx = (FileCtx*)context;
    return CompareInt(ctx->Info->size, operation, value);
}