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

#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define FILE_INFO_COLUMN_SEPARATOR " | "
#define MAX_DEFAULT "10"

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_hash_t* ht = NULL;
BOOL dontRunActions = FALSE;
StatementContext* current = NULL;

static char* alphabet = DIGITS LOW_CASE UPPER_CASE;

static apr_size_t hashLengths[] = {
    APR_MD5_DIGESTSIZE,
    APR_SHA1_DIGESTSIZE,
    APR_MD4_DIGESTSIZE
};

static Digest* (*hashFunctions[])(const char* string) = {
    HashMD5,
    HashSHA1,
    HashMD4
};

static apr_status_t (*digestFunctions[])(apr_byte_t* digest, const void* input, const apr_size_t inputLen) = {
    MD5CalculateDigest,
    SHA1CalculateDigest,
    MD4CalculateDigest
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
}
void CloseStatement(const char* identifier)
{
    StatementContext* context = NULL;
    Digest* digest = NULL;
    DataContext dataCtx = { 0 };

    dataCtx.PfnOutput = OutputToConsole;
    
    if (!identifier) {
        goto cleanup;
    }
    context = apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
    current = context;
    if (NULL == context) {
        goto cleanup;
    }
    if (dontRunActions) {
        goto cleanup;
    }
    if (context->String) {
        if (context->BruteForce) {
            CrackHash(context->Dictionary, context->String, context->Min, context->Max);
        } else {
            digest = hashFunctions[context->HashAlgorithm](context->String);
            OutputDigest(digest->Data, &dataCtx, digest->Size);
        }

        goto cleanup;
    }
    // TODO: run query
    CrtPrintf("root: %s Recursively: %s" NEW_LINE, context->SearchRoot, context->Recursively ? "yes" : "no");
    CrtPrintf("action: %s" NEW_LINE, context->ActionTarget);
cleanup:
    apr_pool_destroy(statementPool);
}

void CreateStatementContext(const char* identifier)
{
    StatementContext* context = (StatementContext*)apr_pcalloc(statementPool, sizeof(StatementContext));
    apr_hash_set(ht, (char*)identifier, APR_HASH_KEY_STRING, context);
}

void SetRecursively(const char* identifier)
{
    StatementContext* context = apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
    context->Recursively = TRUE;
}

void SetBruteForce()
{
    StatementContext* context = NULL;
    context = apr_hash_get(ht, SPECIAL_STR_ID, APR_HASH_KEY_STRING);
    
    if (context) {
        context->BruteForce = TRUE;
        context->Min = 1;
        context->Max = 10;
        context->Dictionary = alphabet;
    }
}

BOOL CallAttiribute(pANTLR3_UINT8 identifier)
{
    return apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING) != NULL;
}

void SetSearchRoot(pANTLR3_UINT8 str, const char* identifier)
{
    StatementContext* context = NULL;
    char* tmp = Trim(str);
    
    if (NULL == identifier) {
        return;
    }
    context = apr_hash_get(ht, identifier, APR_HASH_KEY_STRING);
    
    if (context && str) {
        context->SearchRoot = apr_pstrdup(statementPool, tmp);
    }
}

void SetString(const char* str)
{
    StatementContext* context = NULL;
    char* tmp = Trim(str);
    
    if (NULL == tmp) {
        return;
    }
    context = apr_hash_get(ht, SPECIAL_STR_ID, APR_HASH_KEY_STRING);
    
    if (context) {
        context->String = apr_pstrdup(statementPool, tmp);
    }
}

void SetHashAlgorithm(HASH_ALGORITHM algorithm)
{
    StatementContext* context = NULL;
    context = apr_hash_get(ht, SPECIAL_STR_ID, APR_HASH_KEY_STRING);
    
    if (context) {
        context->HashAlgorithm = algorithm;
        context->HashLength = hashLengths[algorithm];
    }
}

void SetActionTarget(pANTLR3_UINT8 str, const char* identifier)
{
    StatementContext* context = apr_hash_get(ht, identifier, APR_HASH_KEY_STRING);
    char* tmp = Trim(str);

    if (NULL == identifier) {
        return;
    }
    
    if (context && tmp) {
        context->ActionTarget = apr_pstrdup(statementPool, tmp);
    }
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
    return tmp;
}

void OutputToConsole(OutputContext* ctx)
{
    if (ctx == NULL) {
        return;
    }
    CrtPrintf("%s", ctx->StringToPrint);
    if (ctx->IsPrintSeparator) {
        CrtPrintf(FILE_INFO_COLUMN_SEPARATOR);
    }
    if (ctx->IsFinishLine) {
        NewLine();
    }
}

void OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_size_t sz)
{
    OutputContext output = { 0 };
    output.IsFinishLine = TRUE;
    output.IsPrintSeparator = FALSE;
    output.StringToPrint = HashToString(digest, ctx->IsPrintLowCase, sz);
    ctx->PfnOutput(&output);
}

const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_size_t sz)
{
    int i = 0;
    char* str = apr_pcalloc(statementPool, sz * BYTE_CHARS_SIZE + 1); // iteration ponter
    char* result = str; // result pointer

    for (; i < sz; ++i) {
        apr_snprintf(str, BYTE_CHARS_SIZE + 1, isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
        str += BYTE_CHARS_SIZE;
    }
    return result;
}

/*!
 * It's so ugly to improve performance
 */
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2, apr_size_t size)
{
    int i = 0;

    for (; i <= size - (size >> 2); i += 4) {
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
    apr_byte_t attempt[64]; // hack to improve performance
    
    digestFunctions[current->HashAlgorithm](attempt, pass, length);
    return CompareDigests(attempt, hash, current->HashLength);
}

void ToDigest(const char* hash, apr_byte_t* digest)
{
    int i = 0;
    int to = MIN(current->HashLength, strlen(hash) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(hash + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}

void* CreateDigest(const char* hash, apr_pool_t* p)
{
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(p, current->HashLength);
    ToDigest(hash, result);
    return result;
}

apr_status_t CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return digestFunctions[current->HashAlgorithm](digest, input, inputLen);
}

int CompareHash(apr_byte_t* digest, const char* checkSum)
{
    apr_byte_t bytes[64];

    ToDigest(checkSum, bytes);
    return CompareDigests(bytes, digest, current->HashLength);
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
    apr_byte_t digest[64]; // HACK!
    uint64_t attempts = 0;
    Time time = { 0 };
    double ratio = 0;
    double maxAttepts = 0;
    Time maxTime = { 0 };

    CalculateStringHash("1234", digest, digestFunctions[current->HashAlgorithm]);
    str1234 = HashToString(digest, FALSE, current->HashLength);
    
    StartTimer();

    BruteForce(1,
                atoi(MAX_DEFAULT),
                alphabet,
                str1234,
                &attempts,
                CreateDigest,
                statementPool);

    StopTimer();
    time = ReadElapsedTime();
    ratio = attempts / time.seconds;

    attempts = 0;
    StartTimer();

    // Empty string validation
    CalculateDigest(digest, NULL, 0);

    passmax = passmax ? passmax : atoi(MAX_DEFAULT);

    if (!CompareHash(digest, hash)) {
        passmax = passmax ? passmax : atoi(MAX_DEFAULT);
        maxAttepts = pow(strlen(PrepareDictionary(dict)), passmax);
        maxTime = NormalizeTime(maxAttepts / ratio);
        maxTimeMsg = (char*)apr_pcalloc(statementPool, maxTimeMsgSz + 1);
        TimeToString(maxTime, maxTimeMsgSz, maxTimeMsg);
        CrtPrintf("May take approximatelly: %s (%.0f attempts)", maxTimeMsg, maxAttepts);
        str = BruteForce(passmin, passmax, dict, hash, &attempts, CreateDigest, statementPool);
    } else {
        str = "Empty string";
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

void CalculateStringHash(
    const char* string, 
    apr_byte_t* digest, 
    apr_status_t (*fn)(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
    )
{
    fn(digest, string, strlen(string));
}

Digest* Hash(const char* string, apr_size_t size, void (*fn)(const char* string,  apr_byte_t* digest))
{
    Digest* result = (Digest*)apr_pcalloc(statementPool, sizeof(Digest));
    result->Size = size;
    result->Data = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * result->Size);
    fn(string, result->Data);
    return result;
}

Digest* HashMD4(const char* string)
{
    return Hash(string, APR_MD4_DIGESTSIZE, CalculateStringHashMD4);
}

Digest* HashMD5(const char* string)
{
    return Hash(string, APR_MD5_DIGESTSIZE, CalculateStringHashMD5);
}

Digest* HashSHA1(const char* string)
{
    return Hash(string, APR_SHA1_DIGESTSIZE, CalculateStringHashSHA1);
}

void CalculateStringHashMD4(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, MD4CalculateDigest);
}

void CalculateStringHashMD5(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, MD5CalculateDigest);
}

void CalculateStringHashSHA1(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, SHA1CalculateDigest);
}