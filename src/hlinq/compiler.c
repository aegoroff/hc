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

#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define FILE_INFO_COLUMN_SEPARATOR " | "
#define MAX_DEFAULT 10
#define MAX_ATTR "max"
#define MIN_ATTR "min"
#define DICT_ATTR "dict"

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_hash_t* ht = NULL;
BOOL dontRunActions = FALSE;

ContextType currentContext = File;
const char* currentId = NULL;

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

static Digest* (*hashFunctions[])(const char* string) = {
    HashMD5,
    HashSHA1,
    HashMD4,
    HashSHA256,
    HashSHA384,
    HashSHA512,
    HashWhirlpool,
    HashCrc32,
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

static void (*intOperations[])(int) = {
    NULL,
    SetLimit,
    SetOffset,
    SetMin,
    SetMax
};

static void (*strOperations[])(const char*) = {
    NULL,
    NULL,
    SetDictionary,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
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
}

void CloseStatement()
{
    Digest* digest = NULL;
    DataContext dataCtx = { 0 };
    FileStatementContext* ctxFile = NULL;
    StringStatementContext* ctxStr = NULL;

    dataCtx.PfnOutput = OutputToConsole;

    if (dontRunActions) {
        goto cleanup;
    }
    switch(currentContext) {
        case String:
            ctxStr = GetStringContext();
            if (NULL == ctxStr || ctxStr->HashAlgorithm == Undefined) {
                goto cleanup;
            }
        
            if (ctxStr->BruteForce) {
                CrackHash(ctxStr->Dictionary, ctxStr->String, ctxStr->Min, ctxStr->Max);
            } else {
                digest = hashFunctions[ctxStr->HashAlgorithm](ctxStr->String);
                OutputDigest(digest->Data, &dataCtx, digest->Size);
            }
            break;
        case File:
            // TODO: run query
            ctxFile = GetFileContext();
            if (NULL == ctxFile) {
                goto cleanup;
            }
            CrtPrintf("root: %s Recursively: %s" NEW_LINE, ctxFile->SearchRoot, ctxFile->Recursively ? "yes" : "no");
            break;
    }

cleanup:
    if (statementPool) {
        apr_pool_destroy(statementPool);
        statementPool = NULL;
    }
    currentId = NULL;
    ht = NULL;
}

void SetRecursively()
{
    GetFileContext()->Recursively = TRUE;
}

void SetBruteForce()
{
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

void SetMin(int value)
{
    if (currentContext == File) {
        return;
    }
    GetStringContext()->Min = value;
}

void SetMax(int value)
{
    if (currentContext == File) {
        return;
    }
     GetStringContext()->Max = value;
}

void SetDictionary(const char* value)
{
    if (currentContext == File) {
        return;
    }
     GetStringContext()->Dictionary = value;
}

void SetLimit(int value)
{
    if (currentContext == String) {
        return;
    }
    GetFileContext()->Limit = value;
}

void SetOffset(int value)
{
    if (currentContext == String) {
        return;
    }
    GetFileContext()->Offset = value;
}

void AssignStrAttribute(int code, pANTLR3_UINT8 value)
{
    void (*op)(const char*) = strOperations[code];
    if (!op) {
        return;
    }
    op((const char*)value);
}

void AssignIntAttribute(int code, pANTLR3_UINT8 value)
{
    void (*op)(int) = intOperations[code];
    if (!op) {
        return;
    }
    op(atoi((const char*)value));
}

void RegisterIdentifier(pANTLR3_UINT8 identifier, ContextType type)
{
    void* ctx = NULL;
    StringStatementContext* strCtx = NULL;

    switch(type) {
        case File:
            ctx = apr_pcalloc(statementPool, sizeof(FileStatementContext));
            break;
        case String:
            ctx = apr_pcalloc(statementPool, sizeof(StringStatementContext));
            strCtx = (StringStatementContext*)ctx;
            strCtx->HashAlgorithm = Undefined;
            strCtx->BruteForce = FALSE;
            break;
    }
    currentContext = type;
    currentId = (const char*)identifier;
    apr_hash_set(ht, currentId, APR_HASH_KEY_STRING, ctx);
}

BOOL CallAttiribute(pANTLR3_UINT8 identifier)
{
    return apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
}

void* GetContext()
{
    if (NULL == currentId) {
        return NULL;
    }
    return apr_hash_get(ht, currentId, APR_HASH_KEY_STRING);
}

FileStatementContext* GetFileContext()
{
    return (FileStatementContext*)GetContext();
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
    switch(currentContext) {
        case File:
            GetFileContext()->SearchRoot = tmp;
            break;
        case String:
            GetStringContext()->String = tmp;
            break;
    }
}

void SetHashAlgorithm(HASH_ALGORITHM algorithm)
{
    switch(currentContext) {
        case File:
            GetFileContext()->HashAlgorithm = algorithm;
            break;
        case String:
            GetStringContext()->HashAlgorithm = algorithm;
            GetStringContext()->HashLength = hashLengths[algorithm];
            break;
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
    return tmp == NULL ? tmp : apr_pstrdup(statementPool, tmp);
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

    digestFunction = digestFunctions[GetStringContext()->HashAlgorithm];
    hashLength = GetStringContext()->HashLength;

    // Empty string validation
    CalculateDigest(digest, NULL, 0);

    passmax = passmax ? passmax : MAX_DEFAULT;

    if (CompareHash(digest, hash)) { 
        str = "Empty string";
    } else {
        
        CalculateStringHash("1234", digest, digestFunction);
        str1234 = HashToString(digest, FALSE, hashLength);
    
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

Digest* HashSHA256(const char* string)
{
    return Hash(string, SHA256_HASH_SIZE, CalculateStringHashSHA256);
}

Digest* HashSHA384(const char* string)
{
    return Hash(string, SHA384_HASH_SIZE, CalculateStringHashSHA384);
}

Digest* HashSHA512(const char* string)
{
    return Hash(string, SHA512_HASH_SIZE, CalculateStringHashSHA512);
}

Digest* HashWhirlpool(const char* string)
{
    return Hash(string, WHIRLPOOL_DIGEST_LENGTH, CalculateStringHashWhirlpool);
}

Digest* HashCrc32(const char* string)
{
    return Hash(string, CRC32_HASH_SIZE, CalculateStringHashCrc32);
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

void CalculateStringHashSHA256(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, SHA256CalculateDigest);
}

void CalculateStringHashSHA384(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, SHA384CalculateDigest);
}

void CalculateStringHashSHA512(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, SHA512CalculateDigest);
}

void CalculateStringHashWhirlpool(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, WHIRLPOOLCalculateDigest);
}

void CalculateStringHashCrc32(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, CRC32CalculateDigest);
}