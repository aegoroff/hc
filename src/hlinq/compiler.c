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

#include "compiler.h"
#include "..\srclib\lib.h"
#include "md5.h"

#define BYTE_CHARS_SIZE 2   // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define FILE_INFO_COLUMN_SEPARATOR " | "

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_hash_t* ht = NULL;
BOOL dontRunActions = FALSE;

static Digest* (*hashFunctions[])(const char* string) = {
    HashMD5,
    HashSHA1
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
    if (NULL == context) {
        goto cleanup;
    }
    if (dontRunActions) {
        goto cleanup;
    }
    if (context->String) {
        // TODO: string actions
        digest = hashFunctions[context->HashAlgorithm](context->String);
        OutputDigest(digest->Digest, &dataCtx, digest->Size);
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

void CalculateStringHash(
    const char* string, 
    apr_byte_t* digest, 
    apr_status_t (*fn)(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
    )
{
    fn(digest, string, strlen(string));
}

Digest* HashMD5(const char* string)
{
    Digest* result = (Digest*)apr_pcalloc(statementPool, sizeof(Digest));
    result->Size = APR_MD5_DIGESTSIZE;
    result->Digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * result->Size);
    CalculateStringHashMD5(string, result->Digest);
    return result;
}

Digest* HashSHA1(const char* string)
{
    Digest* result = (Digest*)apr_pcalloc(statementPool, sizeof(Digest));
    result->Size = 20; // TODO: APR_SHA1_DIGESTSIZE
    result->Digest = (apr_byte_t*)apr_pcalloc(statementPool, sizeof(apr_byte_t) * result->Size); 
    CalculateStringHashSHA1(string, result->Digest);
    return result;
}

void CalculateStringHashMD5(const char* string,  apr_byte_t* digest)
{
    CalculateStringHash(string, digest, MD5CalculateDigest);
}

void CalculateStringHashSHA1(const char* string,  apr_byte_t* digest)
{

}