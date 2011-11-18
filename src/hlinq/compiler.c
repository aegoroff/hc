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

apr_pool_t* pool = NULL;
apr_pool_t* statementPool = NULL;
apr_hash_t* ht = NULL;
BOOL dontRunActions = FALSE;

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
    // TODO: run query
    CrtPrintf("root: %s Recursively: %s" NEW_LINE, context->SearchRoot, context->Recursively ? "yes" : "no");
    CrtPrintf("action: %s" NEW_LINE, context->ActionTarget);
cleanup:
    apr_pool_destroy(statementPool);
}

void CreateStatementContext(const char* identifier)
{
    StatementContext* context = (StatementContext*)apr_pcalloc(statementPool, sizeof(StatementContext));;
    apr_hash_set(ht, (char*)identifier, APR_HASH_KEY_STRING, context);
}

void SetRecursively(const char* identifier)
{
    StatementContext* context = apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
    context->Recursively = TRUE;
}

BOOL CallAttiribute(pANTLR3_UINT8 identifier)
{
    StatementContext* context = apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
    if (!context) {
        CrtPrintf("error: unknown identifier %s" NEW_LINE, identifier);
        return FALSE;
    }
    return TRUE;
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