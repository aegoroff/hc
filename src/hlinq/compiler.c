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
char* currentString = NULL;
char* currentId = NULL;

void InitProgram(apr_pool_t* root)
{
    apr_pool_create(&pool, root);
}

void OpenStatement()
{
    apr_pool_create(&statementPool, pool);
    ht = apr_hash_make(statementPool);
}

void CloseStatement()
{
    if (currentId) {
        StatementContext* context = apr_hash_get(ht, (const char*)currentId, APR_HASH_KEY_STRING);
        if (context) {
            // TODO: run query
            CrtPrintf("root: %s" NEW_LINE, context->SearchRoot);
            CrtPrintf("action: %s" NEW_LINE, context->ActionTarget);
        }
    }
    
    apr_pool_destroy(statementPool);
}

void RegisterIdentifier(pANTLR3_UINT8 identifier)
{
    StatementContext* context = (StatementContext*)apr_pcalloc(statementPool, sizeof(StatementContext));;

    currentId = (const char*)identifier;
    apr_hash_set(ht, currentId, APR_HASH_KEY_STRING, context);
}

void CallAttiribute(pANTLR3_UINT8 identifier)
{
    StatementContext* context = apr_hash_get(ht, (const char*)identifier, APR_HASH_KEY_STRING);
    if (!context) {
        CrtPrintf("error: unknown identifier %s", identifier);
    }
}

void SetCurrentString(pANTLR3_UINT8 str)
{
    currentString = (char*)str;
}

void SetSearchRoot(pANTLR3_UINT8 str)
{
    char* tmp = Trim(str);
    StatementContext* context = apr_hash_get(ht, (const char*)currentId, APR_HASH_KEY_STRING);
    
    if (context && str) {
        context->SearchRoot = apr_pstrdup(statementPool, tmp);
    }
}

void SetActionTarget(pANTLR3_UINT8 str)
{
    StatementContext* context = apr_hash_get(ht, (const char*)currentId, APR_HASH_KEY_STRING);
    char* tmp = Trim(str);
    
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