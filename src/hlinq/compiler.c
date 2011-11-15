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
apr_hash_t* ht = NULL;
char* currentString = NULL;

void InitProgram(apr_pool_t* root)
{
    apr_pool_create(&pool, root);
}

void OpenStatement()
{
    ht = apr_hash_make(pool);
}

void CloseStatement()
{
    apr_pool_destroy(pool);
}

void RegisterIdentifier(pANTLR3_UINT8 identifier)
{
    StatementContext* context = (StatementContext*)apr_pcalloc(pool, sizeof(StatementContext));;
    apr_hash_set(ht, (const char*)identifier, APR_HASH_KEY_STRING, context);
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