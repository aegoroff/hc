/*!
 * \brief   The file contains HLINQ compiler API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#ifndef COMPILER_HCALC_H_
#define COMPILER_HCALC_H_

#include    <antlr3.h>
#include "apr.h"
#include "apr_pools.h"
#include "apr_strings.h"
#include "apr_hash.h"

typedef struct StatementContext {
    const char* SearchRoot;
    const char* ActionTarget;
} StatementContext;

void InitProgram(apr_pool_t* root);
void OpenStatement();
void CloseStatement(const char* identifier);
void CreateStatementContext(const char* identifier);
void CallAttiribute(pANTLR3_UINT8 identifier);
void SetSearchRoot(pANTLR3_UINT8 str, const char* identifier);
void SetActionTarget(pANTLR3_UINT8 str, const char* identifier);
char* Trim(pANTLR3_UINT8 str);

#endif // COMPILER_HCALC_H_