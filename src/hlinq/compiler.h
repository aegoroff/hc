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
} StatementContext;

void InitProgram(apr_pool_t* root);
void OpenStatement();
void CloseStatement();
void RegisterIdentifier(pANTLR3_UINT8 identifier);
void CallAttiribute(pANTLR3_UINT8 identifier);

#endif // COMPILER_HCALC_H_