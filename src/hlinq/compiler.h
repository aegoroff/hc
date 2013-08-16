/*!
 * \brief   The file contains HLINQ compiler API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#ifndef COMPILER_HCALC_H_
#define COMPILER_HCALC_H_

#include    <antlr3.h>
#include    <tomcrypt.h>
#include "apr.h"
#include "apr_pools.h"
#include "apr_strings.h"

#include "..\srclib\lib.h"
#include "..\srclib\bf.h"
#include "..\srclib\traverse.h"
#include "hashes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ProgramOptions {
    BOOL PrintCalcTime;
	BOOL PrintLowCase;
	BOOL PrintSfv;
    BOOL OnlyValidate;
} ProgramOptions;

typedef enum CondOp {
    CondOpUndefined = -1,
    CondOpEq,
    CondOpNotEq,
    CondOpMatch,
    CondOpNotMatch,
    CondOpGe,
    CondOpLe,
    CondOpGeEq,
    CondOpLeEq,
    CondOpOr,
    CondOpAnd,
    CondOpNot,
} CondOp;

typedef enum Attr {
    AttrUndefined = -1,
    AttrName,
    AttrPath,
    AttrDict,
    AttrSize,
    AttrLimit,
    AttrOffset,
    AttrMin,
    AttrMax,
    AttrHash
} Attr;

typedef enum CtxType {
    CtxTypeUndefined = -1,
    CtxTypeFile,
    CtxTypeString,
    CtxTypeDir,
    CtxTypeHash
} CtxType;

typedef struct BoolOperation {
    const char*   Value;
    Attr          Attribute;
    const char*   AttributeName;
    CondOp        Operation;
    void*         Token;
    int           Weight;
} BoolOperation;

typedef struct FileCtx {
    apr_finfo_t* Info;
    const char*  Dir;
    void        (* PfnOutput)(OutputContext* ctx);
} FileCtx;

typedef struct StatementCtx {
    const char* Id;
    const char* Source;
    CtxType     Type;
    HashDefinition* HashAlgorithm;
} StatementCtx;

typedef struct StringStatementContext {
    BOOL        BruteForce;
    int         Min;
    int         Max;
    const char* Dictionary;
} StringStatementContext;

typedef struct DirStatementContext {
    const char* HashToSearch;
    const char* NameFilter;
    CondOp      Operation;
    BOOL        FindFiles;
    BOOL        Recursively;
    apr_off_t   Limit;
    apr_off_t   Offset;
    const char* ExcludePattern;
    const char* IncludePattern;
} DirStatementContext;

void        InitProgram(ProgramOptions* po, const char* fileParam, apr_pool_t* root);
void        OpenStatement(pANTLR3_RECOGNIZER_SHARED_STATE state);
void        CloseStatement(void);
void        DefineQueryType(CtxType type);
void        RegisterIdentifier(pANTLR3_UINT8 identifier);
void        RegisterVariable(pANTLR3_UINT8 var, pANTLR3_UINT8 value);
BOOL        CallAttiribute(pANTLR3_UINT8 identifier, void* token);

const char* GetValue(pANTLR3_UINT8 variable, void* token);
void        SetSource(pANTLR3_UINT8 str, void* token);

void AssignAttribute(Attr code, pANTLR3_UINT8 value, void* valueToken, pANTLR3_UINT8 attrubute);
void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode, void* token, pANTLR3_UINT8 attrubute);
void WhereClauseCond(CondOp opcode, void* token);

void SetHashAlgorithmIntoContext(pANTLR3_UINT8 str);
void SetRecursively();
void SetFindFiles();
void SetBruteForce();

DirStatementContext*    GetDirContext();
StringStatementContext* GetStringContext();

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_
