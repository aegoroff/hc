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
#include "..\srclib\lib.h"
#include "..\srclib\bf.h"
#include "..\srclib\traverse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CondOp
{
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

typedef enum Attr
{
    AttrUndefined = -1,
    AttrName,
    AttrPath,
    AttrDict,
    AttrMd5,
    AttrSha1,
    AttrSha256,
    AttrSha384,
    AttrSha512,
    AttrMd4,
    AttrCrc32,
    AttrWhirlpool,
    AttrSize,
    AttrLimit,
    AttrOffset,
    AttrMin,
    AttrMax
} Attr;

typedef enum Alg
{
    AlgUndefined = -1,
    AlgMd5,
    AlgSha1,
    AlgMd4,
    AlgSha256,
    AlgSha384,
    AlgSha512,
    AlgWhirlpool,
    AlgCrc32
} Alg;

typedef enum CtxType
{
    CtxTypeUndefined = -1,
    CtxTypeFile,
    CtxTypeString,
    CtxTypeDir,
    CtxTypeHash
} CtxType;

typedef struct BoolOperation {
    Attr Attribute;
    const char* Value;
    CondOp Operation;
} BoolOperation;

typedef struct StatementCtx {
    const char* Id;
    const char* Source;
    Alg HashAlgorithm;
    apr_size_t HashLength;
    CtxType Type;
} StatementCtx;

typedef struct StringStatementContext {
    BOOL BruteForce;
    int Min;
    int Max;
    const char* Dictionary;
} StringStatementContext;

typedef struct DirStatementContext {
    const char* HashToSearch;
    const char* NameFilter;
    BOOL Recursively;
    apr_off_t Limit;
    apr_off_t Offset;
} DirStatementContext;

void InitProgram(BOOL onlyValidate, apr_pool_t* root);
void OpenStatement();
void CloseStatement(BOOL isPrintCalcTime);
void DefineQueryType(CtxType type);
void RegisterIdentifier(pANTLR3_UINT8 identifier);
BOOL CallAttiribute(pANTLR3_UINT8 identifier);
char* Trim(pANTLR3_UINT8 str);
void SetSource(pANTLR3_UINT8 str);

void AssignAttribute(Attr code, pANTLR3_UINT8 value);
void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode);
void WhereClauseCond(CondOp opcode);

void SetHashAlgorithm(Alg algorithm);
void SetRecursively();
void SetBruteForce();
void* GetContext();
DirStatementContext* GetDirContext();
StringStatementContext* GetStringContext();

void RunString(DataContext* dataCtx);
void RunDir(DataContext* dataCtx);
void RunHash();
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);
BOOL FilterFiles(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* pool);

void SetMin(const char* value);
void SetMax(const char* value);
void SetLimit(const char* value);
void SetOffset(const char* value);
void SetDictionary(const char* value);
void SetName(const char* value);

void SetHashToSearch(const char* value, Alg algorithm);
void SetMd5ToSearch(const char* value);
void SetSha1ToSearch(const char* value);
void SetSha256ToSearch(const char* value);
void SetSha384ToSearch(const char* value);
void SetSha512ToSearch(const char* value);
void SetShaMd4ToSearch(const char* value);
void SetShaCrc32ToSearch(const char* value);
void SetShaWhirlpoolToSearch(const char* value);

BOOL CompareName(const char* value, CondOp operation, void* context);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax);

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_