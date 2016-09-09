/*!
 * \brief   The file contains HLINQ compiler API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef COMPILER_HCALC_H_
#define COMPILER_HCALC_H_

#include <apr.h>
#include <apr_pools.h>

#include "..\srclib\lib.h"
#include "..\srclib\traverse.h"
#include "../linq2hash/hashes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ProgramOptions {
    BOOL PrintCalcTime;
    BOOL PrintLowCase;
    BOOL PrintSfv;
    BOOL PrintVerify;
    BOOL OnlyValidate;
    const char* FileToSave;
    BOOL NoProbe;
    BOOL NoErrorOnFind;
    uint32_t NumOfThreads;
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
    const char* Value;
    Attr Attribute;
    const char* AttributeName;
    CondOp Operation;
    void* Token;
    int Weight;
} BoolOperation;

typedef struct FileCtx {
    apr_finfo_t* Info;
    const char* Dir;
    void (* PfnOutput)(out_context_t* ctx);
} FileCtx;

typedef struct StatementCtx {
    const char* Id;
    const char* Source;
    CtxType Type;
    hash_definition_t* HashAlgorithm;
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
    CondOp Operation;
    BOOL FindFiles;
    BOOL Recursively;
    apr_off_t Limit;
    apr_off_t Offset;
    const char* ExcludePattern;
    const char* IncludePattern;
} DirStatementContext;

void InitProgram(ProgramOptions* po, const char* fileParam, apr_pool_t* root);
void OpenStatement();
void CloseStatement(void);
void DefineQueryType(CtxType type);
void RegisterIdentifier(const char* identifier);
void RegisterVariable(const char* var, const char* value);
BOOL CallAttiribute(const char* identifier, void* token);

void SetSource(const char* str, void* token);

void SetHashAlgorithmIntoContext(const char* str);
void SetRecursively();
void SetFindFiles();
void SetBruteForce();

DirStatementContext* GetDirContext();
StringStatementContext* GetStringContext();

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_
