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
#include "apr_hash.h"
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
} DirStatementContext;

void        InitProgram(ProgramOptions* po, const char* fileParam, apr_pool_t* root);
void        OpenStatement(pANTLR3_RECOGNIZER_SHARED_STATE state);
void        CloseStatement(void);
void        DefineQueryType(CtxType type);
void        RegisterIdentifier(pANTLR3_UINT8 identifier);
void        RegisterVariable(pANTLR3_UINT8 var, pANTLR3_UINT8 value);
BOOL        CallAttiribute(pANTLR3_UINT8 identifier, void* token);
const char* Trim(pANTLR3_UINT8 str);
const char* GetValue(pANTLR3_UINT8 variable, void* token);
void        SetSource(pANTLR3_UINT8 str, void* token);

void AssignAttribute(Attr code, pANTLR3_UINT8 value, void* valueToken, pANTLR3_UINT8 attrubute);
void WhereClauseCall(Attr code, pANTLR3_UINT8 value, CondOp opcode, void* token, pANTLR3_UINT8 attrubute);
void WhereClauseCond(CondOp opcode, void* token);

void                    SetHashAlgorithmIntoContext(pANTLR3_UINT8 str);
void                    SetRecursively();
void                    SetFindFiles();
void                    SetBruteForce();
void*                   GetContext();
DirStatementContext*    GetDirContext();
StringStatementContext* GetStringContext();

void         RunString(DataContext* dataCtx);
void         RunDir(DataContext* dataCtx);
void         RunFile(DataContext* dataCtx);
void         RunHash();
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);
BOOL         FilterFiles(apr_finfo_t* info, const char* dir, TraverseContext* ctx, apr_pool_t* p);
apr_status_t FindFile(const char* fullPathToFile, DataContext* ctx, apr_pool_t* p);

BOOL SetMin(const char* value, const char* attr);
BOOL SetMax(const char* value, const char* attr);
BOOL SetLimit(const char* value, const char* attr);
BOOL SetOffset(const char* value, const char* attr);
BOOL SetDictionary(const char* value, const char* attr);
BOOL SetName(const char* value, const char* attr);
BOOL SetHashToSearch(const char* value, const char* attr);


BOOL CompareName(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareSize(BoolOperation* op, void* context, apr_pool_t* p);
BOOL ComparePath(BoolOperation* op, void* context, apr_pool_t* p);

BOOL CompareStr(const char* value, CondOp operation, const char* str, apr_pool_t* p);
BOOL CompareInt(apr_off_t value, CondOp operation, const char* integer);

BOOL Compare(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareLimit(BoolOperation* op, void* context, apr_pool_t* p);
BOOL CompareOffset(BoolOperation* op, void* context, apr_pool_t* p);

void* FileAlloc(size_t size);

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_
