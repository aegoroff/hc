/*!
 * \brief   The file contains compiler frontend interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#ifndef LINQ2HASH_FRONTEND_H_
#define LINQ2HASH_FRONTEND_H_

#include "apr_pools.h"

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
} CondOp_t;

typedef enum TypeDef {
	TypeDefDynamic, // needs to be derived
	TypeDefFile,
	TypeDefDir,
	TypeDefString,
	TypeDefUser,
} TypeDef_t;

typedef struct TypeInfo {
    TypeDef_t Type;
    char* Info;
} TypeInfo_t;

typedef enum Ordering {
	OrderingAsc,
	OrderingDesc
} Ordering_t;

// Defines all possible unary expression types
typedef enum UnaryExpType {
	UnaryExpTypeUndefined = -1,
	UnaryExpTypeString,
	UnaryExpTypeNumber,
	UnaryExpTypePropertyCall,
	UnaryExpTypeIdentifier,
	UnaryExpTypeMehtodCall,
} UnaryExpType_t;

typedef struct UnaryExprDesciptor {
	UnaryExpType_t Type;
	char* Info;
} UnaryExprDesciptor_t;

typedef union NodeValue {
	TypeDef_t Type;
	long long Number;
	char* String;
	CondOp_t RelationOp;
	Ordering_t Ordering;
} NodeValue_t;

typedef enum NodeType {
	NodeTypeQuery,
	NodeTypeFrom,
	NodeTypeWhere,
	NodeTypeNotRel,
	NodeTypeAndRel,
	NodeTypeOrRel,
	NodeTypeRelation,
	NodeTypeInternalType,
	NodeTypeStringLiteral,
	NodeTypeNumericLiteral,
	NodeTypeIdentifier,
	NodeTypeProperty,
	NodeTypeUnaryExpression,
	NodeTypeEnum,
	NodeTypeGroup,
	NodeTypeLet,
	NodeTypeQueryBody,
	NodeTypeQueryContinuation,
	NodeTypeSelect,
	NodeTypeMethodCall,
	NodeTypeJoin,
	NodeTypeOn,
	NodeTypeIn,
	NodeTypeInto,
	NodeTypeOrderBy,
	NodeTypeOrdering,
} NodeType_t;

typedef struct Node {
	NodeType_t Type;
	NodeValue_t Value;
	struct Node* Left;
	struct Node* Right;
} Node_t;


void FrontendInit(apr_pool_t* pool);

void TranslationUnitInit(void (*onQueryComplete)(Node_t* ast));
void TranslationUnitCleanup();
char* TranslationUnitStrdup(char* str);

void QueryInit();
Node_t* QueryComplete(Node_t* from, Node_t* body);
char* QueryStrdup(char* str);
void QueryCleanup(Node_t* result);

long long ToNumber(char* str);
TypeInfo_t* OnComplexTypeDef(TypeDef_t type, char* info);
TypeInfo_t* OnSimpleTypeDef(TypeDef_t type);
Node_t* OnIdentifierDeclaration(TypeInfo_t* type, Node_t* identifier);
Node_t* OnUnaryExpression(UnaryExpType_t type, void* leftValue, void* rightValue);
Node_t* OnFrom(Node_t* type, Node_t* datasource);
Node_t* OnWhere(Node_t* expr);
Node_t* OnReleationalExpr(Node_t* left, Node_t* right, CondOp_t relation);
Node_t* OnPredicate(Node_t* left, Node_t* right, NodeType_t type);
Node_t* OnEnum(Node_t* left, Node_t* right);
Node_t* OnGroup(Node_t* left, Node_t* right);
Node_t* OnLet(Node_t* id, Node_t* expr);
Node_t* OnQueryBody(Node_t* opt_query_body_clauses, Node_t* select_or_group_clause, Node_t* opt_query_continuation);
Node_t* OnStringAttribute(char* str);
Node_t* OnTypeAttribute(TypeInfo_t* type);
Node_t* OnContinuation(Node_t* id, Node_t* query_body);
Node_t* OnMethodCall(char* method, Node_t* arguments);
Node_t* OnIdentifier(char* id);
Node_t* OnJoin(Node_t* identifier, Node_t* in, Node_t* onFirst, Node_t* onSecond);
Node_t* OnOrderBy(Node_t* ordering);
Node_t* OnOrdering(Node_t* ordering, Ordering_t direction);


#endif // LINQ2HASH_FRONTEND_H_

