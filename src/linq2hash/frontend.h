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

typedef struct fend_node {
	NodeType_t type;
	NodeValue_t value;
	struct fend_node* left;
	struct fend_node* right;
} fend_node_t;


void fend_init(apr_pool_t* pool);

void fend_translation_unit_init(void (*onQueryComplete)(fend_node_t* ast));
void fend_translation_unit_cleanup();
char* fend_translation_unit_strdup(char* str);

void fend_query_init();
fend_node_t* fend_query_complete(fend_node_t* from, fend_node_t* body);
char* fend_query_strdup(char* str);
void fend_query_cleanup(fend_node_t* result);

long long fend_to_number(char* str);
TypeInfo_t* fend_on_complex_type_def(TypeDef_t type, char* info);
TypeInfo_t* fend_on_simple_type_def(TypeDef_t type);
fend_node_t* fend_on_identifier_declaration(TypeInfo_t* type, fend_node_t* identifier);
fend_node_t* fend_on_unary_expression(UnaryExpType_t type, void* leftValue, void* rightValue);
fend_node_t* fend_on_from(fend_node_t* type, fend_node_t* datasource);
fend_node_t* fend_on_where(fend_node_t* expr);
fend_node_t* fend_on_releational_expr(fend_node_t* left, fend_node_t* right, CondOp_t relation);
fend_node_t* fend_on_predicate(fend_node_t* left, fend_node_t* right, NodeType_t type);
fend_node_t* fend_on_enum(fend_node_t* left, fend_node_t* right);
fend_node_t* fend_on_group(fend_node_t* left, fend_node_t* right);
fend_node_t* fend_on_let(fend_node_t* id, fend_node_t* expr);
fend_node_t* fend_on_query_body(fend_node_t* opt_query_body_clauses, fend_node_t* select_or_group_clause, fend_node_t* opt_query_continuation);
fend_node_t* fend_on_string_attribute(char* str);
fend_node_t* fend_on_type_attribute(TypeInfo_t* type);
fend_node_t* fend_on_continuation(fend_node_t* id, fend_node_t* query_body);
fend_node_t* fend_on_method_call(char* method, fend_node_t* arguments);
fend_node_t* fend_on_identifier(char* id);
fend_node_t* fend_on_join(fend_node_t* identifier, fend_node_t* in, fend_node_t* onFirst, fend_node_t* onSecond);
fend_node_t* fend_on_order_by(fend_node_t* ordering);
fend_node_t* fend_on_ordering(fend_node_t* ordering, Ordering_t direction);


#endif // LINQ2HASH_FRONTEND_H_

