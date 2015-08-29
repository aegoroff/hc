/*!
 * \brief   The file contains compiler frontend implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#include <stdio.h>
#include <stdlib.h>
#include "frontend.h"
#include "apr.h"
#include "apr_strings.h"
#include "apr_hash.h"

apr_pool_t* fend_pool = NULL;
apr_pool_t* fend_translation_unit_pool = NULL;
apr_pool_t* fend_query_pool = NULL;
apr_hash_t* fend_query_identifiers = NULL;
void (*fend_callback)(fend_node_t* ast) = NULL;

void fend_init(apr_pool_t* pool) {
    fend_pool = pool;
}

void fend_query_init() {
    apr_pool_create(&fend_query_pool, fend_translation_unit_pool);
    fend_query_identifiers = apr_hash_make(fend_query_pool);
}

void fend_translation_unit_init(void (*onQueryComplete)(fend_node_t* ast)) {
    apr_pool_create(&fend_translation_unit_pool, fend_pool);
    fend_callback = onQueryComplete;
}

fend_node_t* CreateNode(fend_node_t* left, fend_node_t* right, NodeType_t type) {
    fend_node_t* node = (fend_node_t*)apr_pcalloc(fend_query_pool, sizeof(fend_node_t));
    node->type = type;
    node->left = left;
    node->right = right;
    return node;
}

fend_node_t* create_string_node(fend_node_t* left, fend_node_t* right, NodeType_t type, char* value) {
    fend_node_t* node = CreateNode(left, right, type);
    node->value.String = value;
    return node;
}

fend_node_t* create_number_node(fend_node_t* left, fend_node_t* right, NodeType_t type, long long value) {
    fend_node_t* node = CreateNode(left, right, type);
    node->value.Number = value;
    return node;
}

fend_node_t* fend_query_complete(fend_node_t* from, fend_node_t* body) {
    return CreateNode(from, body, NodeTypeQuery);
}

void fend_query_cleanup(fend_node_t* result) {
    fend_callback(result);
    apr_pool_destroy(fend_query_pool);
    fend_query_identifiers = NULL;
}

void fend_translation_unit_cleanup() {
    apr_pool_destroy(fend_translation_unit_pool);
}

char* fend_translation_unit_strdup(char* str) {
    return apr_pstrdup(fend_translation_unit_pool, str);
}

char* fend_query_strdup(char* str) {
    return apr_pstrdup(fend_query_pool, str);
}

long long fend_to_number(char* str) {
    apr_off_t result = 0;
    apr_strtoff(&result, str, NULL, 0);
    return result;
}

TypeInfo_t* fend_on_simple_type_def(TypeDef_t type) {
    TypeInfo_t* result = (TypeInfo_t*)apr_pcalloc(fend_query_pool, sizeof(TypeInfo_t));
    result->Type = type;
    return result;
}

TypeInfo_t* fend_on_complex_type_def(TypeDef_t type, char* info) {
    TypeInfo_t* result = fend_on_simple_type_def(type);
    result->Info = fend_query_strdup(info);
    return result;
}

fend_node_t* fend_on_identifier_declaration(TypeInfo_t* type, fend_node_t* identifier) {
    apr_hash_set(fend_query_identifiers, identifier->value.String, APR_HASH_KEY_STRING, type);
    identifier->left = fend_on_type_attribute(type);
    return identifier;
}

fend_node_t* fend_on_unary_expression(UnaryExpType_t type, void* leftValue, void* rightValue) {
    fend_node_t* expr = CreateNode(NULL, NULL, NodeTypeUnaryExpression);
    switch(type) {
        case UnaryExpTypeIdentifier:
            expr->left = leftValue;
            break;
        case UnaryExpTypeString:
            expr->left = create_string_node(NULL, NULL, NodeTypeStringLiteral, leftValue);
            break;
        case UnaryExpTypeNumber:
            expr->left = create_number_node(NULL, NULL, NodeTypeNumericLiteral, leftValue);
            break;
        case UnaryExpTypePropertyCall:
        case UnaryExpTypeMehtodCall:
            expr->left = leftValue;
            expr->right = rightValue;
            break;
    }
    return expr;

}

fend_node_t* fend_on_from(fend_node_t* type, fend_node_t* datasource) {
    return CreateNode(type, datasource, NodeTypeFrom);
}

fend_node_t* fend_on_where(fend_node_t* expr) {
    return CreateNode(expr, NULL, NodeTypeWhere);
}

fend_node_t* fend_on_releational_expr(fend_node_t* left, fend_node_t* right, CondOp_t relation) {
    fend_node_t* node = CreateNode(left, right, NodeTypeRelation);
    node->value.RelationOp = relation;
    return node;
}

fend_node_t* fend_on_predicate(fend_node_t* left, fend_node_t* right, NodeType_t type) {
    return CreateNode(left, right, type);
}

fend_node_t* fend_on_enum(fend_node_t* left, fend_node_t* right) {
    return CreateNode(left, right, NodeTypeEnum);
}

fend_node_t* fend_on_group(fend_node_t* left, fend_node_t* right) {
    return CreateNode(left, right, NodeTypeGroup);
}

fend_node_t* fend_on_let(fend_node_t* id, fend_node_t* expr) {
    return CreateNode(id, expr, NodeTypeLet);
}

fend_node_t* fend_on_query_body(fend_node_t* opt_query_body_clauses, fend_node_t* select_or_group_clause, fend_node_t* opt_query_continuation) {
    fend_node_t* select = CreateNode(opt_query_body_clauses, select_or_group_clause, NodeTypeSelect);
    return CreateNode(select, opt_query_continuation, NodeTypeQueryBody);
}

fend_node_t* fend_on_string_attribute(char* str) {
    return create_string_node(NULL, NULL, NodeTypeProperty, str);
}

fend_node_t* fend_on_type_attribute(TypeInfo_t* type) {
    if (type->Info != NULL) {
        fend_node_t* typeNode = CreateNode(NULL, NULL, NodeTypeInternalType);
        typeNode->value.Type = type->Type;
        return create_string_node(typeNode, NULL, NodeTypeIdentifier, type->Info);
    }
    else {
        fend_node_t* typeNode = CreateNode(NULL, NULL, NodeTypeInternalType);
        typeNode->value.Type = type->Type;
        return typeNode;
    }
}

fend_node_t* fend_on_continuation(fend_node_t* id, fend_node_t* query_body) {
    return CreateNode(id, query_body, NodeTypeQueryContinuation);
}

fend_node_t* fend_on_method_call(char* method, fend_node_t* arguments) {
    return create_string_node(arguments, NULL, NodeTypeMethodCall, method);
}

fend_node_t* fend_on_identifier(char* id) {
    return create_string_node(NULL, NULL, NodeTypeIdentifier, id);
}

fend_node_t* fend_on_join(fend_node_t* identifier, fend_node_t* in, fend_node_t* onFirst, fend_node_t* onSecond) {
    fend_node_t* onNode = CreateNode(onFirst, onSecond, NodeTypeOn);
    fend_node_t* inNode = CreateNode(in, onNode, NodeTypeIn);
    return CreateNode(identifier, inNode, NodeTypeJoin);
}

fend_node_t* fend_on_order_by(fend_node_t* ordering) {
    return CreateNode(ordering, NULL, NodeTypeOrderBy);
}

fend_node_t* fend_on_ordering(fend_node_t* ordering, Ordering_t direction) {
    fend_node_t* node = CreateNode(ordering, NULL, NodeTypeOrdering);
    node->value.Ordering = direction;
    return node;
}
