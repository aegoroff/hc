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

apr_pool_t* rootPool = NULL;
apr_pool_t* translationUnitPool = NULL;
apr_pool_t* queryPool = NULL;
apr_hash_t* htQueryIdentifiers = NULL;
void (*callback)(Node_t* ast) = NULL;

void FrontendInit(apr_pool_t* pool) {
    rootPool = pool;
}

void QueryInit() {
    apr_pool_create(&queryPool, translationUnitPool);
    htQueryIdentifiers = apr_hash_make(queryPool);
}

void TranslationUnitInit(void (*onQueryComplete)(Node_t* ast)) {
    apr_pool_create(&translationUnitPool, rootPool);
    callback = onQueryComplete;
}

Node_t* CreateNode(Node_t* left, Node_t* right, NodeType_t type) {
    Node_t* node = (Node_t*)apr_pcalloc(queryPool, sizeof(Node_t));
    node->Type = type;
    node->Left = left;
    node->Right = right;
    return node;
}

Node_t* CreateStringNode(Node_t* left, Node_t* right, NodeType_t type, char* value) {
    Node_t* node = CreateNode(left, right, type);
    node->Value.String = value;
    return node;
}

Node_t* CreateNumberNode(Node_t* left, Node_t* right, NodeType_t type, long long value) {
    Node_t* node = CreateNode(left, right, type);
    node->Value.Number = value;
    return node;
}

Node_t* QueryComplete(Node_t* from, Node_t* body) {
    return CreateNode(from, body, NodeTypeQuery);
}

void QueryCleanup(Node_t* result) {
    callback(result);
    apr_pool_destroy(queryPool);
    htQueryIdentifiers = NULL;
}

void TranslationUnitCleanup() {
    apr_pool_destroy(translationUnitPool);
}

char* TranslationUnitStrdup(char* str) {
    return apr_pstrdup(translationUnitPool, str);
}

char* QueryStrdup(char* str) {
    return apr_pstrdup(queryPool, str);
}

long long ToNumber(char* str) {
    apr_off_t result = 0;
    apr_strtoff(&result, str, NULL, 0);
    return result;
}

TypeInfo_t* OnSimpleTypeDef(TypeDef_t type) {
    TypeInfo_t* result = (TypeInfo_t*)apr_pcalloc(queryPool, sizeof(TypeInfo_t));
    result->Type = type;
    return result;
}

TypeInfo_t* OnComplexTypeDef(TypeDef_t type, char* info) {
    TypeInfo_t* result = OnSimpleTypeDef(type);
    result->Info = QueryStrdup(info);
    return result;
}

Node_t* OnIdentifierDeclaration(TypeInfo_t* type, Node_t* identifier) {
    apr_hash_set(htQueryIdentifiers, identifier->Value.String, APR_HASH_KEY_STRING, type);
    identifier->Left = OnTypeAttribute(type);
    return identifier;
}

Node_t* OnUnaryExpression(UnaryExpType_t type, void* leftValue, void* rightValue) {
    Node_t* expr = CreateNode(NULL, NULL, NodeTypeUnaryExpression);
    switch(type) {
        case UnaryExpTypeIdentifier:
            expr->Left = leftValue;
            break;
        case UnaryExpTypeString:
            expr->Left = CreateStringNode(NULL, NULL, NodeTypeStringLiteral, leftValue);
            break;
        case UnaryExpTypeNumber:
            expr->Left = CreateNumberNode(NULL, NULL, NodeTypeNumericLiteral, leftValue);
            break;
        case UnaryExpTypePropertyCall:
        case UnaryExpTypeMehtodCall:
            expr->Left = leftValue;
            expr->Right = rightValue;
            break;
    }
    return expr;

}

Node_t* OnFrom(Node_t* type, Node_t* datasource) {
    return CreateNode(type, datasource, NodeTypeFrom);
}

Node_t* OnWhere(Node_t* expr) {
    return CreateNode(expr, NULL, NodeTypeWhere);
}

Node_t* OnReleationalExpr(Node_t* left, Node_t* right, CondOp_t relation) {
    Node_t* node = CreateNode(left, right, NodeTypeRelation);
    node->Value.RelationOp = relation;
    return node;
}

Node_t* OnPredicate(Node_t* left, Node_t* right, NodeType_t type) {
    return CreateNode(left, right, type);
}

Node_t* OnEnum(Node_t* left, Node_t* right) {
    return CreateNode(left, right, NodeTypeEnum);
}

Node_t* OnGroup(Node_t* left, Node_t* right) {
    return CreateNode(left, right, NodeTypeGroup);
}

Node_t* OnLet(Node_t* id, Node_t* expr) {
    return CreateNode(id, expr, NodeTypeLet);
}

Node_t* OnQueryBody(Node_t* opt_query_body_clauses, Node_t* select_or_group_clause, Node_t* opt_query_continuation) {
    Node_t* select = CreateNode(opt_query_body_clauses, select_or_group_clause, NodeTypeSelect);
    return CreateNode(select, opt_query_continuation, NodeTypeQueryBody);
}

Node_t* OnStringAttribute(char* str) {
    return CreateStringNode(NULL, NULL, NodeTypeProperty, str);
}

Node_t* OnTypeAttribute(TypeInfo_t* type) {
    if(type->Info != NULL) {
        Node_t* typeNode = CreateNode(NULL, NULL, NodeTypeInternalType);
        typeNode->Value.Type = type->Type;
        return CreateStringNode(typeNode, NULL, NodeTypeIdentifier, type->Info);
    }
    else {
        Node_t* typeNode = CreateNode(NULL, NULL, NodeTypeInternalType);
        typeNode->Value.Type = type->Type;
        return typeNode;
    }
}

Node_t* OnContinuation(Node_t* id, Node_t* query_body) {
    return CreateNode(id, query_body, NodeTypeQueryContinuation);
}

Node_t* OnMethodCall(char* method, Node_t* arguments) {
    return CreateStringNode(arguments, NULL, NodeTypeMethodCall, method);
}

Node_t* OnIdentifier(char* id) {
    return CreateStringNode(NULL, NULL, NodeTypeIdentifier, id);
}

Node_t* OnJoin(Node_t* identifier, Node_t* in, Node_t* onFirst, Node_t* onSecond) {
    Node_t* onNode = CreateNode(onFirst, onSecond, NodeTypeOn);
    Node_t* inNode = CreateNode(in, onNode, NodeTypeIn);
    return CreateNode(identifier, inNode, NodeTypeJoin);
}

Node_t* OnOrderBy(Node_t* ordering) {
    return CreateNode(ordering, NULL, NodeTypeOrderBy);
}

Node_t* OnOrdering(Node_t* ordering, Ordering_t direction) {
    Node_t* node = CreateNode(ordering, NULL, NodeTypeOrdering);
    node->Value.Ordering = direction;
    return node;
}
