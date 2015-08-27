/*!
 * \brief   The file contains backend implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#define PCRE2_CODE_UNIT_WIDTH 8

#include <pcre2.h>
#include <apr_tables.h>
#include <apr_strings.h>
#include "backend.h"

#define STACK_INIT_SZ 32


static char* condOpNames[] = {
    "==",
    "!=",
    "~",
    "!~",
    ">",
    "<",
    ">=",
    "<=",
    "or",
    "and",
    "not"
};

static char* typeNames[] = {
    "dynamic",
    "file",
    "dir",
    "string",
    "user"
};

static char* orderings[] = {
    "asc",
    "desc"
};

apr_array_header_t* emitStack;
apr_pool_t* backendPool = NULL;
pcre2_general_context* pcre_context = NULL;

void* pcre_alloc(size_t size, void* memory_data) {
    return apr_palloc(backendPool, size);
}

void  pcre_free(void * p1, void * p2) {
    
}

void backend_init(apr_pool_t* pool) {
    backendPool = pool;
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
    emitStack = apr_array_make(backendPool, STACK_INIT_SZ, sizeof(char*));
}

void backend_cleanup() {
    pcre2_general_context_free(pcre_context);
}

void print_label(Node_t* node, apr_pool_t* pool) {
    char* type = create_label(node, pool);
    printf("%s\n", type);
}

void emit(Node_t* node, apr_pool_t* pool) {
    switch(node->Type) {
        case NodeTypeQuery:
        case NodeTypeUnaryExpression:
        case NodeTypeQueryBody:
        case NodeTypeEnum:
        case NodeTypeJoin:
            return;
    }
    char* statement = create_label(node, pool);
    if(statement != NULL) {
        printf("%s\n", statement);
    }
}

char* create_label(Node_t* node, apr_pool_t* pool) {
    char* type = NULL;

    switch(node->Type) {
        case NodeTypeQuery:
            type = "query";
            break;
        case NodeTypeFrom:
            type = "from";
            break;
        case NodeTypeWhere:
            type = "where";
            break;
        case NodeTypeNotRel:
            type = "not";
            break;
        case NodeTypeAndRel:
            type = "and";
            break;
        case NodeTypeOrRel:
            type = "or";
            break;
        case NodeTypeRelation:
            type = apr_psprintf(pool, "rel(%s)", condOpNames[node->Value.RelationOp]);
            break;
        case NodeTypeInternalType:
            type = apr_psprintf(pool, "type(%s)", typeNames[node->Value.Type]);
            break;
        case NodeTypeStringLiteral:
            type = apr_psprintf(pool, "str(%s)", node->Value.String);
            break;
        case NodeTypeNumericLiteral:
            type = apr_psprintf(pool, "num(%d)", node->Value.Number);
            break;
        case NodeTypeIdentifier:
            type = apr_psprintf(pool, "id(%s)", node->Value.String);
            break;
        case NodeTypeProperty:
            type = apr_psprintf(pool, "prop(%s)", node->Value.String);
            break;
        case NodeTypeUnaryExpression:
            type = "unary";
            break;
        case NodeTypeEnum:
            type = "enum";
            break;
        case NodeTypeGroup:
            type = "grp";
            break;
        case NodeTypeLet:
            type = "let";
            break;
        case NodeTypeQueryBody:
            type = "qbody";
            break;
        case NodeTypeQueryContinuation:
            type = "into";
            break;
        case NodeTypeSelect:
            type = "select";
            break;
        case NodeTypeJoin:
            type = "join";
            break;
        case NodeTypeOn:
            type = "on";
            break;
        case NodeTypeIn:
            type = "in";
            break;
        case NodeTypeOrderBy:
            type = "OrderBy";
            break;
        case NodeTypeOrdering:
            type = apr_psprintf(pool, "order(%s)", orderings[node->Value.Ordering]);
            break;
        case NodeTypeMethodCall:
            type = apr_psprintf(pool, "method(%s)", node->Value.String);
            break;
    }
    return type;
}

BOOL match_re(char* pattern, char* subject) {
    pcre2_code *re = NULL;
    int errornumber;
    size_t erroroffset;

    re = pcre2_compile(
        pattern,       /* the pattern */
        PCRE2_ZERO_TERMINATED, /* indicates pattern is zero-terminated */
        0,                     /* default options */
        &errornumber,          /* for error number */
        &erroroffset,          /* for error offset */
        NULL);                 /* use default compile context */

    int rc = pcre2_match(
        re,                   /* the compiled pattern */
        subject,              /* the subject string */
        strlen(subject),       /* the length of the subject */
        0,                    /* start at offset 0 in the subject */
        0,                    /* default options */
        NULL,           /* block for storing the result */
        NULL);                /* use default match context */
    return rc >= 0;
}