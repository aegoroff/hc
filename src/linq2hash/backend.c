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


static char* bend_cond_op_names[] = {
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

static char* bend_type_names[] = {
    "dynamic",
    "file",
    "dir",
    "string",
    "user"
};

static char* bend_orderings[] = {
    "asc",
    "desc"
};

apr_array_header_t* bend_emit_stack;
apr_pool_t* bend_pool = NULL;
pcre2_general_context* pcre_context = NULL;

void* pcre_alloc(size_t size, void* memory_data) {
    return apr_palloc(bend_pool, size);
}

void  pcre_free(void * p1, void * p2) {
    
}

void bend_init(apr_pool_t* pool) {
    bend_pool = pool;
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
    bend_emit_stack = apr_array_make(bend_pool, STACK_INIT_SZ, sizeof(char*));
}

void bend_cleanup() {
    pcre2_general_context_free(pcre_context);
}

void bend_print_label(fend_node_t* node, apr_pool_t* pool) {
    char* type = bend_create_label(node, pool);
    printf("%s\n", type);
}

void bend_emit(fend_node_t* node, apr_pool_t* pool) {
    switch(node->type) {
        case NodeTypeQuery:
        case NodeTypeUnaryExpression:
        case NodeTypeQueryBody:
        case NodeTypeEnum:
        case NodeTypeJoin:
            return;
    }
    char* statement = bend_create_label(node, pool);
    if(statement != NULL) {
        printf("%s\n", statement);
    }
}

char* bend_create_label(fend_node_t* node, apr_pool_t* pool) {
    char* type = NULL;

    switch(node->type) {
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
            type = apr_psprintf(pool, "rel(%s)", bend_cond_op_names[node->value.RelationOp]);
            break;
        case NodeTypeInternalType:
            type = apr_psprintf(pool, "type(%s)", bend_type_names[node->value.Type]);
            break;
        case NodeTypeStringLiteral:
            type = apr_psprintf(pool, "str(%s)", node->value.String);
            break;
        case NodeTypeNumericLiteral:
            type = apr_psprintf(pool, "num(%d)", node->value.Number);
            break;
        case NodeTypeIdentifier:
            type = apr_psprintf(pool, "id(%s)", node->value.String);
            break;
        case NodeTypeProperty:
            type = apr_psprintf(pool, "prop(%s)", node->value.String);
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
            type = apr_psprintf(pool, "order(%s)", bend_orderings[node->value.Ordering]);
            break;
        case NodeTypeMethodCall:
            type = apr_psprintf(pool, "method(%s)", node->value.String);
            break;
    }
    return type;
}

BOOL bend_match_re(const char* pattern, const char* subject) {
    int errornumber = 0;
    size_t erroroffset = 0;

    pcre2_code* re = pcre2_compile(
        pattern,       /* the pattern */
        PCRE2_ZERO_TERMINATED, /* indicates pattern is zero-terminated */
        0,                     /* default options */
        &errornumber,          /* for error number */
        &erroroffset,          /* for error offset */
        NULL);                 /* use default compile context */

    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        return FALSE;
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);

    int flags = PCRE2_NOTEMPTY;
    if (!strstr(subject, "^")) {
        flags |= PCRE2_NOTBOL;
    }
    if (!strstr(subject, "$")) {
        flags |= PCRE2_NOTEOL;
    }

    int rc = pcre2_match(
        re,                   /* the compiled pattern */
        subject,              /* the subject string */
        strlen(subject),       /* the length of the subject */
        0,                    /* start at offset 0 in the subject */
        flags,
        match_data,           /* block for storing the result */
        NULL);                /* use default match context */
    return rc >= 0;
}