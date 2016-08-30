/*!
 * \brief   The file contains backend implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#define PCRE2_CODE_UNIT_WIDTH 8

#include <pcre2.h>
#include <apr_tables.h>
#include <apr_strings.h>
#include <lib.h>
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

static char* bend_opcode_names[] = {
    "opcode_from     ",
    "opcode_def      ",
    "opcode_let      ",
    "opcode_select   ",
    "opcode_call     ",
    "opcode_property ",
    "opcode_type     ",
    "opcode_usage    ",
    "opcode_const    ",
    "opcode_and_rel  ",
    "opcode_or_rel   ",
    "opcode_not_rel  ",
    "opcode_relation "
};

static char* bend_orderings[] = {
    "asc",
    "desc"
};

static apr_array_header_t* bend_emit_stack;
static apr_array_header_t* bend_instructions;
static apr_pool_t* bend_pool = NULL;
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
    bend_instructions = apr_array_make(bend_pool, STACK_INIT_SZ, sizeof(triple_t*));
}

void bend_cleanup() {
    int i;
    char* type;

    for (i = 0; i < bend_instructions->nelts; i++) {
        triple_t* triple = ((triple_t**)bend_instructions->elts)[i];
        if(triple->op2 != NULL) {
            type = apr_psprintf(bend_pool, "%d: %s %s, %s", i, bend_opcode_names[triple->code], triple->op1, triple->op2);
        } else {
            type = apr_psprintf(bend_pool, "%d: %s %s", i, bend_opcode_names[triple->code], triple->op1);
        }
        lib_printf("%s\n", type);
    }
    pcre2_general_context_free(pcre_context);
}

void bend_print_label(fend_node_t* node, apr_pool_t* pool) {
    char* type = bend_create_label(node, pool);
    lib_printf("%s\n", type);
}

void bend_emit(fend_node_t* node, apr_pool_t* pool) {
    char* type = NULL;
    switch(node->type) {
        case node_type_query:
        case node_type_unary_expression:
        case node_type_query_body:
        case node_type_enum:
        case node_type_join:
            return;
    }
    bend_create_triple(node, pool);
}

char* bend_create_label(fend_node_t* node, apr_pool_t* pool) {
    char* type = NULL;

    switch(node->type) {
        case node_type_query:
            type = "query";
            break;
        case node_type_from:
            type = "from";
            break;
        case node_type_where:
            type = "where";
            break;
        case node_type_not_rel:
            type = "not";
            break;
        case node_type_and_rel:
            type = "and";
            break;
        case node_type_or_rel:
            type = "or";
            break;
        case node_type_relation:
            type = apr_psprintf(pool, "rel(%s)", bend_cond_op_names[node->value.relation_op]);
            break;
        case node_type_internal_type:
            type = apr_psprintf(pool, "type(%s)", bend_type_names[node->value.type]);
            break;
        case node_type_string_literal:
            type = apr_psprintf(pool, "str(%s)", node->value.string);
            break;
        case node_type_numeric_literal:
            type = apr_psprintf(pool, "num(%d)", node->value.number);
            break;
        case node_type_identifier:
            type = apr_psprintf(pool, "id(%s)", node->value.string);
            break;
        case node_type_property:
            type = apr_psprintf(pool, "prop(%s)", node->value.string);
            break;
        case node_type_unary_expression:
            type = "unary";
            break;
        case node_type_enum:
            type = "enum";
            break;
        case node_type_group:
            type = "grp";
            break;
        case node_type_let:
            type = "let";
            break;
        case node_type_query_body:
            type = "qbody";
            break;
        case node_type_query_continuation:
            type = "into";
            break;
        case node_type_select:
            type = "select";
            break;
        case node_type_join:
            type = "join";
            break;
        case node_type_on:
            type = "on";
            break;
        case node_type_in:
            type = "in";
            break;
        case node_type_order_by:
            type = "OrderBy";
            break;
        case node_type_ordering:
            type = apr_psprintf(pool, "order(%s)", bend_orderings[node->value.ordering]);
            break;
        case node_type_method_call:
            type = apr_psprintf(pool, "method(%s)", node->value.string);
            break;
    }
    return type;
}

void bend_create_triple(fend_node_t* node, apr_pool_t* pool) {
    char* left = NULL;
    char* right = NULL;
    triple_t* instruction = NULL;
    triple_t* prev = NULL;
    int instructions_count = bend_instructions->nelts;

    switch (node->type) {
        case node_type_query:
            break;
        case node_type_from:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_from;
            instruction->op1 = apr_psprintf(pool, "%d", instructions_count - 2);
            instruction->op2 = apr_psprintf(pool, "%d", instructions_count - 1);
            break;
        case node_type_where:
            // TODO: type = "where";
            break;
        case node_type_not_rel:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_not_rel;
            break;
        case node_type_and_rel:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_and_rel;
            break;
        case node_type_or_rel:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_or_rel;
            break;
        case node_type_relation:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_relation;
            instruction->op1 = apr_psprintf(pool, "%s", bend_cond_op_names[node->value.relation_op]);;
            break;
        case node_type_internal_type:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_type;
            instruction->op1 = apr_psprintf(pool, "%s", bend_type_names[node->value.type]);
            break;
        case node_type_string_literal:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_const;
            instruction->op1 = apr_psprintf(pool, "%s", node->value.string);
            break;
        case node_type_numeric_literal:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_const;
            instruction->op1 = apr_psprintf(pool, "%d", node->value.number);
            break;
        case node_type_identifier: // identifier either definition or usage (method or property call)
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            if(instructions_count > 0) {
                prev = ((triple_t**)bend_instructions->elts)[instructions_count - 1];
                if(prev->code == opcode_type) {
                    prev = *(triple_t**)apr_array_pop(bend_instructions);
                    instruction->code = opcode_def;
                    instruction->op1 = prev->op1;
                } /*if (prev->code == opcode_usage) {
                    instruction->code = opcode_property;
                    instruction->op1 = prev->op2;
                }*/ else {
                    instruction->code = opcode_usage;
                }

            } else {
                instruction->code = opcode_type;
                instruction->op1 = "dynamic";
                *(triple_t**)apr_array_push(bend_instructions) = instruction;

                instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
                instruction->code = opcode_def;
                instruction->op1 = "0";
            }
            instruction->op2 = apr_psprintf(pool, "%s", node->value.string);

            break;
        case node_type_property:
            prev = *(triple_t**)apr_array_pop(bend_instructions);

            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_property;
            instruction->op1 = prev->op2;
            instruction->op2 = apr_psprintf(pool, "%s", node->value.string);
            break;
        case node_type_method_call:
            prev = *(triple_t**)apr_array_pop(bend_instructions);

            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_call;
            instruction->op1 = prev->op2;
            instruction->op2 = apr_psprintf(pool, "%s", node->value.string);
            break;
        case node_type_unary_expression:
            // TODO: type = "unary";
            break;
        case node_type_enum:
            // TODO: type = "enum";
            break;
        case node_type_group:
            // TODO: type = "grp";
            break;
        case node_type_let:
            right = *(char**)apr_array_pop(bend_emit_stack);
            if (bend_emit_stack->nelts > 0) {
                left = *(char**)apr_array_pop(bend_emit_stack);
            }
            if (left == NULL) {
                left = "";
            }

            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_let;
            instruction->op1 = left;
            instruction->op2 = right;

            *(char**)apr_array_push(bend_emit_stack) = apr_psprintf(pool, "%s", left);

            break;
        case node_type_query_body:
            // TODO: type = "qbody";
            break;
        case node_type_query_continuation:
            // TODO: type = "into";
            break;
        case node_type_select:
            //right = *(char**)apr_array_pop(bend_emit_stack);

            if (bend_emit_stack->nelts > 0) {
                left = *(char**)apr_array_pop(bend_emit_stack);
            } else {
                left = "";
            }

            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_select;
            instruction->op1 = left;
            instruction->op2 = right;

            *(char**)apr_array_push(bend_emit_stack) = apr_psprintf(pool, "ds: %s", left);
            break;
        case node_type_join:
            // TODO: type = "join";
            break;
        case node_type_on:
            // TODO: type = "on";
            break;
        case node_type_in:
            // TODO: type = "in";
            break;
        case node_type_order_by:
            // TODO: type = "OrderBy";
            break;
        case node_type_ordering:
            // TODO: type = apr_psprintf(pool, "order(%s)", bend_orderings[node->value.ordering]);
            break;
        case node_type_into:
            // TODO: type = "into";
            break;
        default: break;
    }
    if(instruction != NULL) {
        *(triple_t**)apr_array_push(bend_instructions) = instruction;
    }
}

BOOL bend_match_re(const char* pattern, const char* subject) {
    int errornumber = 0;
    size_t erroroffset = 0;

    pcre2_code* re = pcre2_compile(
        (unsigned char*)pattern,       /* the pattern */
        PCRE2_ZERO_TERMINATED, /* indicates pattern is zero-terminated */
        0,                     /* default options */
        &errornumber,          /* for error number */
        &erroroffset,          /* for error offset */
        NULL);                 /* use default compile context */

    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        lib_printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
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
        (unsigned char*)subject,              /* the subject string */
        strlen(subject),       /* the length of the subject */
        0,                    /* start at offset 0 in the subject */
        flags,
        match_data,           /* block for storing the result */
        NULL);                /* use default match context */
    return rc >= 0;
}
