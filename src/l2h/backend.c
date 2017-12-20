/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains backend implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#define PCRE2_CODE_UNIT_WIDTH 8

#include <pcre2.h>
#include <apr_tables.h>
#include <apr_strings.h>
#include <lib.h>
#include "backend.h"

/*
    bend_ - public members
    prbend_ - private members
*/

#define STACK_INIT_SZ 32

static op_value_t* prbend_create_string(fend_node_t* node, apr_pool_t* pool);
static op_value_t* prbend_create_number(fend_node_t* node, apr_pool_t* pool);
static void prbend_print_op(triple_t* triple, int i);
static const char* prbend_to_string(opcode_t code, op_value_t* value, int position);

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

static const char* bend_type_names[] = {
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
    "opcode_integer  ",
    "opcode_string   ",
    "opcode_and_rel  ",
    "opcode_or_rel   ",
    "opcode_not_rel  ",
    "opcode_relation "
};

static char* bend_orderings[] = {
    "asc",
    "desc"
};

static apr_array_header_t* bend_instructions;
static apr_pool_t* bend_pool = NULL;
pcre2_general_context* pcre_context = NULL;

/**
 * \brief PCRE requred function. Allocates memory from apache pool
 * \param size the number of bytes to allocate
 * \param memory_data 
 * \return 
 */
void* pcre_alloc(size_t size, void* memory_data) {
    return apr_palloc(bend_pool, size);
}

/**
 * \brief Frees memory allocated. Requied by PCRE engine. Does nothing because memory released by destroying apache pool
 * \param p1 
 * \param p2 
 */
void pcre_free(void* p1, void* p2) {

}

void bend_init(apr_pool_t* pool) {
    bend_pool = pool;
    pcre_context = pcre2_general_context_create(&pcre_alloc, &pcre_free, NULL);
    bend_instructions = apr_array_make(bend_pool, STACK_INIT_SZ, sizeof(triple_t*));
}

void bend_emit(fend_node_t* node, apr_pool_t* pool) {
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
    triple_t* instruction = NULL;
    triple_t* prev;
    int instructions_count = bend_instructions->nelts;

    switch(node->type) {
        case node_type_query:
            break;
        case node_type_from:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_from;
            instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
            instruction->op1->number = instructions_count - 2;
            instruction->op2 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
            instruction->op2->number = instructions_count - 1;
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
            instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
            instruction->op1->relation_op = node->value.relation_op;
            break;
        case node_type_internal_type:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_type;

            if(node->value.type == type_def_user) {
                // remove user type identifier
                prev = *(triple_t**)apr_array_pop(bend_instructions);
                instruction->op1 = prev->op2;
                // remove dynamic from instructions
                *(triple_t**)apr_array_pop(bend_instructions);
            } else {
                instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
                instruction->op1->type = node->value.type;
            }

            break;
        case node_type_string_literal:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_string;
            instruction->op1 = prbend_create_string(node, pool);
            break;
        case node_type_numeric_literal:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_integer;
            instruction->op1 = prbend_create_number(node, pool);
            break;
        case node_type_identifier: // identifier either definition or usage (method or property call)
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            if(instructions_count > 0) {
                prev = ((triple_t**)bend_instructions->elts)[instructions_count - 1];
                if(prev->code == opcode_type) {
                    prev = *(triple_t**)apr_array_pop(bend_instructions);
                    instruction->code = opcode_def;
                    instruction->op1 = prev->op1;
                } else {
                    instruction->code = opcode_usage;
                }

            } else {
                instruction->code = opcode_type;
                instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
                instruction->op1->type = type_def_dynamic;
                *(triple_t**)apr_array_push(bend_instructions) = instruction;

                instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
                instruction->code = opcode_def;
                instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
                instruction->op1->number = 0;
            }
            instruction->op2 = prbend_create_string(node, pool);
            break;
        case node_type_property:
            prev = *(triple_t**)apr_array_pop(bend_instructions);

            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_property;
            instruction->op1 = prev->op2;
            instruction->op2 = prbend_create_string(node, pool);
            break;
        case node_type_method_call:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_call;

            // parameterless method
            if(node->left == NULL && node->right == NULL) {
                prev = *(triple_t**)apr_array_pop(bend_instructions);
                instruction->op1 = prev->op2;
            } else {
                instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
                instruction->op1->string = "";
            }
            instruction->op2 = prbend_create_string(node, pool);
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
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_let;
            break;
        case node_type_query_body:
            // TODO: type = "qbody";
            break;
        case node_type_query_continuation:
            // TODO: type = "into";
            break;
        case node_type_select:
            instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
            instruction->code = opcode_select;
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
        default:
            break;
    }
    if(instruction != NULL) {
        *(triple_t**)apr_array_push(bend_instructions) = instruction;
    }
}

BOOL bend_match_re(const char* pattern, const char* subject) {
    int errornumber = 0;
    size_t erroroffset = 0;

    pcre2_code* re = pcre2_compile(
                                   (unsigned char*)pattern, /* the pattern */
                                   PCRE2_ZERO_TERMINATED, /* indicates pattern is zero-terminated */
                                   0, /* default options */
                                   &errornumber, /* for error number */
                                   &erroroffset, /* for error offset */
                                   NULL); /* use default compile context */

    if(re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        lib_printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        return FALSE;
    }
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);

    int flags = PCRE2_NOTEMPTY;
    if(!strchr(subject, '^')) {
        flags |= PCRE2_NOTBOL;
    }
    if(!strchr(subject, '$')) {
        flags |= PCRE2_NOTEOL;
    }

    int rc = pcre2_match(
                         re, /* the compiled pattern */
                         (unsigned char*)subject, /* the subject string */
                         strlen(subject), /* the length of the subject */
                         0, /* start at offset 0 in the subject */
                         flags,
                         match_data, /* block for storing the result */
                         NULL); /* use default match context */
    return rc >= 0;
}

op_value_t* prbend_create_string(fend_node_t* node, apr_pool_t* pool) {
    op_value_t* result = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
    result->string = apr_psprintf(pool, "%s", node->value.string);
    return result;
}

op_value_t* prbend_create_number(fend_node_t* node, apr_pool_t* pool) {
    op_value_t* result = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
    result->number = node->value.number;
    return result;
}

void bend_complete() {
    int i;
    for(i = 0; i < bend_instructions->nelts; i++) {
        triple_t* triple = ((triple_t**)bend_instructions->elts)[i];
        prbend_print_op(triple, i);
    }
    pcre2_general_context_free(pcre_context);
}

void prbend_print_op(triple_t* triple, int i) {
    char* type;
    if(triple->op2 != NULL) {
        type = apr_psprintf(bend_pool, "%2d: %s %s, %s", i, bend_opcode_names[triple->code],
                            prbend_to_string(triple->code, triple->op1, 0),
                            prbend_to_string(triple->code, triple->op2, 1));
    } else {
        type = apr_psprintf(bend_pool, "%2d: %s %s", i, bend_opcode_names[triple->code],
                            prbend_to_string(triple->code, triple->op1, 0));
    }
    lib_printf("%s\n", type);
}

const char* prbend_to_string(opcode_t code, op_value_t* value, int position) {
    switch(code) {
        case opcode_integer:
        case opcode_from:
            return apr_psprintf(bend_pool, "%d", value->number);
        case opcode_string:
        case opcode_property:
        case opcode_call:
            return value->string;
        case opcode_usage:
            if(position) {
                return value->string;
            }
            return "";
        case opcode_relation:
            return bend_cond_op_names[value->relation_op];
        case opcode_def:
            // 0
            if(value->type >= type_def_dynamic && value->type <= type_def_user) {
                return bend_type_names[value->type];
            }
            return value->string;
        default:
            return "";
    }
}
