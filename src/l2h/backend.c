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
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include <apr_strings.h>
#include <lib.h>
#include "backend.h"
#include "processor.h"

/*
    bend_ - public members
    prbend_ - private members
*/

#define STACK_INIT_SZ 32

static op_value_t* prbend_create_string(fend_node_t* node, apr_pool_t* pool);
static op_value_t* prbend_create_number(fend_node_t* node, apr_pool_t* pool);
static void prbend_create_triple(fend_node_t* t, apr_pool_t* pool);

// Processors
static triple_t* prbend_create_from_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_not_rel_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_and_rel_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_or_rel_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_relation_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_internal_type_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_string_literal_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_numeric_literal_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_identifier_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_property_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_let_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_select_triple(fend_node_t* node, apr_pool_t* pool);
static triple_t* prbend_create_method_call_triple(fend_node_t* node, apr_pool_t* pool);

static char* bend_orderings[] = {
    "asc",
    "desc"
};

static triple_t* (*bend_processors[])(fend_node_t*, apr_pool_t*) = {
    NULL, // node_type_query
    &prbend_create_from_triple, // node_type_from
    NULL, // node_type_where
    &prbend_create_not_rel_triple, // node_type_not_rel
    &prbend_create_and_rel_triple, // node_type_and_rel
    &prbend_create_or_rel_triple, // node_type_or_rel
    &prbend_create_relation_triple, // node_type_relation
    &prbend_create_internal_type_triple, // node_type_internal_type
    &prbend_create_string_literal_triple, // node_type_string_literal
    &prbend_create_numeric_literal_triple, // node_type_numeric_literal
    &prbend_create_identifier_triple, // node_type_identifier
    &prbend_create_property_triple, // node_type_property
    NULL, // node_type_unary_expression
    NULL, // node_type_enum
    NULL, // node_type_group
    &prbend_create_let_triple, // node_type_let
    NULL, // node_type_query_body
    NULL, // node_type_query_continuations
    &prbend_create_select_triple, // node_type_select
    &prbend_create_method_call_triple, // node_type_method_call
    NULL, // node_type_join
    NULL, // node_type_on
    NULL, // node_type_in
    NULL, // node_type_order_by
    NULL // node_type_ordering
};

static apr_array_header_t* bend_instructions;
static apr_pool_t* bend_pool = NULL;

void bend_init(apr_pool_t* pool) {
    bend_pool = pool;
    bend_instructions = apr_array_make(bend_pool, STACK_INIT_SZ, sizeof(triple_t*));
    proc_init(bend_pool);
}

void bend_complete() {
    proc_run(bend_instructions);
    proc_complete();
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
    prbend_create_triple(node, pool);
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
            type = apr_psprintf(pool, "rel(%s)", proc_get_cond_op_name(node->value.relation_op));
            break;
        case node_type_internal_type:
            type = apr_psprintf(pool, "type(%s)", proc_get_type_name(node->value.type));
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

triple_t* prbend_create_from_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    int instructions_count = bend_instructions->nelts;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_from;
    instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
    instruction->op1->number = instructions_count - 2;
    instruction->op2 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
    instruction->op2->number = instructions_count - 1;
    return instruction;
}

triple_t* prbend_create_not_rel_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_not_rel;
    return instruction;
}

triple_t* prbend_create_and_rel_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_and_rel;
    return instruction;
}

triple_t* prbend_create_or_rel_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_or_rel;
    return instruction;
}

triple_t* prbend_create_relation_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_relation;
    instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
    instruction->op1->relation_op = node->value.relation_op;
    return instruction;
}

triple_t* prbend_create_internal_type_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    triple_t* prev;

    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_type;

    if (node->value.type == type_def_user) {
        // remove user type identifier
        prev = *(triple_t * *)apr_array_pop(bend_instructions);
        instruction->op1 = prev->op2;
        // remove dynamic from instructions
        *(triple_t * *)apr_array_pop(bend_instructions);
    }
    else {
        instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
        instruction->op1->type = node->value.type;
    }
    return instruction;
}

triple_t* prbend_create_string_literal_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_string;
    instruction->op1 = prbend_create_string(node, pool);
    return instruction;
}

triple_t* prbend_create_numeric_literal_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_integer;
    instruction->op1 = prbend_create_number(node, pool);
    return instruction;
}

// identifier either definition or usage (method or property call)
triple_t* prbend_create_identifier_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* prev = NULL;
    
    triple_t* instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));

    int instructions_count = bend_instructions->nelts;

    if (instructions_count > 0) {
        prev = ((triple_t * *)bend_instructions->elts)[instructions_count - 1];
        if (prev->code == opcode_type) {
            prev = *(triple_t * *)apr_array_pop(bend_instructions);
            instruction->code = opcode_def;
            instruction->op1 = prev->op1;
        }
        else if (prev->code == opcode_select) {
            instruction->code = opcode_into;
            instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
            instruction->op1->number = instructions_count - 1;
        }
        else {
            instruction->code = opcode_usage;
        }


    }
    else {
        instruction->code = opcode_type;
        instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
        instruction->op1->type = type_def_dynamic;
        *(triple_t * *)apr_array_push(bend_instructions) = instruction;

        instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
        instruction->code = opcode_def;
        instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
        instruction->op1->number = 0;
    }
    instruction->op2 = prbend_create_string(node, pool);
    return instruction;
}

triple_t* prbend_create_property_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    triple_t* prev = *(triple_t * *)apr_array_pop(bend_instructions);

    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_property;
    instruction->op1 = prev->op2;
    instruction->op2 = prbend_create_string(node, pool);
    return instruction;
}

triple_t* prbend_create_let_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_let;
    return instruction;
}

triple_t* prbend_create_select_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_select;
    return instruction;
}

triple_t* prbend_create_method_call_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* instruction = NULL;
    triple_t* prev = NULL;

    instruction = (triple_t*)apr_pcalloc(pool, sizeof(triple_t));
    instruction->code = opcode_call;

    // parameterless method
    if (node->left == NULL && node->right == NULL) {
        prev = *(triple_t * *)apr_array_pop(bend_instructions);
        instruction->op1 = prev->op2;
    }
    else {
        instruction->op1 = (op_value_t*)apr_pcalloc(pool, sizeof(op_value_t));
        instruction->op1->string = "";
    }
    instruction->op2 = prbend_create_string(node, pool);
    return instruction;
}

void prbend_create_triple(fend_node_t* node, apr_pool_t* pool) {
    triple_t* (*processor)(fend_node_t*, apr_pool_t*) = bend_processors[node->type];
    triple_t* instruction = NULL;
    if (processor != NULL) {
        instruction = processor(node, pool);
    }

    if(instruction != NULL) {
        *(triple_t**)apr_array_push(bend_instructions) = instruction;
    }
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