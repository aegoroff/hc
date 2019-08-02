/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains compiler frontend interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#ifndef LINQ2HASH_FRONTEND_H_
#define LINQ2HASH_FRONTEND_H_

#include "apr_pools.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cond_op_t {
    cond_op_undefined = -1,
    cond_op_eq,
    cond_op_not_eq,
    cond_op_match,
    cond_op_not_match,
    cond_op_ge,
    cond_op_le,
    cond_op_ge_eq,
    cond_op_le_eq,
    cond_op_or,
    cond_op_and,
    cond_op_not,
} cond_op_t;

typedef enum type_def_t {
    type_def_dynamic, // needs to be derived
    type_def_file,
    type_def_dir,
    type_def_string,
    type_def_user,
} type_def_t;

typedef struct type_info_t {
    type_def_t type;
    char* info;
} type_info_t;

typedef enum ordering_t {
    ordering_asc,
    ordering_desc
} ordering_t;

// Defines all possible unary expression types
typedef enum unary_exp_type_t {
    unary_exp_type_undefined = -1,
    unary_exp_type_string,
    unary_exp_type_number,
    unary_exp_type_property_call,
    unary_exp_type_identifier,
    unary_exp_type_mehtod_call,
} unary_exp_type_t;

typedef struct unary_expr_descriptor_t {
    unary_exp_type_t type;
    char* info;
} unary_expr_descriptor_t;

typedef union node_value_t {
    type_def_t type;
    long long number;
    char* string;
    cond_op_t relation_op;
    ordering_t ordering;
} node_value_t;

typedef enum node_type_t {
    node_type_query,
    node_type_from,
    node_type_where,
    node_type_not_rel,
    node_type_and_rel,
    node_type_or_rel,
    node_type_relation,
    node_type_internal_type,
    node_type_string_literal,
    node_type_numeric_literal,
    node_type_identifier,
    node_type_property,
    node_type_unary_expression,
    node_type_enum,
    node_type_group,
    node_type_let,
    node_type_query_body,
    node_type_query_continuation,
    node_type_select,
    node_type_method_call,
    node_type_join,
    node_type_on,
    node_type_in,
    node_type_order_by,
    node_type_ordering,
} node_type_t;

typedef struct fend_node_t {
    node_type_t type;
    node_value_t value;
    struct fend_node_t* left;
    struct fend_node_t* right;
} fend_node_t;

void fend_init(apr_pool_t* pool);

void fend_translation_unit_init(void (*pfn_on_query_complete)(fend_node_t* ast));
void fend_translation_unit_cleanup();
char* fend_translation_unit_strdup(char* str);

void fend_query_init();
fend_node_t* fend_query_complete(fend_node_t* from, fend_node_t* body);
char* fend_query_strdup(char* str);
void fend_query_cleanup(fend_node_t* result);

long long fend_to_number(char* str);
type_info_t* fend_on_complex_type_def(type_def_t type, char* info);
type_info_t* fend_on_simple_type_def(type_def_t type);
fend_node_t* fend_on_identifier_declaration(type_info_t* type, fend_node_t* identifier);
fend_node_t* fend_on_unary_expression(unary_exp_type_t type, void* leftValue, void* rightValue);
fend_node_t* fend_on_from(fend_node_t* type, fend_node_t* datasource);
fend_node_t* fend_on_where(fend_node_t* expr);
fend_node_t* fend_on_releational_expr(fend_node_t* left, fend_node_t* right, cond_op_t relation);
fend_node_t* fend_on_predicate(fend_node_t* left, fend_node_t* right, node_type_t type);
fend_node_t* fend_on_enum(fend_node_t* left, fend_node_t* right);
fend_node_t* fend_on_group(fend_node_t* left, fend_node_t* right);
fend_node_t* fend_on_let(fend_node_t* id, fend_node_t* expr);
fend_node_t* fend_on_query_body(fend_node_t* opt_query_body_clauses, fend_node_t* select_or_group_clause, fend_node_t* opt_query_continuation);
fend_node_t* fend_on_string_attribute(char* str);
fend_node_t* fend_on_type_attribute(type_info_t* type);
fend_node_t* fend_on_continuation(fend_node_t* id, fend_node_t* query_body);
fend_node_t* fend_on_method_call(char* method, fend_node_t* arguments);
fend_node_t* fend_on_identifier(char* id);
fend_node_t* fend_on_join(fend_node_t* identifier, fend_node_t* in, fend_node_t* onFirst, fend_node_t* onSecond);
fend_node_t* fend_on_order_by(fend_node_t* ordering);
fend_node_t* fend_on_ordering(fend_node_t* ordering, ordering_t direction);
BOOL fend_is_identifier_defined(fend_node_t* id);
void fend_register_identifier(fend_node_t* id, type_def_t type);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_FRONTEND_H_
