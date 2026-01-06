/*!
 * \brief   The file contains compiler frontend implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include "frontend.h"
#include "apr.h"
#include "apr_hash.h"
#include "apr_strings.h"
#include <stdio.h>
#include <stdlib.h>

/*
    fend_ - public members
    prfend_ - private members
*/

static apr_pool_t *fend_pool = NULL;
static apr_pool_t *fend_translation_unit_pool = NULL;
static apr_pool_t *fend_query_pool = NULL;
static apr_hash_t *fend_query_identifiers = NULL;
void (*fend_callback)(fend_node_t *ast) = NULL;
int fend_error_count = 0;

void fend_init(apr_pool_t *pool) { fend_pool = pool; }

void fend_query_init() {
    apr_pool_create(&fend_query_pool, fend_translation_unit_pool);
    fend_query_identifiers = apr_hash_make(fend_query_pool);
}

void fend_translation_unit_init(void (*pfn_on_query_complete)(fend_node_t *ast)) {
    apr_pool_create(&fend_translation_unit_pool, fend_pool);
    fend_callback = pfn_on_query_complete;
}

static fend_node_t *prfend_create_node(fend_node_t *left, fend_node_t *right, node_type_t type) {
    fend_node_t *node = (fend_node_t *)apr_pcalloc(fend_query_pool, sizeof(fend_node_t));
    node->type = type;
    node->left = left;
    node->right = right;
    return node;
}

static fend_node_t *prfend_create_string_node(fend_node_t *left, fend_node_t *right, node_type_t type, char *value) {
    fend_node_t *node = prfend_create_node(left, right, type);
    node->value.string = value;
    return node;
}

static fend_node_t *prfend_create_number_node(fend_node_t *left, fend_node_t *right, node_type_t type,
                                              long long value) {
    fend_node_t *node = prfend_create_node(left, right, type);
    node->value.number = value;
    return node;
}

fend_node_t *fend_query_complete(fend_node_t *from, fend_node_t *body) {
    return prfend_create_node(from, body, node_type_query);
}

void fend_query_cleanup(fend_node_t *result) {
    fend_callback(result);
    apr_pool_destroy(fend_query_pool);
    fend_query_identifiers = NULL;
}

void fend_translation_unit_cleanup() { apr_pool_destroy(fend_translation_unit_pool); }

char *fend_translation_unit_strdup(char *str) { return apr_pstrdup(fend_translation_unit_pool, str); }

char *fend_query_strdup(char *str) { return apr_pstrdup(fend_query_pool, str); }

long long fend_to_number(char *str) {
    apr_off_t result = 0;
    apr_strtoff(&result, str, NULL, 0);
    return result;
}

type_info_t *fend_on_simple_type_def(type_def_t type) {
    type_info_t *result = (type_info_t *)apr_pcalloc(fend_query_pool, sizeof(type_info_t));
    result->type = type;
    return result;
}

type_info_t *fend_on_complex_type_def(type_def_t type, char *info) {
    type_info_t *result = fend_on_simple_type_def(type);
    result->info = fend_query_strdup(info);
    return result;
}

fend_node_t *fend_on_identifier_declaration(type_info_t *type, fend_node_t *identifier) {
    apr_hash_set(fend_query_identifiers, identifier->value.string, APR_HASH_KEY_STRING, type);
    identifier->left = fend_on_type_attribute(type);
    return identifier;
}

fend_node_t *fend_on_unary_expression(const unary_exp_type_t type, void *left_value, void *right_value) {
    fend_node_t *expr = prfend_create_node(NULL, NULL, node_type_unary_expression);
    switch (type) {
    case unary_exp_type_identifier:
        expr->left = left_value;
        break;
    case unary_exp_type_string:
        expr->left = prfend_create_string_node(NULL, NULL, node_type_string_literal, left_value);
        break;
    case unary_exp_type_number:
        expr->left = prfend_create_number_node(NULL, NULL, node_type_numeric_literal, (long long)left_value);
        break;
    case unary_exp_type_property_call:
    case unary_exp_type_mehtod_call:
        expr->left = left_value;
        expr->right = right_value;
        break;
    case unary_exp_type_undefined:
        break;
    default:
        break;
    }
    return expr;
}

fend_node_t *fend_on_from(fend_node_t *type, fend_node_t *datasource) {
    return prfend_create_node(type, datasource, node_type_from);
}

fend_node_t *fend_on_where(fend_node_t *expr) { return prfend_create_node(expr, NULL, node_type_where); }

fend_node_t *fend_on_releational_expr(fend_node_t *left, fend_node_t *right, cond_op_t relation) {
    fend_node_t *node = prfend_create_node(left, right, node_type_relation);
    node->value.relation_op = relation;
    return node;
}

fend_node_t *fend_on_predicate(fend_node_t *left, fend_node_t *right, node_type_t type) {
    return prfend_create_node(left, right, type);
}

fend_node_t *fend_on_enum(fend_node_t *left, fend_node_t *right) {
    return prfend_create_node(left, right, node_type_enum);
}

fend_node_t *fend_on_group(fend_node_t *left, fend_node_t *right) {
    return prfend_create_node(left, right, node_type_group);
}

fend_node_t *fend_on_let(fend_node_t *id, fend_node_t *expr) {
    if (fend_is_identifier_defined(expr)) {
        type_info_t *type = apr_hash_get(fend_query_identifiers, expr->value.string, APR_HASH_KEY_STRING);
        apr_hash_set(fend_query_identifiers, id->value.string, APR_HASH_KEY_STRING, type);
    }
    return prfend_create_node(id, expr, node_type_let);
}

fend_node_t *fend_on_query_body(fend_node_t *opt_query_body_clauses, fend_node_t *select_or_group_clause,
                                fend_node_t *opt_query_continuation) {
    fend_node_t *select = prfend_create_node(opt_query_body_clauses, select_or_group_clause, node_type_select);
    return prfend_create_node(select, opt_query_continuation, node_type_query_body);
}

fend_node_t *fend_on_string_attribute(char *str) {
    return prfend_create_string_node(NULL, NULL, node_type_property, str);
}

fend_node_t *fend_on_type_attribute(type_info_t *type) {
    fend_node_t *node = prfend_create_node(NULL, NULL, node_type_internal_type);
    node->value.type = type->type;
    if (type->info != NULL) {
        node->left = prfend_create_string_node(NULL, NULL, node_type_string_literal, type->info);
    }
    return node;
}

fend_node_t *fend_on_continuation(fend_node_t *id, fend_node_t *query_body) {
    return prfend_create_node(id, query_body, node_type_query_continuation);
}

fend_node_t *fend_on_method_call(char *method, fend_node_t *arguments) {
    return prfend_create_string_node(arguments, NULL, node_type_method_call, method);
}

fend_node_t *fend_on_identifier(char *id) { return prfend_create_string_node(NULL, NULL, node_type_identifier, id); }

fend_node_t *fend_on_join(fend_node_t *identifier, fend_node_t *in, fend_node_t *on_first, fend_node_t *on_second) {
    fend_node_t *on_node = prfend_create_node(on_first, on_second, node_type_on);
    fend_node_t *in_node = prfend_create_node(in, on_node, node_type_in);
    return prfend_create_node(identifier, in_node, node_type_join);
}

fend_node_t *fend_on_order_by(fend_node_t *ordering) { return prfend_create_node(ordering, NULL, node_type_order_by); }

fend_node_t *fend_on_ordering(fend_node_t *ordering, ordering_t direction) {
    fend_node_t *node = prfend_create_node(ordering, NULL, node_type_ordering);
    node->value.ordering = direction;
    return node;
}

BOOL fend_is_identifier_defined(fend_node_t *str) {
    type_info_t *result = apr_hash_get(fend_query_identifiers, str->value.string, APR_HASH_KEY_STRING);
    return result != NULL;
}

void fend_register_identifier(fend_node_t *id) {
    // TODO: Remove or make type detection
    apr_hash_set(fend_query_identifiers, id->value.string, APR_HASH_KEY_STRING, NULL);
}
