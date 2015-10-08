/*!
 * \brief   The file contains backend interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-22
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#ifndef LINQ2HASH_BACKEND_H_
#define LINQ2HASH_BACKEND_H_
#include "frontend.h"

typedef struct bend_data_source_t {
    type_def_t type;
    char* name;
    long long items_count;
    char* (*get_item)(long long index);
} bend_data_source_t;

void bend_init(apr_pool_t* pool);
void bend_cleanup();

void bend_print_label(fend_node_t* node, apr_pool_t* pool);
void bend_emit(fend_node_t* node, apr_pool_t* pool);
char* bend_create_label(fend_node_t* t, apr_pool_t* pool);
BOOL bend_match_re(const char* pattern, const char* subject);

#endif // LINQ2HASH_BACKEND_H_