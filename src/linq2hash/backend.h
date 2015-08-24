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
#include <apr_general.h>
#include "frontend.h"

void backend_init(apr_pool_t* pool);

void print_label(Node_t* node, apr_pool_t* pool);
void emit(Node_t* node, apr_pool_t* pool);
char* create_label(Node_t* t, apr_pool_t* pool);

#endif // LINQ2HASH_BACKEND_H_