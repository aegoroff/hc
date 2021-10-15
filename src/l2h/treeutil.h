/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains tree utils interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#ifndef LINQ2HASH_TREEUTIL_H_
#define LINQ2HASH_TREEUTIL_H_
#include "frontend.h"

typedef struct asciinode_t asciinode_t;

struct asciinode_t {
    asciinode_t* left;
    asciinode_t* right;

    //length of the edge from this node to its children
    int edge_length;

    int height;

    int lablen;

    //-1=I am left, 0=I am root, 1=right   
    int parent_dir;

    char label[80];
};

void tree_inorder(fend_node_t* root, void (*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool);
void tree_preorder(fend_node_t* root, void (*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool);
void tree_postorder(fend_node_t* root, void (*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool);

void tree_print_ascii_tree(fend_node_t* t, apr_pool_t* pool);

#endif // LINQ2HASH_TREEUTIL_H_
