/*!
 * \brief   The file contains tree utils interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#ifndef LINQ2HASH_TREEUTIL_H_
#define LINQ2HASH_TREEUTIL_H_
#include <apr_general.h>
#include "frontend.h"

typedef struct asciinode_struct asciinode;

struct asciinode_struct
{
	asciinode * left, *right;

	//length of the edge from this node to its children
	int edge_length;

	int height;

	int lablen;

	//-1=I am left, 0=I am root, 1=right   
	int parent_dir;

	char label[80];
};

void inorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool);
void preorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool);
void postorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool);

void print_ascii_tree(Node_t * t, apr_pool_t* pool);

#endif // LINQ2HASH_TREEUTIL_H_