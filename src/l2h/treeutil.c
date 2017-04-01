/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains tree utils implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <apr_tables.h>
#include "lib.h"
#include "treeutil.h"
#include "backend.h"

#define MAX_HEIGHT 1000
#define STACK_INIT_SZ 128

static int tree_lprofile[MAX_HEIGHT];
static int tree_rprofile[MAX_HEIGHT];

#define H2L_INFINITY (1<<20)

//used for printing next node in the same level, 
//this is the x coordinate of the next char printed
static int tree_print_next;
//adjust gap between left and right nodes
static int tree_gap = 3;
static apr_pool_t* tree_pool = NULL;

void tree_inorder(fend_node_t* root, void(*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(fend_node_t*));
    fend_node_t* current = root;
    BOOL done = FALSE;

    while (!done) {
        if (current != NULL) {
            *(fend_node_t**)apr_array_push(stack) = current;
            current = current->left;
        }
        else {
            if (stack->nelts > 0) {
                current = *((fend_node_t**)apr_array_pop(stack));
                action(current, pool);
                current = current->right;
            }
            else {
                done = TRUE;
            }
        }
    }
}

void tree_postorder(fend_node_t* root, void(*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    if (root == NULL) {
        return;
    }

    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(fend_node_t*));
    *(fend_node_t**)apr_array_push(stack) = root;

    while (stack->nelts > 0) {
        fend_node_t* next = ((fend_node_t**)stack->elts)[stack->nelts - 1];

        BOOL finished_subtrees = (next->right == root || next->left == root);
        BOOL is_leaf = (next->left == NULL && next->right == NULL);
        if (finished_subtrees || is_leaf) {
            *((fend_node_t**)apr_array_pop(stack));
            action(next, pool);
            root = next;
        }
        else {
            if (next->right != NULL) {
                *(fend_node_t**)apr_array_push(stack) = next->right;
            }
            if (next->left != NULL) {
                *(fend_node_t**)apr_array_push(stack) = next->left;
            }
        }
    }
}

void tree_preorder(fend_node_t* root, void(*action)(fend_node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    if (root == NULL) {
        return;
    }

    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(fend_node_t*));
    *(fend_node_t**)apr_array_push(stack) = root;

    while (stack->nelts > 0) {
        fend_node_t* current = *((fend_node_t**)apr_array_pop(stack));
        action(current, pool);
        if (current->right != NULL) {
            *(fend_node_t**)apr_array_push(stack) = current->right;
        }
        if (current->left != NULL) {
            *(fend_node_t**)apr_array_push(stack) = current->left;
        }
    }
}

asciinode_t* build_ascii_tree_recursive(fend_node_t* t) {
	if(t == NULL) return NULL;

	asciinode_t* node = (asciinode_t*)apr_pcalloc(tree_pool, sizeof(asciinode_t));
	node->left = build_ascii_tree_recursive(t->left);
	node->right = build_ascii_tree_recursive(t->right);

	if(node->left != NULL) {
		node->left->parent_dir = -1;
	}

	if(node->right != NULL) {
		node->right->parent_dir = 1;
	}
	char* type = bend_create_label(t, tree_pool);

	sprintf(node->label, "%s", type);
	node->lablen = strlen(node->label);

	return node;
}

//The following function fills in the tree_lprofile array for the given tree.
//It assumes that the center of the label of the root of this tree
//is located at a position (x,y).  It assumes that the edge_length
//fields have been computed for this tree.
void compute_tree_lprofile(asciinode_t* node, int x, int y) {
	int i, isleft;
	if(node == NULL) return;
	isleft = (node->parent_dir == -1);
	tree_lprofile[y] = MIN(tree_lprofile[y], x - ((node->lablen - isleft) / 2));
	if(node->left != NULL) {
		for(i = 1; i <= node->edge_length && y + i < MAX_HEIGHT; i++) {
			tree_lprofile[y + i] = MIN(tree_lprofile[y + i], x - i);
		}
	}
	compute_tree_lprofile(node->left, x - node->edge_length - 1, y + node->edge_length + 1);
	compute_tree_lprofile(node->right, x + node->edge_length + 1, y + node->edge_length + 1);
}

void compute_tree_rprofile(asciinode_t* node, int x, int y) {
	int i, notleft;
	if(node == NULL) return;
	notleft = (node->parent_dir != -1);
	tree_rprofile[y] = MAX(tree_rprofile[y], x + ((node->lablen - notleft) / 2));
	if(node->right != NULL) {
		for(i = 1; i <= node->edge_length && y + i < MAX_HEIGHT; i++) {
			tree_rprofile[y + i] = MAX(tree_rprofile[y + i], x + i);
		}
	}
	compute_tree_rprofile(node->left, x - node->edge_length - 1, y + node->edge_length + 1);
	compute_tree_rprofile(node->right, x + node->edge_length + 1, y + node->edge_length + 1);
}

//This function fills in the edge_length and 
//height fields of the specified tree
void compute_edge_lengths(asciinode_t* node) {
	int h, hmin, i, delta;
	if(node == NULL) return;
	compute_edge_lengths(node->left);
	compute_edge_lengths(node->right);

	/* first fill in the edge_length of node */
	if(node->right == NULL && node->left == NULL) {
		node->edge_length = 0;
	}
	else {
		if(node->left != NULL) {
			for(i = 0; i < node->left->height && i < MAX_HEIGHT; i++) {
				tree_rprofile[i] = -H2L_INFINITY;
			}
			compute_tree_rprofile(node->left, 0, 0);
			hmin = node->left->height;
		}
		else {
			hmin = 0;
		}
		if(node->right != NULL) {
			for(i = 0; i < node->right->height && i < MAX_HEIGHT; i++) {
				tree_lprofile[i] = H2L_INFINITY;
			}
			compute_tree_lprofile(node->right, 0, 0);
			hmin = MIN(node->right->height, hmin);
		}
		else {
			hmin = 0;
		}
		delta = 4;
		for(i = 0; i < hmin; i++) {
			delta = MAX(delta, tree_gap + 1 + tree_rprofile[i] - tree_lprofile[i]);
		}

		//If the node has two children of height 1, then we allow the
		//two leaves to be within 1, instead of 2 
		if((node->left != NULL && node->left->height == 1 ||
			node->right != NULL && node->right->height == 1) && delta > 4) {
			delta--;
		}

		node->edge_length = ((delta + 1) / 2) - 1;
	}

	//now fill in the height of node
	h = 1;
	if(node->left != NULL) {
		h = MAX(node->left->height + node->edge_length + 1, h);
	}
	if(node->right != NULL) {
		h = MAX(node->right->height + node->edge_length + 1, h);
	}
	node->height = h;
}

//Copy the tree into the ascii node structre
asciinode_t* build_ascii_tree(fend_node_t* t) {
	asciinode_t* node;
	if(t == NULL) return NULL;
	node = build_ascii_tree_recursive(t);
	node->parent_dir = 0;
	return node;
}

//This function prints the given level of the given tree, assuming
//that the node has the given x cordinate.
void print_level(asciinode_t* node, int x, int level) {
	int i, isleft;
	if(node == NULL) return;
	isleft = (node->parent_dir == -1);
	if(level == 0) {
		for(i = 0; i < (x - tree_print_next - ((node->lablen - isleft) / 2)); i++) {
			lib_printf(" ");
		}
		tree_print_next += i;
		lib_printf("%s", node->label);
		tree_print_next += node->lablen;
	}
	else if(node->edge_length >= level) {
		if(node->left != NULL) {
			for(i = 0; i < (x - tree_print_next - (level)); i++) {
				lib_printf(" ");
			}
			tree_print_next += i;
			lib_printf("/");
			tree_print_next++;
		}
		if(node->right != NULL) {
			for(i = 0; i < (x - tree_print_next + (level)); i++) {
				lib_printf(" ");
			}
			tree_print_next += i;
			lib_printf("\\");
			tree_print_next++;
		}
	}
	else {
		print_level(node->left,
		            x - node->edge_length - 1,
		            level - node->edge_length - 1);
		print_level(node->right,
		            x + node->edge_length + 1,
		            level - node->edge_length - 1);
	}
}

//prints ascii tree for given Tree structure
void tree_print_ascii_tree(fend_node_t* t, apr_pool_t* pool) {
	asciinode_t* proot;
	int xmin, i;
	tree_pool = pool;
	if(t == NULL) return;
	proot = build_ascii_tree(t);
	compute_edge_lengths(proot);
	for(i = 0; i < proot->height && i < MAX_HEIGHT; i++) {
		tree_lprofile[i] = H2L_INFINITY;
	}
	compute_tree_lprofile(proot, 0, 0);
	xmin = 0;
	for(i = 0; i < proot->height && i < MAX_HEIGHT; i++) {
		xmin = MIN(xmin, tree_lprofile[i]);
	}
	for(i = 0; i < proot->height; i++) {
		tree_print_next = 0;
		print_level(proot, -xmin, i);
		lib_printf("\n");
	}
	if(proot->height >= MAX_HEIGHT) {
		lib_printf("(This tree is taller than %d, and may be drawn incorrectly.)\n", MAX_HEIGHT);
	}
}
