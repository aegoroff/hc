/*!
 * \brief   The file contains tree utils implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#include <apr_tables.h>
#include "lib.h"
#include "treeutil.h"
#include "backend.h"

#define MAX_HEIGHT 1000
#define STACK_INIT_SZ 128

int lprofile[MAX_HEIGHT];
int rprofile[MAX_HEIGHT];
#ifndef INFINITY
    #define INFINITY (1<<20)
#endif

//used for printing next node in the same level, 
//this is the x coordinate of the next char printed
int print_next;
//adjust gap between left and right nodes
int gap = 3;
apr_pool_t* treePool = NULL;

void inorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(Node_t*));
    Node_t* current = root;
    BOOL done = FALSE;

    while (!done) {
        if (current != NULL) {
            *(Node_t**)apr_array_push(stack) = current;
            current = current->Left;
        }
        else {
            if (stack->nelts > 0) {
                current = *((Node_t**)apr_array_pop(stack));
                action(current, pool);
                current = current->Right;
            }
            else {
                done = TRUE;
            }
        }
    }
}

void postorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    if (root == NULL) {
        return;
    }

    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(Node_t*));
    *(Node_t**)apr_array_push(stack) = root;

    while (stack->nelts > 0) {
        Node_t* next = ((Node_t**)stack->elts)[stack->nelts - 1];

        BOOL finished_subtrees = (next->Right == root || next->Left == root);
        BOOL is_leaf = (next->Left == NULL && next->Right == NULL);
        if (finished_subtrees || is_leaf) {
            *((Node_t**)apr_array_pop(stack));
            action(next, pool);
            root = next;
        }
        else {
            if (next->Right != NULL) {
                *(Node_t**)apr_array_push(stack) = next->Right;
            }
            if (next->Left != NULL) {
                *(Node_t**)apr_array_push(stack) = next->Left;
            }
        }
    }
}

void preorder(Node_t* root, void(*action)(Node_t* node, apr_pool_t* pool), apr_pool_t* pool) {
    if (root == NULL) {
        return;
    }

    apr_array_header_t* stack = apr_array_make(pool, STACK_INIT_SZ, sizeof(Node_t*));
    *(Node_t**)apr_array_push(stack) = root;

    while (stack->nelts > 0) {
        Node_t* current = *((Node_t**)apr_array_pop(stack));
        action(current, pool);
        if (current->Right != NULL) {
            *(Node_t**)apr_array_push(stack) = current->Right;
        }
        if (current->Left != NULL) {
            *(Node_t**)apr_array_push(stack) = current->Left;
        }
    }
}

asciinode* build_ascii_tree_recursive(Node_t* t) {
	if(t == NULL) return NULL;

	asciinode* node = (asciinode*)apr_pcalloc(treePool, sizeof(asciinode));
	node->left = build_ascii_tree_recursive(t->Left);
	node->right = build_ascii_tree_recursive(t->Right);

	if(node->left != NULL) {
		node->left->parent_dir = -1;
	}

	if(node->right != NULL) {
		node->right->parent_dir = 1;
	}
	char* type = create_label(t, treePool);

	sprintf(node->label, "%s", type);
	node->lablen = strlen(node->label);

	return node;
}

//The following function fills in the lprofile array for the given tree.
//It assumes that the center of the label of the root of this tree
//is located at a position (x,y).  It assumes that the edge_length
//fields have been computed for this tree.
void compute_lprofile(asciinode* node, int x, int y) {
	int i, isleft;
	if(node == NULL) return;
	isleft = (node->parent_dir == -1);
	lprofile[y] = MIN(lprofile[y], x - ((node->lablen - isleft) / 2));
	if(node->left != NULL) {
		for(i = 1; i <= node->edge_length && y + i < MAX_HEIGHT; i++) {
			lprofile[y + i] = MIN(lprofile[y + i], x - i);
		}
	}
	compute_lprofile(node->left, x - node->edge_length - 1, y + node->edge_length + 1);
	compute_lprofile(node->right, x + node->edge_length + 1, y + node->edge_length + 1);
}

void compute_rprofile(asciinode* node, int x, int y) {
	int i, notleft;
	if(node == NULL) return;
	notleft = (node->parent_dir != -1);
	rprofile[y] = MAX(rprofile[y], x + ((node->lablen - notleft) / 2));
	if(node->right != NULL) {
		for(i = 1; i <= node->edge_length && y + i < MAX_HEIGHT; i++) {
			rprofile[y + i] = MAX(rprofile[y + i], x + i);
		}
	}
	compute_rprofile(node->left, x - node->edge_length - 1, y + node->edge_length + 1);
	compute_rprofile(node->right, x + node->edge_length + 1, y + node->edge_length + 1);
}

//This function fills in the edge_length and 
//height fields of the specified tree
void compute_edge_lengths(asciinode* node) {
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
				rprofile[i] = -INFINITY;
			}
			compute_rprofile(node->left, 0, 0);
			hmin = node->left->height;
		}
		else {
			hmin = 0;
		}
		if(node->right != NULL) {
			for(i = 0; i < node->right->height && i < MAX_HEIGHT; i++) {
				lprofile[i] = INFINITY;
			}
			compute_lprofile(node->right, 0, 0);
			hmin = MIN(node->right->height, hmin);
		}
		else {
			hmin = 0;
		}
		delta = 4;
		for(i = 0; i < hmin; i++) {
			delta = MAX(delta, gap + 1 + rprofile[i] - lprofile[i]);
		}

		//If the node has two children of height 1, then we allow the
		//two leaves to be within 1, instead of 2 
		if(((node->left != NULL && node->left->height == 1) ||
			(node->right != NULL && node->right->height == 1)) && delta > 4) {
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
asciinode* build_ascii_tree(Node_t* t) {
	asciinode* node;
	if(t == NULL) return NULL;
	node = build_ascii_tree_recursive(t);
	node->parent_dir = 0;
	return node;
}

//This function prints the given level of the given tree, assuming
//that the node has the given x cordinate.
void print_level(asciinode* node, int x, int level) {
	int i, isleft;
	if(node == NULL) return;
	isleft = (node->parent_dir == -1);
	if(level == 0) {
		for(i = 0; i < (x - print_next - ((node->lablen - isleft) / 2)); i++) {
			CrtPrintf(" ");
		}
		print_next += i;
		CrtPrintf("%s", node->label);
		print_next += node->lablen;
	}
	else if(node->edge_length >= level) {
		if(node->left != NULL) {
			for(i = 0; i < (x - print_next - (level)); i++) {
				CrtPrintf(" ");
			}
			print_next += i;
			CrtPrintf("/");
			print_next++;
		}
		if(node->right != NULL) {
			for(i = 0; i < (x - print_next + (level)); i++) {
				CrtPrintf(" ");
			}
			print_next += i;
			CrtPrintf("\\");
			print_next++;
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
void print_ascii_tree(Node_t* t, apr_pool_t* pool) {
	asciinode* proot;
	int xmin, i;
	treePool = pool;
	if(t == NULL) return;
	proot = build_ascii_tree(t);
	compute_edge_lengths(proot);
	for(i = 0; i < proot->height && i < MAX_HEIGHT; i++) {
		lprofile[i] = INFINITY;
	}
	compute_lprofile(proot, 0, 0);
	xmin = 0;
	for(i = 0; i < proot->height && i < MAX_HEIGHT; i++) {
		xmin = MIN(xmin, lprofile[i]);
	}
	for(i = 0; i < proot->height; i++) {
		print_next = 0;
		print_level(proot, -xmin, i);
		CrtPrintf("\n");
	}
	if(proot->height >= MAX_HEIGHT) {
		CrtPrintf("(This tree is taller than %d, and may be drawn incorrectly.)\n", MAX_HEIGHT);
	}
}
