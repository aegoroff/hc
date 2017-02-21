// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "TreeTest.h"
extern "C" {
    #include <treeutil.h>
}

void TreeTest::SetUp() {
    apr_pool_create(&testPool_, pool_);
    root_ = this->CreateNode(1);
    root_->left = this->CreateNode(2);
    root_->right = this->CreateNode(3);
    root_->left->left = this->CreateNode(4);
    root_->right->left = this->CreateNode(5);
    root_->right->right = this->CreateNode(6);
    /*
             1 
            / \
           2   3
          /   / \
         4   5   6
    */
}

std::vector<long long> path;

void TreeTest::TearDown() {
    apr_pool_destroy(testPool_);
    path.clear();
}

fend_node_t* TreeTest::CreateNode(long long value) const {
    auto result = static_cast<fend_node_t*>(apr_pcalloc(testPool_, sizeof(fend_node_t)));
    result->value.number = value;
    return result;
}

void onVisit(fend_node_t* node, apr_pool_t* pool) {
    path.push_back(node->value.number);
}

TEST_F(TreeTest, inorder) {
    tree_inorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(4, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(1, path[2]);
    ASSERT_EQ(5, path[3]);
    ASSERT_EQ(3, path[4]);
    ASSERT_EQ(6, path[5]);
}

TEST_F(TreeTest, preorder) {
    tree_preorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(1, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(4, path[2]);
    ASSERT_EQ(3, path[3]);
    ASSERT_EQ(5, path[4]);
    ASSERT_EQ(6, path[5]);
}

TEST_F(TreeTest, postorder) {
    tree_postorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(4, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(5, path[2]);
    ASSERT_EQ(6, path[3]);
    ASSERT_EQ(3, path[4]);
    ASSERT_EQ(1, path[5]);
}