/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2015
 */

#include "TreeTest.h"
extern "C" {
    #include <treeutil.h>
}

void TreeTest::SetUp() {
    apr_pool_create(&testPool_, pool_);
    root_ = this->CreateNode(1);
    root_->Left = this->CreateNode(2);
    root_->Right = this->CreateNode(3);
    root_->Left->Left = this->CreateNode(4);
    root_->Right->Left = this->CreateNode(5);
    root_->Right->Right = this->CreateNode(6);
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

Node_t* TreeTest::CreateNode(long long value) const {
    auto result = static_cast<Node_t*>(apr_pcalloc(testPool_, sizeof(Node_t)));
    result->Value.Number = value;
    return result;
}

void onVisit(Node_t* node, apr_pool_t* pool) {
    path.push_back(node->Value.Number);
}

TEST_F(TreeTest, inorder) {
    inorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(4, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(1, path[2]);
    ASSERT_EQ(5, path[3]);
    ASSERT_EQ(3, path[4]);
    ASSERT_EQ(6, path[5]);
}

TEST_F(TreeTest, preorder) {
    preorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(1, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(4, path[2]);
    ASSERT_EQ(3, path[3]);
    ASSERT_EQ(5, path[4]);
    ASSERT_EQ(6, path[5]);
}

TEST_F(TreeTest, postorder) {
    postorder(root_, &onVisit, testPool_);
    ASSERT_EQ(6, path.size());
    
    ASSERT_EQ(4, path[0]);
    ASSERT_EQ(2, path[1]);
    ASSERT_EQ(5, path[2]);
    ASSERT_EQ(6, path[3]);
    ASSERT_EQ(3, path[4]);
    ASSERT_EQ(1, path[5]);
}