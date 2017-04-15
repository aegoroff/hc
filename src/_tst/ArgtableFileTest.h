#pragma once
#include "gtest.h"

class ArgtableFileTest : public testing::Test {
public:
    void** argtable;
    struct arg_file* a;
    size_t n;
    void SetUp() override;
    void TearDown() override;
};
