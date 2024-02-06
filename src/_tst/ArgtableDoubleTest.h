/*!
 * \brief   The file contains argtable test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-13
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#pragma once
#include "ArgtableTestBase.h"

class ArgtableDoubleTest : public ArgtableTestBase {
public:
protected:
    struct arg_dbl* a;
    struct arg_dbl* b;
    struct arg_dbl* c;
    struct arg_dbl* d;
    struct arg_dbl* e;
    void SetUp() override;
    size_t GetOptionsCount() override;
};
