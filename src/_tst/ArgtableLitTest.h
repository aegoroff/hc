/*!
 * \brief   The file contains argtable test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-17
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#pragma once
#include "ArgtableTestBase.h"

class ArgtableLitTest : public ArgtableTestBase {
public:
protected:
    struct arg_lit* a;
    struct arg_lit* b;
    struct arg_lit* c;
    struct arg_lit* d;
    struct arg_lit* help;
    void SetUp() override;
    size_t GetOptionsCount() override;
};
