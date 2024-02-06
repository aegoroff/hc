/*!
 * \brief   The file contains argtable test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#pragma once
#include "ArgtableTestBase.h"

class ArgtableDateTest : public ArgtableTestBase {
public:
protected:
    struct arg_date* a;
    struct arg_date* b;
    struct arg_date* c;
    void SetUp() override;
    size_t GetOptionsCount() override;
};
