/*!
 * \brief   The file contains argtable test implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#include "ArgtableTestBase.h"
#include "argtable3.h"

void ArgtableTestBase::TearDown() {
    arg_freetable(argtable, GetOptionsCount() + 1);
}
