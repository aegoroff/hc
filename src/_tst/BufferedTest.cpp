/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#include "BufferedTest.h"

using namespace std;

void BufferedTest::SetUp() {
    buffer_ = make_unique<char[]>(GetBufferSize());
}

char* BufferedTest::GetBuffer() const {
    return buffer_.get();
}
