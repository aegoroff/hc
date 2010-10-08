/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-08
            \endverbatim
 * Copyright: (c) Alexander Egorov 2010
 */

#include <memory>
#include "BufferedTest.h"

using namespace std;

void BufferedTest::SetUp()
{
    buffer_ = std::auto_ptr<char>(new char[GetBufferSize()]);
    memset(buffer_.get(), 0, GetBufferSize());
}

char* BufferedTest::GetBuffer() const
{
    return buffer_.get();
}
