/*!
 * \brief   The file contains test of SizeToString function interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2010
 */

#pragma once

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include "BufferedTest.h"

class TSizeToString : public BufferedTest {
    protected:
        size_t GetBufferSize() const;
};
