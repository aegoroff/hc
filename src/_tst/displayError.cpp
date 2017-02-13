// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include "displayError.h"


void OutputToCppConsole(out_context_t* ctx)
{
    std::cout << ctx->string_to_print_;
    if (ctx->is_print_separator_) {
        std::cout << FILE_INFO_COLUMN_SEPARATOR;
    }
    if (ctx->is_finish_line_) {
        std::cout << std::endl;
    }
}