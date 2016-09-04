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