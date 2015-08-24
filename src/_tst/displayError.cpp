#include <iostream>
#include "displayError.h"


void OutputToCppConsole(OutputContext* ctx)
{
    std::cout << ctx->StringToPrint;
    if (ctx->IsPrintSeparator) {
        std::cout << FILE_INFO_COLUMN_SEPARATOR;
    }
    if (ctx->IsFinishLine) {
        std::cout << std::endl;
    }
}