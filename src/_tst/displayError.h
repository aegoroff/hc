#pragma once

#include    <antlr3.h>
#include    "output.h"

#ifdef __cplusplus
extern "C" {
#endif

void displayRecognitionErrorNew	    (pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_UINT8 * tokenNames);
void OutputToCppConsole(OutputContext* ctx);

#ifdef __cplusplus
}
#endif