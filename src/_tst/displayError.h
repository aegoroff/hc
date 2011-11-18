#pragma once

#include    <antlr3.h>

#ifdef __cplusplus
extern "C" {
#endif

void displayRecognitionErrorNew	    (pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_UINT8 * tokenNames);

#ifdef __cplusplus
}
#endif