/*
 * Minimal C implementation of the lib.h printing helpers used by the generated
 * bison/flex parser (l2h.tab.c calls lib_fprintf from yyerror/lyyerror).
 *
 * Zig cannot export a C-variadic function, so these thin wrappers stay in C and
 * are compiled into the l2h-c static library. The rest of the "lib" surface is
 * provided by the Zig lib.zig module.
 */
#include <stdio.h>
#include <stdarg.h>
#include "lib.h"

int lib_printf(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    const int result = vprintf(format, ap);
    va_end(ap);
    return result;
}

int lib_fprintf(FILE* file, const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    const int result = vfprintf(file, format, ap);
    va_end(ap);
    return result;
}

int lib_snprintf(char* buffer, size_t size, const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    const int result = vsnprintf(buffer, size, format, ap);
    va_end(ap);
    return result;
}

void lib_new_line(void) {
    fputc('\n', stdout);
}
