/*
 * Minimal stub for openssl/internal/cryptlib.h — only the symbols the
 * vendored whirlpool sources (wp_dgst.c, wp_block.c) actually reference.
 * Compiled from openssl-4.0.0 sources without WHIRLPOOL_ASM, so no
 * OPENSSL_ia32cap_P dependency.
 */
#ifndef OPENSSL_INTERNAL_CRYPTLIB_H
#define OPENSSL_INTERNAL_CRYPTLIB_H

#include <stddef.h>
#include <stdlib.h>

#ifndef OPENSSL_NO_DEPRECATED_3_0
# define OPENSSL_NO_DEPRECATED_3_0
#endif

void *OPENSSL_malloc(size_t num);
void OPENSSL_free(void *ptr);
void OPENSSL_cleanse(void *ptr, size_t len);
char *OPENSSL_strdup(const char *str);

#endif
