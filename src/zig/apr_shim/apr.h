#ifndef HC_APR_SHIM_APR_H
#define HC_APR_SHIM_APR_H

/*
 * Minimal APR type shim, consumed ONLY by translate-c (build.zig translate_bf)
 * on the Windows target. The real external_lib/apr/include/apr.h includes
 * windows.h / winsock2.h, and libclang's translate-c cannot handle the Win32
 * SDK headers (SAL annotations, PVOID64, PCONTEXT, ...). This shim mirrors just
 * the C ABI that bf.zig calls; the real symbols still resolve from apr-1.lib at
 * link time. Types must stay byte-for-byte compatible with Apache APR 1.x.
 */

#include <stddef.h>
#include <stdint.h>

typedef unsigned char apr_byte_t;
typedef size_t apr_size_t;
typedef int apr_int32_t;
typedef apr_int32_t apr_status_t;

#endif /* HC_APR_SHIM_APR_H */
