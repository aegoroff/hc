/* OPENSSL_cleanse/malloc/free are normally provided by OpenSSL's x86_64cpuid.o,
 * which also contributes a .init fragment (`call OPENSSL_cpuid_setup` without
 * `ret`). Zig's linker turns that fragment into DT_INIT for Debug binaries and
 * they SEGV before main. The vendored whirlpool sources only need these. */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void OPENSSL_cleanse(void *ptr, size_t len) {
    if (ptr != NULL && len != 0) {
        memset(ptr, 0, len);
    }
}

void *OPENSSL_malloc(size_t num) {
    return malloc(num != 0 ? num : 1);
}

void OPENSSL_free(void *ptr) {
    free(ptr);
}

char *OPENSSL_strdup(const char *str) {
    return str != NULL ? strdup(str) : NULL;
}
