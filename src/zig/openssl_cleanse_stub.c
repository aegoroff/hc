/* OPENSSL_cleanse is normally provided by OpenSSL's x86_64cpuid.o, which also
 * contributes a .init fragment (`call OPENSSL_cpuid_setup` without `ret`).
 * Zig's linker turns that fragment into DT_INIT for Debug binaries and they
 * SEGV before main. WHIRLPOOL only needs cleanse + the wp_*.o objects. */
#include <stddef.h>
#include <string.h>

void OPENSSL_cleanse(void *ptr, size_t len) {
    if (ptr != NULL && len != 0) {
        memset(ptr, 0, len);
    }
}
