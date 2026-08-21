#ifndef HC_HASHES_C_H_
#define HC_HASHES_C_H_

/*
 * translate-c root for hashes.zig — hash algorithm C APIs.
 * OPENSSL_API_COMPAT is defined on the TranslateC step.
 */

#include "sph_tiger.h"
#include "sph_md2.h"
#include "sph_md4.h"
#include "sph_ripemd.h"
#include "sph_haval.h"
#include "blake3.h"
#include "gost.h"
#include "tth.h"
#include "snefru.h"
#include "edonr.h"
#include "crc32.h"
#include "openssl/sha.h"
#include "openssl/md5.h"
#include "openssl/whrlpool.h"
#include "openssl/ripemd.h"

/*
 * OpenSSL SM3 has no public SM3_* API (only EVP_sm3). The low-level
 * ossl_sm3_* + SM3_CTX live in OpenSSL's internal header; declare them here
 * so we can use the same stack-ctx pattern as MD5/SHA (thread-safe oneshot
 * for brute force; EVP_sm3()/EVP_Digest races under threads).
 */
#define SM3_DIGEST_LENGTH 32
typedef struct SM3state_st {
    unsigned int A, B, C, D, E, F, G, H;
    unsigned int Nl, Nh;
    unsigned int data[16];
    unsigned int num;
} SM3_CTX;
int ossl_sm3_init(SM3_CTX *c);
int ossl_sm3_update(SM3_CTX *c, const void *data, size_t len);
int ossl_sm3_final(unsigned char *md, SM3_CTX *c);

/*
 * ASM CPU-cap probe. Defined in OpenSSL's cpuid.c; not always in public
 * headers. Static link into a Zig executable may skip the .init constructor
 * that normally runs it, leaving OPENSSL_ia32cap_P at 0 (software SHA path).
 * Call once from process startup (see hashes.ensureOpenSslReady).
 */
void OPENSSL_cpuid_setup(void);

#endif /* HC_HASHES_C_H_ */
