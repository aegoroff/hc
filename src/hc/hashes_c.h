#ifndef HC_HASHES_C_H_
#define HC_HASHES_C_H_

/*
 * translate-c root for hashes.zig — hash algorithm C APIs.
 * OPENSSL_API_COMPAT and USE_KECCAK are defined on the TranslateC step.
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
#include "sha3.h"
#include "crc32.h"
#include "openssl/sha.h"
#include "openssl/md5.h"
#include "openssl/whrlpool.h"
#include "openssl/ripemd.h"

/*
 * ASM CPU-cap probe. Defined in OpenSSL's cpuid.c; not always in public
 * headers. Static link into a Zig executable may skip the .init constructor
 * that normally runs it, leaving OPENSSL_ia32cap_P at 0 (software SHA path).
 * Call once from process startup (see hashes.ensureOpenSslReady).
 */
void OPENSSL_cpuid_setup(void);

#endif /* HC_HASHES_C_H_ */
