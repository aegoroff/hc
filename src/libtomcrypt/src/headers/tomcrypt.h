#ifndef TOMCRYPT_H_
#define TOMCRYPT_H_
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>

/* use configuration data */
#include <tomcrypt_custom.h>

#ifdef __cplusplus
extern "C" {
#endif

/* version */
#define CRYPT   0x0117
#define SCRYPT  "1.17"

/* max size of either a cipher/hash block or symmetric key [largest of the two] */
#define MAXBLOCKSIZE  128

/* descriptor table size */
#define TAB_SIZE      32

/* error codes [will be expanded in future releases] */
enum {
   CRYPT_OK=0,             /* Result OK */
   CRYPT_INVALID_ARG,      /* Generic invalid argument */
   CRYPT_ERROR,
   CRYPT_NOP
};

#include <tomcrypt_cfg.h>
#include <tomcrypt_macros.h>
#include <tomcrypt_hash.h>
#include <tomcrypt_argchk.h>

void zeromem(volatile void *out, size_t outlen);

#ifdef __cplusplus
   }
#endif

#endif /* TOMCRYPT_H_ */


/* $Source: /cvs/libtom/libtomcrypt/src/headers/tomcrypt.h,v $ */
/* $Revision: 1.21 $ */
/* $Date: 2006/12/16 19:34:05 $ */
