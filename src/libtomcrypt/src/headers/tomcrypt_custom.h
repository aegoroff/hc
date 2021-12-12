#ifndef TOMCRYPT_CUSTOM_H_
#define TOMCRYPT_CUSTOM_H_


#ifndef XMEMCPY
   #ifdef memcpy
   #define LTC_NO_PROTOTYPES
   #endif
#define XMEMCPY  memcpy
#endif

/* ---> One-Way Hash Functions <--- */

#define LTC_RIPEMD128
#define LTC_RIPEMD160
#define LTC_RIPEMD256
#define LTC_RIPEMD320

#endif

/* $Source: /cvs/libtom/libtomcrypt/src/headers/tomcrypt_custom.h,v $ */
/* $Revision: 1.73 $ */
/* $Date: 2007/05/12 14:37:41 $ */
