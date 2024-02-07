#ifndef TOMCRYPT_HASH_H_
#define TOMCRYPT_HASH_H_

/* ---- HASH FUNCTIONS ---- */
#ifdef LTC_RIPEMD128
struct rmd128_state {
    ulong64 length;
    unsigned char buf[64];
    ulong32 curlen, state[4];
};
#endif

#ifdef LTC_RIPEMD160
struct rmd160_state {
    ulong64 length;
    unsigned char buf[64];
    ulong32 curlen, state[5];
};
#endif

#ifdef LTC_RIPEMD256
struct rmd256_state {
    ulong64 length;
    unsigned char buf[64];
    ulong32 curlen, state[8];
};
#endif

#ifdef LTC_RIPEMD320
struct rmd320_state {
    ulong64 length;
    unsigned char buf[64];
    ulong32 curlen, state[10];
};
#endif

#ifdef LTC_BLAKE2S
struct blake2s_state {
    ulong32 h[8];
    ulong32 t[2];
    ulong32 f[2];
    unsigned char buf[64];
    unsigned long curlen;
    unsigned long outlen;
    unsigned char last_node;
};
#endif

#ifdef LTC_BLAKE2B
struct blake2b_state {
    ulong64 h[8];
    ulong64 t[2];
    ulong64 f[2];
    unsigned char buf[128];
    unsigned long curlen;
    unsigned long outlen;
    unsigned char last_node;
};
#endif

typedef union Hash_state {
    char dummy[1];
#ifdef LTC_RIPEMD128
    struct rmd128_state rmd128;
#endif
#ifdef LTC_RIPEMD160
    struct rmd160_state rmd160;
#endif
#ifdef LTC_RIPEMD256
    struct rmd256_state rmd256;
#endif
#ifdef LTC_RIPEMD320
    struct rmd320_state rmd320;
#endif
#ifdef LTC_BLAKE2S
    struct blake2s_state blake2s;
#endif
#ifdef LTC_BLAKE2B
    struct blake2b_state blake2b;
#endif
    void *data;
} hash_state;

#ifdef LTC_RIPEMD128
int rmd128_init(hash_state * md);
int rmd128_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int rmd128_done(hash_state * md, unsigned char *hash);
extern const struct ltc_hash_descriptor rmd128_desc;
#endif

#ifdef LTC_RIPEMD160
int rmd160_init(hash_state * md);
int rmd160_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int rmd160_done(hash_state * md, unsigned char *hash);
extern const struct ltc_hash_descriptor rmd160_desc;
#endif

#ifdef LTC_RIPEMD256
int rmd256_init(hash_state * md);
int rmd256_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int rmd256_done(hash_state * md, unsigned char *hash);
extern const struct ltc_hash_descriptor rmd256_desc;
#endif

#ifdef LTC_RIPEMD320
int rmd320_init(hash_state * md);
int rmd320_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int rmd320_done(hash_state * md, unsigned char *hash);
extern const struct ltc_hash_descriptor rmd320_desc;
#endif

#ifdef LTC_BLAKE2S
extern const struct ltc_hash_descriptor blake2s_256_desc;
int blake2s_256_init(hash_state * md);
int blake2s_256_test(void);

extern const struct ltc_hash_descriptor blake2s_224_desc;
int blake2s_224_init(hash_state * md);
int blake2s_224_test(void);

extern const struct ltc_hash_descriptor blake2s_160_desc;
int blake2s_160_init(hash_state * md);
int blake2s_160_test(void);

extern const struct ltc_hash_descriptor blake2s_128_desc;
int blake2s_128_init(hash_state * md);
int blake2s_128_test(void);

int blake2s_init(hash_state * md, unsigned long outlen, const unsigned char *key, unsigned long keylen);
int blake2s_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int blake2s_done(hash_state * md, unsigned char *out);
#endif

#ifdef LTC_BLAKE2B
extern const struct ltc_hash_descriptor blake2b_512_desc;
int blake2b_512_init(hash_state * md);
int blake2b_512_test(void);

extern const struct ltc_hash_descriptor blake2b_384_desc;
int blake2b_384_init(hash_state * md);
int blake2b_384_test(void);

extern const struct ltc_hash_descriptor blake2b_256_desc;
int blake2b_256_init(hash_state * md);
int blake2b_256_test(void);

extern const struct ltc_hash_descriptor blake2b_160_desc;
int blake2b_160_init(hash_state * md);
int blake2b_160_test(void);

int blake2b_init(hash_state * md, unsigned long outlen, const unsigned char *key, unsigned long keylen);
int blake2b_process(hash_state * md, const unsigned char *in, unsigned long inlen);
int blake2b_done(hash_state * md, unsigned char *out);
#endif

/* a simple macro for making hash "process" functions */
#define HASH_PROCESS(func_name, compress_name, state_var, block_size)                       \
int func_name (hash_state * md, const unsigned char *in, unsigned long inlen)               \
{                                                                                           \
    unsigned long n;                                                                        \
    int           err;                                                                      \
    LTC_ARGCHK(md != NULL);                                                                 \
    LTC_ARGCHK(in != NULL);                                                                 \
    if (md-> state_var .curlen > sizeof(md-> state_var .buf)) {                             \
       return CRYPT_INVALID_ARG;                                                            \
    }                                                                                       \
    while (inlen > 0) {                                                                     \
        if (md-> state_var .curlen == 0 && inlen >= block_size) {                           \
           if ((err = compress_name (md, (unsigned char *)in)) != CRYPT_OK) {               \
              return err;                                                                   \
           }                                                                                \
           md-> state_var .length += block_size * 8;                                        \
           in             += block_size;                                                    \
           inlen          -= block_size;                                                    \
        } else {                                                                            \
           n = MIN(inlen, (block_size - md-> state_var .curlen));                           \
           memcpy(md-> state_var .buf + md-> state_var.curlen, in, (size_t)n);              \
           md-> state_var .curlen += n;                                                     \
           in             += n;                                                             \
           inlen          -= n;                                                             \
           if (md-> state_var .curlen == block_size) {                                      \
              if ((err = compress_name (md, md-> state_var .buf)) != CRYPT_OK) {            \
                 return err;                                                                \
              }                                                                             \
              md-> state_var .length += 8*block_size;                                       \
              md-> state_var .curlen = 0;                                                   \
           }                                                                                \
       }                                                                                    \
    }                                                                                       \
    return CRYPT_OK;                                                                        \
}

/* $Source: /cvs/libtom/libtomcrypt/src/headers/tomcrypt_hash.h,v $ */
/* $Revision: 1.22 $ */
/* $Date: 2007/05/12 14:32:35 $ */

#endif // TOMCRYPT_HASH_H_