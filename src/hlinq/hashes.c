/*!
 * \brief   The file contains hashes from libtom lib API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include    <tomcrypt.h>
#include "hashes.h"
#include "sph_md2.h"
#include "sph_ripemd.h"
#include "sph_sha2.h"
#include "sph_tiger.h"
#include "sph_sha2.h"
#include "..\md5\sph_md5.h"
#include "..\md4\sph_md4.h"
#include "..\crc32\crc32.h"
#include "..\sha1\sph_sha1.h"
#include "..\whirlpool\sph_whirlpool.h"
#include "gost.h"

void LibtomInitContext(void* context, int (* PfnInit)(hash_state* md))
{
    PfnInit((hash_state*)context);
}

void LibtomFinalHash(apr_byte_t* digest, void* context, int (* PfnDone)(hash_state*    md,
                                                                                unsigned char* hash))
{
    PfnDone((hash_state*)context, digest);
}

void LibtomUpdateHash(void*                    context,
                              const void*              input,
                              const apr_size_t         inputLen,
                              int                      (* PfnProcess)(
                                  hash_state*          md,
                                  const unsigned char* in,
                                  unsigned long        inlen))
{
    if (input == NULL) {
        PfnProcess((hash_state*)context, "", 0);
    } else {
        PfnProcess((hash_state*)context, input, inputLen);
    }
}



void LibtomCalculateDigest(
    apr_byte_t* digest,
    const void* input,
    const apr_size_t inputLen,
    int (* PfnInit)(hash_state* md),
    int (* PfnProcess)(hash_state* md, const unsigned char* in, unsigned long inlen),
    int (* PfnDone)(hash_state* md, unsigned char* hash)
                                  )
{
    hash_state context = { 0 };

    LibtomInitContext(&context, PfnInit);
    LibtomUpdateHash(&context, input, inputLen, PfnProcess);
    LibtomFinalHash(digest, &context, PfnDone);
}

void WHIRLPOOLCalculateDigest(apr_byte_t*      digest,
                                      const void*      input,
                                      const apr_size_t inputLen)
{
    sph_whirlpool_context context = { 0 };

    WHIRLPOOLInitContext(&context);
    WHIRLPOOLUpdateHash(&context, input, inputLen);
    WHIRLPOOLFinalHash(digest, &context);
}

void WHIRLPOOLInitContext(void* context)
{
    sph_whirlpool_init(context);
}

void WHIRLPOOLFinalHash(apr_byte_t* digest, void* context)
{
    sph_whirlpool_close(context, digest);
}

void WHIRLPOOLUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_whirlpool(context, input, inputLen);
}


void SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha512_context context = { 0 };

    SHA512InitContext(&context);
    SHA512UpdateHash(&context, input, inputLen);
    SHA512FinalHash(digest, &context);
}

void SHA512InitContext(void* context)
{
    sph_sha512_init(context);
}

void SHA512FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha512_close(context, digest);
}

void SHA512UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha512(context, input, inputLen);
}


void SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha384_context context = { 0 };

    SHA384InitContext(&context);
    SHA384UpdateHash(&context, input, inputLen);
    SHA384FinalHash(digest, &context);
}

void SHA384InitContext(void* context)
{
    sph_sha384_init(context);
}

void SHA384FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha384_close(context, digest);
}

void SHA384UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha384(context, input, inputLen);
}


void SHA256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha256_context context = { 0 };

    SHA256InitContext(&context);
    SHA256UpdateHash(&context, input, inputLen);
    SHA256FinalHash(digest, &context);
}

void SHA256InitContext(void* context)
{
    sph_sha256_init(context);
}

void SHA256FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha256_close(context, digest);
}

void SHA256UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha256(context, input, inputLen);
}


void SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha1_context context = { 0 };

    SHA1InitContext(&context);
    SHA1UpdateHash(&context, input, inputLen);
    SHA1FinalHash(digest, &context);
}

void SHA1InitContext(void* context)
{
    sph_sha1_init(context);
}

void SHA1FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha1_close(context, digest);
}

void SHA1UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha1(context, input, inputLen);
}


void CRC32CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    Crc32Context context = { 0 };
    
    CRC32InitContext(&context);
    CRC32UpdateHash(&context, input, inputLen);
    CRC32FinalHash(digest, &context);
}

void CRC32InitContext(void* context)
{
    Crc32Init((Crc32Context*)context);
}

void CRC32FinalHash(apr_byte_t* digest, void* context)
{
    Crc32Final(digest, (Crc32Context*)context);
}

void CRC32UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    Crc32Update((Crc32Context*)context, input, inputLen);
}


void MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md2_context context = { 0 };

    MD2InitContext(&context);
    MD2UpdateHash(&context, input, inputLen);
    MD2FinalHash(digest, &context);
}

void MD2InitContext(void* context)
{
    sph_md2_init(context);
}

void MD2FinalHash(apr_byte_t* digest, void* context)
{
    sph_md2_close(context, digest);
}

void MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_md2(context, input, inputLen);
}

void MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md4_context context = { 0 };

    MD4InitContext(&context);
    MD4UpdateHash(&context, input, inputLen);
    MD4FinalHash(digest, &context);
}


void MD4InitContext(void* context)
{
    sph_md4_init(context);
}

void MD4FinalHash(apr_byte_t* digest, void* context)
{
    sph_md4_close(context, digest);
}

void MD4UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_md4(context, input, inputLen);
}


void MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md5_context context = { 0 };
    MD5InitContext(&context);
    MD5UpdateHash(&context, input, inputLen);
    MD5FinalHash(digest, &context);
}

void MD5InitContext(void* context)
{
    sph_md5_init(context);
}

void MD5FinalHash(apr_byte_t* digest, void* context)
{
    sph_md5_close(context, digest);
}

void MD5UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_md5(context, input, inputLen);
}

void TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_tiger_context context = { 0 };
    TIGERInitContext(&context);
    TIGERUpdateHash(&context, input, inputLen);
    TIGERFinalHash(digest, &context);
}

void TIGERInitContext(void* context)
{
    sph_tiger_init(context);
}

void TIGERFinalHash(apr_byte_t* digest, void* context)
{
    sph_tiger_close(context, digest);
}

void TIGERUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_tiger(context, input, inputLen);
}

void TIGER2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_tiger2_context context = { 0 };
    TIGER2InitContext(&context);
    TIGER2UpdateHash(&context, input, inputLen);
    TIGER2FinalHash(digest, &context);
}

void TIGER2InitContext(void* context)
{
    sph_tiger2_init(context);
}

void TIGER2FinalHash(apr_byte_t* digest, void* context)
{
    sph_tiger2_close(context, digest);
}

void TIGER2UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_tiger2(context, input, inputLen);
}

void SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha224_context context = { 0 };
    SHA224InitContext(&context);
    SHA224UpdateHash(&context, input, inputLen);
    SHA224FinalHash(digest, &context);
}

void SHA224InitContext(void* context)
{
    sph_sha224_init(context);
}

void SHA224FinalHash(apr_byte_t* digest, void* context)
{
    sph_sha224_close(context, digest);
}

void SHA224UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_sha224(context, input, inputLen);
}

void RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_ripemd128_context context = { 0 };

    RMD128InitContext(&context);
    RMD128UpdateHash(&context, input, inputLen);
    RMD128FinalHash(digest, &context);
}

void RMD128InitContext(void* context)
{
    sph_ripemd128_init(context);
}

void RMD128FinalHash(apr_byte_t* digest, void* context)
{
    sph_ripemd128_close(context, digest);
}

void RMD128UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_ripemd128(context, input, inputLen);
}

void RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_ripemd160_context context = { 0 };

    RMD160InitContext(&context);
    RMD160UpdateHash(&context, input, inputLen);
    RMD160FinalHash(digest, &context);
}

void RMD160InitContext(void* context)
{
    sph_ripemd160_init(context);
}

void RMD160FinalHash(apr_byte_t* digest, void* context)
{
    sph_ripemd160_close(context, digest);
}

void RMD160UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    sph_ripemd160(context, input, inputLen);
}

void RMD256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    LibtomCalculateDigest(digest, input, inputLen, rmd256_init, rmd256_process, rmd256_done);
}

void RMD256InitContext(void* context)
{
    LibtomInitContext(context, rmd256_init);
}

void RMD256FinalHash(apr_byte_t* digest, void* context)
{
    LibtomFinalHash(digest, context, rmd256_done);
}

void RMD256UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    LibtomUpdateHash(context, input, inputLen, rmd256_process);
}

void RMD320CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    LibtomCalculateDigest(digest, input, inputLen, rmd320_init, rmd320_process, rmd320_done);
}

void RMD320InitContext(void* context)
{
    LibtomInitContext(context, rmd320_init);
}

void RMD320FinalHash(apr_byte_t* digest, void* context)
{
    LibtomFinalHash(digest, context, rmd320_done);
}

void RMD320UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    LibtomUpdateHash(context, input, inputLen, rmd320_process);
}

void GOSTCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    gost_ctx context = { 0 };

    GOSTInitContext(&context);
    GOSTUpdateHash(&context, input, inputLen);
    GOSTFinalHash(digest, &context);
}

void GOSTInitContext(void* context)
{
    rhash_gost_cryptopro_init((gost_ctx*)context);
}

void GOSTFinalHash(apr_byte_t* digest, void* context)
{
    rhash_gost_final((gost_ctx*)context, digest);
}

void GOSTUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    rhash_gost_update((gost_ctx*)context, (apr_byte_t*)input, inputLen);
}