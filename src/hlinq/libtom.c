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
#include "libtom.h"

apr_status_t LibtomInitContext(void* context, int (* PfnInit)(hash_state* md))
{
    PfnInit((hash_state*)context);
    return APR_SUCCESS;
}

apr_status_t LibtomFinalHash(apr_byte_t* digest, void* context, int (* PfnDone)(hash_state*    md,
                                                                                unsigned char* hash))
{
    PfnDone((hash_state*)context, digest);
    return APR_SUCCESS;
}

apr_status_t LibtomUpdateHash(void*                    context,
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

    return APR_SUCCESS;
}



apr_status_t LibtomCalculateDigest(
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
    return APR_SUCCESS;
}

apr_status_t MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, md2_init, md2_process, md2_done);
}

apr_status_t MD2InitContext(void* context)
{
    return LibtomInitContext(context, md2_init);
}

apr_status_t MD2FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, md2_done);
}

apr_status_t MD2UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, md2_process);
}



apr_status_t TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, tiger_init, tiger_process, tiger_done);
}

apr_status_t TIGERInitContext(void* context)
{
    return LibtomInitContext(context, tiger_init);
}

apr_status_t TIGERFinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, tiger_done);
}

apr_status_t TIGERUpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, tiger_process);
}



apr_status_t SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, sha224_init, sha224_process, sha224_done);
}

apr_status_t SHA224InitContext(void* context)
{
    return LibtomInitContext(context, sha224_init);
}

apr_status_t SHA224FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, sha224_done);
}

apr_status_t SHA224UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, sha224_process);
}



apr_status_t RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, rmd128_init, rmd128_process, rmd128_done);
}

apr_status_t RMD128InitContext(void* context)
{
    return LibtomInitContext(context, rmd128_init);
}

apr_status_t RMD128FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, rmd128_done);
}

apr_status_t RMD128UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, rmd128_process);
}



apr_status_t RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, rmd160_init, rmd160_process, rmd160_done);
}

apr_status_t RMD160InitContext(void* context)
{
    return LibtomInitContext(context, rmd160_init);
}

apr_status_t RMD160FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, rmd160_done);
}

apr_status_t RMD160UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, rmd160_process);
}




apr_status_t RMD256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, rmd256_init, rmd256_process, rmd256_done);
}

apr_status_t RMD256InitContext(void* context)
{
    return LibtomInitContext(context, rmd256_init);
}

apr_status_t RMD256FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, rmd256_done);
}

apr_status_t RMD256UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, rmd256_process);
}




apr_status_t RMD320CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    return LibtomCalculateDigest(digest, input, inputLen, rmd320_init, rmd320_process, rmd320_done);
}

apr_status_t RMD320InitContext(void* context)
{
    return LibtomInitContext(context, rmd320_init);
}

apr_status_t RMD320FinalHash(apr_byte_t* digest, void* context)
{
    return LibtomFinalHash(digest, context, rmd320_done);
}

apr_status_t RMD320UpdateHash(void* context, const void* input, const apr_size_t inputLen)
{
    return LibtomUpdateHash(context, input, inputLen, rmd320_process);
}
