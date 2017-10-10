/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains base64 encode/decode implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-10-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <stdint.h>
#include <stdlib.h>
#include "b64.h"

static char encoding_table[] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9', '+', '/'
};

static char* decoding_table = NULL;
static size_t mod_table[] = { 0, 2, 1 };

void prb64_decoding_table(apr_pool_t* pool);

char* b64_encode(const unsigned char* data,
                 const size_t input_length,
                 size_t* output_length,
                 apr_pool_t* pool) {

    *output_length = 4 * ((input_length + 2) / 3);

    // Plus one char for trailing zero
    char* encoded_data = (char*)apr_pcalloc(pool, ((*output_length) + 1) * sizeof(char));
    if(encoded_data == NULL) {
        return NULL;
    }

    for(size_t i = 0, j = 0; i < input_length;) {
        const uint32_t octet_a = i < input_length ? (unsigned char)data[i++] : 0;
        const uint32_t octet_b = i < input_length ? (unsigned char)data[i++] : 0;
        const uint32_t octet_c = i < input_length ? (unsigned char)data[i++] : 0;

        const uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
    }

    for(size_t i = 0; i < mod_table[input_length % 3]; i++)
        encoded_data[*output_length - 1 - i] = '=';

    return encoded_data;
}

unsigned char* b64_decode(const char* data,
                          const size_t input_length,
                          size_t* output_length,
                          apr_pool_t* pool) {

    if(decoding_table == NULL) {
        prb64_decoding_table(pool);
    }

    if(input_length % 4 != 0) {
        return NULL;
    }

    *output_length = input_length / 4 * 3;
    if(data[input_length - 1] == '=')
        (*output_length)--;
    if(data[input_length - 2] == '=')
        (*output_length)--;

    unsigned char* decoded_data = (unsigned char*)apr_pcalloc(pool, *output_length);
    if(decoded_data == NULL) {
        return NULL;
    }

    for(size_t i = 0, j = 0; i < input_length;) {
        const uint32_t sextet_a = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        const uint32_t sextet_b = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        const uint32_t sextet_c = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        const uint32_t sextet_d = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];

        const uint32_t triple = (sextet_a << 3 * 6)
                + (sextet_b << 2 * 6)
                + (sextet_c << 1 * 6)
                + (sextet_d << 0 * 6);

        if(j < *output_length)
            decoded_data[j++] = (triple >> 2 * 8) & 0xFF;
        if(j < *output_length)
            decoded_data[j++] = (triple >> 1 * 8) & 0xFF;
        if(j < *output_length)
            decoded_data[j++] = (triple >> 0 * 8) & 0xFF;
    }

    return decoded_data;
}

void prb64_decoding_table(apr_pool_t* pool) {
    decoding_table = (char*)apr_pcalloc(pool, 256);

    for(int i = 0; i < 64; i++) {
        decoding_table[(unsigned char)encoding_table[i]] = i;
    }
}
