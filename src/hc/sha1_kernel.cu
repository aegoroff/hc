/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains SHA-1 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "cuda_runtime.h"
#include <stdint.h>
#include "lib.h"

#define MAXPWDSIZE 10

/* f1 to f4 */

__device__ inline uint32_t f1(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) | (~x & z)); }
__device__ inline uint32_t f2(uint32_t x, uint32_t y, uint32_t z) { return (x ^ y ^ z); }
__device__ inline uint32_t f3(uint32_t x, uint32_t y, uint32_t z) { return ((x & y) | (x & z) | (y & z)); }
__device__ inline uint32_t f4(uint32_t x, uint32_t y, uint32_t z) { return (x ^ y ^ z); }

/* SHA init values */

__constant__ uint32_t I1 = 0x67452301L;
__constant__ uint32_t I2 = 0xEFCDAB89L;
__constant__ uint32_t I3 = 0x98BADCFEL;
__constant__ uint32_t I4 = 0x10325476L;
__constant__ uint32_t I5 = 0xC3D2E1F0L;

/* SHA constants */

__constant__ uint32_t C1 = 0x5A827999L;
__constant__ uint32_t C2 = 0x6Ed9EBA1L;
__constant__ uint32_t C3 = 0x8F1BBCDCL;
__constant__ uint32_t C4 = 0xCA62C1D6L;

/* 32-bit rotate */

__device__ inline uint32_t ROT(uint32_t x, int n) { return ((x << n) | (x >> (32 - n))); }

/* main function */

#define CALC(n,i) temp =  ROT ( A , 5 ) + f##n( B , C, D ) +  W[i] + E + C##n  ; E = D; D = C; C = ROT ( B , 30 ); B = A; A = temp


__device__ void prsha1_mem_init(unsigned int*, const unsigned char*, const int);
__device__ bool prsha1_compare(unsigned char* result, unsigned char* hash, unsigned char* password, const int length);
__global__ void sha1_kernel(unsigned char* result, unsigned char* hash, const int attempt_length, const char* dict, const size_t dict_length);

__shared__ short dev_found;


/*
* kernel-function __global__ void _sha1_kernel(int, char, in)
*
* Initialize with count of possible chars squared as the block-num
* and count of possible chars as the thread-num
* With cx (where cx is char at position x of the tested word) the
* first 3 chars are set like:
*
*   - c0: thread-num
*   - c1: block-num / 95
*   - c2: block-num % 95
*
* That guarantees every possible unique combination of the first
* the chars.
*
* input:
*   - attempt_length: length of the words
*   - result: buffer to write-back result, return value
*   - hash: hash that needs to be decrypted
*
*/
__global__ void sha1_kernel(unsigned char* result, unsigned char* hash, const int attempt_length, const char* dict, const size_t dict_length)
{
    unsigned char password[MAXPWDSIZE];

    // init input_cpy
    password[0] = dict[threadIdx.x];
    if (attempt_length > 1)
        password[1] = dict[(blockIdx.x / 95)];
    if (attempt_length > 2)
        password[2] = dict[(blockIdx.x % 95)];

    // HACK: attempt_length > 4
    if (dev_found || prsha1_compare(result, hash, password, attempt_length) || attempt_length <= 3 || attempt_length > 4) {
        return;
    }

    for (int i = 3; i < attempt_length; i++) {
        for (size_t j = 0; j < dict_length; j++) {
            password[i] = dict[j];
            if (dev_found || prsha1_compare(result, hash, password, attempt_length)) {
                return;
            }
        }
    }
}


__device__ bool prsha1_compare(unsigned char* result, unsigned char* hash, unsigned char* password, const int length) {
    // load into register
    const uint32_t h0 = (unsigned)hash[3] | (unsigned)hash[2] << 8 | (unsigned)hash[1] << 16 | (unsigned)hash[0] << 24;
    const uint32_t h1 = (unsigned)hash[7] | (unsigned)hash[6] << 8 | (unsigned)hash[5] << 16 | (unsigned)hash[4] << 24;
    const uint32_t h2 = (unsigned)hash[11] | (unsigned)hash[10] << 8 | (unsigned)hash[9] << 16 | (unsigned)hash[8] << 24;
    const uint32_t h3 = (unsigned)hash[15] | (unsigned)hash[14] << 8 | (unsigned)hash[13] << 16 | (unsigned)hash[12] << 24;
    const uint32_t h4 = (unsigned)hash[19] | (unsigned)hash[18] << 8 | (unsigned)hash[17] << 16 | (unsigned)hash[16] << 24;

    // Init words for SHA
    uint32_t W[80], temp;

    // Calculate sha for given input.
    // DO THE SHA ------------------------------------------------------

    prsha1_mem_init(W, password, length);
    for (int i = 16; i < 80; i++) {
        W[i] = ROT((W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16]), 1);
    }

    uint32_t A = I1;
    uint32_t B = I2;
    uint32_t C = I3;
    uint32_t D = I4;
    uint32_t E = I5;

    CALC(1, 0);  CALC(1, 1);  CALC(1, 2);  CALC(1, 3);  CALC(1, 4);
    CALC(1, 5);  CALC(1, 6);  CALC(1, 7);  CALC(1, 8);  CALC(1, 9);
    CALC(1, 10); CALC(1, 11); CALC(1, 12); CALC(1, 13); CALC(1, 14);
    CALC(1, 15); CALC(1, 16); CALC(1, 17); CALC(1, 18); CALC(1, 19);
    CALC(2, 20); CALC(2, 21); CALC(2, 22); CALC(2, 23); CALC(2, 24);
    CALC(2, 25); CALC(2, 26); CALC(2, 27); CALC(2, 28); CALC(2, 29);
    CALC(2, 30); CALC(2, 31); CALC(2, 32); CALC(2, 33); CALC(2, 34);
    CALC(2, 35); CALC(2, 36); CALC(2, 37); CALC(2, 38); CALC(2, 39);
    CALC(3, 40); CALC(3, 41); CALC(3, 42); CALC(3, 43); CALC(3, 44);
    CALC(3, 45); CALC(3, 46); CALC(3, 47); CALC(3, 48); CALC(3, 49);
    CALC(3, 50); CALC(3, 51); CALC(3, 52); CALC(3, 53); CALC(3, 54);
    CALC(3, 55); CALC(3, 56); CALC(3, 57); CALC(3, 58); CALC(3, 59);
    CALC(4, 60); CALC(4, 61); CALC(4, 62); CALC(4, 63); CALC(4, 64);
    CALC(4, 65); CALC(4, 66); CALC(4, 67); CALC(4, 68); CALC(4, 69);
    CALC(4, 70); CALC(4, 71); CALC(4, 72); CALC(4, 73); CALC(4, 74);
    CALC(4, 75); CALC(4, 76); CALC(4, 77); CALC(4, 78); CALC(4, 79);

    // That needs to be done, == with like (A + I1) =0 hash[0] 
    // is wrong all the time?!
    const uint32_t tmp1 = A + I1;
    const uint32_t tmp2 = B + I2;
    const uint32_t tmp3 = C + I3;
    const uint32_t tmp4 = D + I4;
    const uint32_t tmp5 = E + I5;

    // if result was found, copy to buffer
    if (tmp1 == h0 &&
        tmp2 == h1 &&
        tmp3 == h2 &&
        tmp4 == h3 &&
        tmp5 == h4)
    {
        for (int i = 0; i < length; i++) {
            result[i] = password[i];
        }
        dev_found = 1;
        return true;
    }

    return false;
}

/*
* device function __device__ void prsha1_mem_init(uint, uchar, int)
* Prepare word for sha-1 (expand, add length etc)
*/
__device__ void prsha1_mem_init(uint32_t* tmp, const unsigned char* input, const int length) {

    int stop = 0;
    // reseting tmp
    for(size_t i = 0; i < 80; i++) tmp[i] = 0;

    // fill tmp like: message char c0,c1,c2,...,cn,10000000,00...000
    for(size_t i = 0; i < length; i += 4) {
        for(size_t j = 0; j < 4; j++)
            if(i + j < length)
                tmp[i / 4] |= input[i + j] << (24 - j * 8);
            else {
                stop = 1;
                break;
            }
        if(stop)
            break;
    }
    tmp[length / 4] |= 0x80 << (24 - (length % 4) * 8); // Append 1 then zeros
    // Adding length as last value
    tmp[15] |= length * 8;
}
