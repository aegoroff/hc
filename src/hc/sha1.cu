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

#include "sha1.h"
#include "cuda_runtime.h"

#define DIGESTSIZE 20

extern __global__ void sha1_kernel(unsigned char* result, unsigned char* hash, const int attempt_length, const char* alphabet, const size_t abc_length);


void sha1_run_on_gpu(tread_ctx_t* ctx, const char* dict, const char* hash) {
    unsigned char* dev_result = NULL;
    char* dev_dict = NULL;
    unsigned char* dev_hash;
    size_t dict_length = strlen(dict);

    cudaMalloc((void**)&dev_result, ctx->pass_length_);
    cudaMalloc((void**)&dev_hash, DIGESTSIZE);
    cudaMalloc((void**)&dev_dict, dict_length + 1);
    cudaMemset(dev_result, 0x0, ctx->pass_length_);

    cudaMemcpy(dev_hash, hash, DIGESTSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dict, dict, dict_length + 1, cudaMemcpyHostToDevice);

    sha1_kernel <<<dict_length * dict_length, dict_length>>>(dev_result, dev_hash, ctx->pass_length_, dev_dict, dict_length);

    cudaDeviceSynchronize();

    cudaMemcpy(ctx->pass_, dev_result, ctx->pass_length_, cudaMemcpyDeviceToHost);

    cudaFree(dev_result);
    cudaFree(dev_hash);
    cudaFree(dev_dict);
}
