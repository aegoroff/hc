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

extern __global__ void sha1_kernel(unsigned char* result, unsigned char* hash, const int attempt_length, const char* dict, const size_t dict_length);
extern __global__ void sha1_kernel2(unsigned char* result, unsigned char* hash, const int attempt_length, const char* dict, const size_t dict_length, char* variants);


void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, device_props_t* device_props, const char* dict, size_t dict_len, const char* hash) {
    unsigned char* dev_result = nullptr;
    char* dev_dict = nullptr;
    unsigned char* dev_hash;

    cudaMalloc(reinterpret_cast<void**>(&dev_result), ctx->pass_length_);
    cudaMalloc(reinterpret_cast<void**>(&dev_hash), DIGESTSIZE);
    cudaMalloc(reinterpret_cast<void**>(&dev_dict), dict_len + 1);
    cudaMemset(dev_result, 0x0, ctx->pass_length_);

    cudaMemcpy(dev_hash, hash, DIGESTSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dict, dict, dict_len + 1, cudaMemcpyHostToDevice);

    sha1_kernel<<<dict_len * dict_len, dict_len>>>(dev_result, dev_hash, ctx->pass_length_, dev_dict, dict_len);

    cudaDeviceSynchronize();

    cudaMemcpy(ctx->attempt_, dev_result, ctx->pass_length_, cudaMemcpyDeviceToHost);

    if(ctx->attempt_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }

    cudaFree(dev_result);
    cudaFree(dev_hash);
    cudaFree(dev_dict);
}

void sha1_run_on_gpu2(gpu_tread_ctx_t* ctx, const char* dict, size_t dict_len, const char* hash, char* variants, const int variants_size) {
    unsigned char* dev_result = nullptr;
    char* dev_dict = nullptr;
    unsigned char* dev_hash;
    char* dev_variants;

    cudaMalloc(&dev_result, ctx->pass_length_);
    cudaMalloc(&dev_hash, DIGESTSIZE);
    cudaMalloc(&dev_dict, dict_len + 1);
    
    cudaMemset(dev_result, 0x0, ctx->pass_length_);

    cudaMemcpy(dev_hash, hash, DIGESTSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dict, dict, dict_len + 1, cudaMemcpyHostToDevice);
    

    cudaMalloc(&dev_variants, variants_size * sizeof(char));

    cudaMemcpy(dev_variants, variants, variants_size * sizeof(int), cudaMemcpyHostToDevice);

    sha1_kernel2<<<ctx->dev_props_->max_blocks_number * 2, ctx->dev_props_->max_threads_per_block>>>(dev_result, dev_hash, ctx->pass_length_, dev_dict, dict_len, dev_variants);

    cudaDeviceSynchronize();

    cudaMemcpy(ctx->result_, dev_result, ctx->pass_length_, cudaMemcpyDeviceToHost);

    if(ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }

    cudaFree(dev_result);
    cudaFree(dev_hash);
    cudaFree(dev_dict);
    cudaFree(dev_variants);
}
