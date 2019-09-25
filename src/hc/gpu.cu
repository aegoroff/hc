/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains GPU related code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include <stdio.h>
#include "cuda_runtime.h"
#include "gpu.h"

void gpu_get_props(device_props_t* prop) {
    struct cudaDeviceProp device_prop;
    int n_dev_count = 0;
    
    CUDA_SAFE_CALL(cudaGetDeviceCount(&n_dev_count));

    prop->device_count = n_dev_count;
    prop->max_blocks_number = 0;
    prop->max_threads_per_block = 0;

    for(int i = 0; i < n_dev_count; i++) {
        if(cudaSuccess != cudaGetDeviceProperties(&device_prop, i)) {
            prop->max_blocks_number += 64;
            prop->max_threads_per_block += 128;
            return;
        }
        prop->max_blocks_number += device_prop.multiProcessorCount;
        prop->max_threads_per_block += device_prop.maxThreadsPerBlock;
    }
}

BOOL gpu_can_use_gpu() {
    int n_dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&n_dev_count);

    if (err != cudaSuccess) {
        return FALSE;
    }

    return TRUE;
}

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaFreeHost(ctx->variants_));
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size, void(*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl)) {
    unsigned char* dev_result = nullptr;
    unsigned char* dev_variants = nullptr;

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_variants), variants_size * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpyAsync(dev_variants, variants, variants_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result), result_size_in_bytes));
    CUDA_SAFE_CALL(cudaMemset(dev_result, 0x0, result_size_in_bytes));

#ifdef MEASURE_CUDA
    cudaEvent_t start;
    cudaEvent_t finish;

    lib_printf("\nVariants memory (bytes): %lli\n", variants_size);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&finish));

    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
#endif

    pfn_kernel(ctx, dev_result, dev_variants, dict_len);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef MEASURE_CUDA
    CUDA_SAFE_CALL(cudaEventRecord(finish, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(finish));

    float elapsed;

    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, finish));

    lib_printf("\nCUDA Kernel time: %3.1f ms", elapsed);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(finish));
#endif

    CUDA_SAFE_CALL(cudaMemcpy(ctx->result_, dev_result, result_size_in_bytes, cudaMemcpyDeviceToHost));

    // IMPORTANT: Do not move this validation into outer scope
    // it's strange but without this call result will be undefined
    if (ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }

    CUDA_SAFE_CALL(cudaFree(dev_result));
    CUDA_SAFE_CALL(cudaFree(dev_variants));
}