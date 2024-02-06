/*!
 * \brief   The file contains GPU related code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#include <stdio.h>
#include "cuda_runtime.h"
#include "gpu.h"

int static prgpu_get_cores_count(struct cudaDeviceProp devProp);

int static prgpu_get_sm_proc_count(struct cudaDeviceProp devProp);

void gpu_get_props(device_props_t* prop) {
    struct cudaDeviceProp device_prop;
    int n_dev_count = 0;
    
    CUDA_SAFE_CALL(cudaGetDeviceCount(&n_dev_count));

    prop->device_count = n_dev_count;
    prop->max_blocks_number = 0;
    prop->max_threads_per_block = 0;
    prop->multiprocessor_count = 0;

    for(int i = 0; i < n_dev_count; i++) {
        if(cudaSuccess != cudaGetDeviceProperties(&device_prop, i)) {
            prop->max_blocks_number += 64;
            prop->max_threads_per_block += 128;
            return;
        }

        prop->max_blocks_number += prgpu_get_cores_count(device_prop);
        prop->max_threads_per_block += device_prop.maxThreadsPerBlock;
        prop->multiprocessor_count += device_prop.multiProcessorCount;
    }
}

int prgpu_get_cores_count(struct cudaDeviceProp devProp) {
    return devProp.multiProcessorCount * prgpu_get_sm_proc_count(devProp);
}

int prgpu_get_sm_proc_count(struct cudaDeviceProp devProp) {
    switch (devProp.major) {
    case 2: // Fermi
    {
        if (devProp.minor == 1) return 48;
        return 32;
    }
        case 3: // Kepler
        return 192;
    case 5: // Maxwell
        return 128;
    case 6: // Pascal
    {
        if (devProp.minor == 1) return 128;
        if (devProp.minor == 0) return 64;
    }
    break;
    case 7: // Volta and Turing
        return 64;
    default:
        break;
    }

    return 16;
}

BOOL gpu_can_use_gpu() {
    int n_dev_count       = 0;
    const cudaError_t err = cudaGetDeviceCount(&n_dev_count);
    
    if (err != cudaSuccess) {
        return FALSE;
    }

    const int driver_ver  = gpu_driver_version();
    const int runtime_ver = gpu_runtime_version();

    return driver_ver >= runtime_ver && driver_ver > 0;
}

int gpu_driver_version() {
    int ver;
    const cudaError_t err = cudaDriverGetVersion(&ver);

    if (err != cudaSuccess) {
        return 0;
    }

    return ver;
}

int gpu_runtime_version() {
    int ver;
    const cudaError_t err = cudaRuntimeGetVersion(&ver);

    if (err != cudaSuccess) {
        return 0;
    }

    return ver;
}

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaFree(ctx->dev_result_));
    CUDA_SAFE_CALL(cudaFree(ctx->dev_variants_));
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size, void(*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl)) {
    size_t k_result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero

    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx->dev_variants_, variants, variants_size * sizeof(unsigned char), cudaMemcpyHostToDevice, 0));
    CUDA_SAFE_CALL(cudaMemset(ctx->dev_result_, 0x0, k_result_size_in_bytes));

#ifdef MEASURE_CUDA
    cudaEvent_t start;
    cudaEvent_t finish;

    lib_printf("\nVariants memory (bytes): %lli\n", variants_size);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&finish));

    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
#endif

    pfn_kernel(ctx, ctx->dev_result_, ctx->dev_variants_, dict_len);

#ifdef MEASURE_CUDA
    CUDA_SAFE_CALL(cudaEventRecord(finish, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(finish));

    float elapsed;

    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, finish));

    lib_printf("\nCUDA Kernel time: %3.1f ms", elapsed);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(finish));
#endif

    CUDA_SAFE_CALL(cudaMemcpy(ctx->result_, ctx->dev_result_, k_result_size_in_bytes, cudaMemcpyDeviceToHost));

    // IMPORTANT: Do not move this validation into outer scope
    // it's strange but without this call result will be undefined
    if (ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }
}