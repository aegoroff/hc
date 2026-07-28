/*!
 * \brief   The file contains GPU related code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"

int static prgpu_get_cores_count(struct cudaDeviceProp devProp);

int static prgpu_get_sm_proc_count(struct cudaDeviceProp devProp);

void gpu_get_props(device_props_t* prop) {
    memset(prop, 0, sizeof(*prop));
    int n_dev_count = 0;
    if (cudaGetDeviceCount(&n_dev_count) == cudaSuccess) {
        prop->device_count = n_dev_count;
    }
}

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) {
    memset(prop, 0, sizeof(*prop));

    int n_dev_count = 0;
    if (cudaGetDeviceCount(&n_dev_count) != cudaSuccess || device_ix < 0 || device_ix >= n_dev_count) {
        return FALSE;
    }

    prop->device_count = n_dev_count;

    struct cudaDeviceProp device_prop;
    if (cudaGetDeviceProperties(&device_prop, device_ix) != cudaSuccess) {
        prop->max_blocks_number = 64;
        prop->max_threads_per_block = 128;
        prop->multiprocessor_count = 1;
        return TRUE;
    }

    prop->max_blocks_number = prgpu_get_cores_count(device_prop);
    prop->max_threads_per_block = device_prop.maxThreadsPerBlock;
    prop->multiprocessor_count = device_prop.multiProcessorCount;
    return TRUE;
}

int prgpu_get_cores_count(struct cudaDeviceProp devProp) {
    return devProp.multiProcessorCount * prgpu_get_sm_proc_count(devProp);
}

int prgpu_get_sm_proc_count(struct cudaDeviceProp devProp) {
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) return 48;
        return 32;
    case 3: // Kepler
        return 192;
    case 5: // Maxwell
        return 128;
    case 6: // Pascal
        if (devProp.minor == 0) return 64;
        return 128;
    case 7: // Volta and Turing
        return 64;
    case 8: // Ampere (8.0 A100 = 64; 8.6/8.9 consumer/Ada = 128)
        if (devProp.minor == 0) return 64;
        return 128;
    case 9: // Hopper
        return 128;
    case 10: // Blackwell (provisional)
        return 128;
    default:
        break;
    }

    /* Prefer a modern default over the historical 16 — under-sizing the grid
     * on unknown future SMs is worse than a modest over-subscribe. */
    if (devProp.major > 10) return 128;
    return 16;
}

BOOL gpu_can_use_gpu() {
    int n_dev_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&n_dev_count);

    if (err != cudaSuccess || n_dev_count <= 0) {
        return FALSE;
    }

    const int driver_ver = gpu_driver_version();
    const int runtime_ver = gpu_runtime_version();

    if (driver_ver <= 0) {
        return FALSE;
    }

    /* Official rule: driver >= toolkit used to build. Same-major soft allow
     * covers packaging quirks where minor encoding looks inverted in logs
     * (e.g. classic "13.3 < required 13.2") but the driver still runs kernels. */
    if (runtime_ver > 0 && driver_ver < runtime_ver) {
        const gpu_versions_t d = gpu_number_to_version(driver_ver);
        const gpu_versions_t r = gpu_number_to_version(runtime_ver);
        if (d.major != r.major || d.major == 0) {
            return FALSE;
        }
    }

    return TRUE;
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

gpu_versions_t gpu_number_to_version(int version_number) {
    gpu_versions_t version = { 0 };
    version.major = version_number / 1000;
    version.minor = (version_number - version.major * 1000) / 10;
    return version;
}

BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx) {
    if (!ctx) {
        return FALSE;
    }

    CUDA_SAFE_CALL(cudaSetDevice(ctx->device_ix_));

    cudaStream_t stream = NULL;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));
    ctx->stream_ = stream;

    /* Index-gen path: no host variant buffers. Keep slots NULL. */
    ctx->variants_bufs_[0] = NULL;
    ctx->variants_bufs_[1] = NULL;
    ctx->variants_ = NULL;
    ctx->fill_buf_ix_ = 0;
    ctx->launch_in_flight_ = FALSE;
    return TRUE;
}

void gpu_synchronize(gpu_tread_ctx_t* ctx) {
    if (!ctx || !ctx->stream_) {
        return;
    }
    CUDA_SAFE_CALL(cudaStreamSynchronize((cudaStream_t)ctx->stream_));
    /* Host result_ must be read only after the stream catches up — otherwise
     * found_in_the_thread_ is a stale/racy view of device memory. */
    if (ctx->result_ && ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }
}

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    if (ctx->launch_in_flight_) {
        gpu_synchronize(ctx);
        ctx->launch_in_flight_ = FALSE;
    }

    if (ctx->stream_) {
        CUDA_SAFE_CALL(cudaStreamDestroy((cudaStream_t)ctx->stream_));
        ctx->stream_ = NULL;
    }

    for (int i = 0; i < 2; ++i) {
        if (ctx->variants_bufs_[i]) {
            CUDA_SAFE_CALL(cudaFreeHost(ctx->variants_bufs_[i]));
            ctx->variants_bufs_[i] = NULL;
        }
    }
    ctx->variants_ = NULL;

    CUDA_SAFE_CALL(cudaFree(ctx->dev_result_));
    if (ctx->dev_variants_) {
        CUDA_SAFE_CALL(cudaFree(ctx->dev_variants_));
    }
    ctx->dev_result_ = NULL;
    ctx->dev_variants_ = NULL;
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size, void(*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl)) {
    (void)variants;
    (void)variants_size;
    size_t k_result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    cudaStream_t stream = GPU_STREAM(ctx);

    CUDA_SAFE_CALL(cudaMemsetAsync(ctx->dev_result_, 0x0, k_result_size_in_bytes, stream));

#ifdef MEASURE_CUDA
    cudaEvent_t start;
    cudaEvent_t finish;

    lib_printf("\nIndex batch: start=%llu count=%u len=%u\n",
               (unsigned long long)ctx->index_start_, ctx->batch_count_, ctx->pass_length_);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&finish));

    CUDA_SAFE_CALL(cudaEventRecord(start, stream));
#endif

    pfn_kernel(ctx, ctx->dev_result_, ctx->dev_variants_, dict_len);

#ifdef MEASURE_CUDA
    CUDA_SAFE_CALL(cudaEventRecord(finish, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(finish));

    float elapsed;

    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, finish));

    lib_printf("\nCUDA Kernel time: %3.1f ms", elapsed);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(finish));
#endif

    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx->result_, ctx->dev_result_, k_result_size_in_bytes, cudaMemcpyDeviceToHost, stream));
    /* Completion + found flag: gpu_synchronize (bf_core waits before next launch). */
}
