#include "ocl_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static hc_ocl_algo_t* g_active_algo;

void hc_ocl_set_active_algo(hc_ocl_algo_t* algo) {
    g_active_algo = algo;
}

void hc_ocl_run_active_cleanup(void) {
    if (g_active_algo) hc_ocl_algo_release_bufs(g_active_algo);
}

void hc_ocl_algo_entry_prepare(hc_ocl_algo_t* algo, const char* src, const char* kernel_name,
                               size_t hash_len, int pass_wide, int device_ix,
                               const unsigned char* dict, size_t dict_len, const unsigned char* hash,
                               gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    algo->pass_wide_arg = pass_wide;
    hc_ocl_set_active_algo(algo);
    (void)hc_ocl_algo_prepare(algo, src, kernel_name, dict, dict_len, hash, hash_len, ctx);
}

static void hc_ocl_active_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(g_active_algo, ctx, dict_len);
}

void hc_ocl_algo_entry_run(gpu_tread_ctx_t* ctx, size_t dict_len) {
    gpu_run(ctx, dict_len, &hc_ocl_active_run);
}

void hc_ocl_algo_release_bufs(hc_ocl_algo_t* algo) {
    if (!algo) return;
    hc_ocl_api_t* api = hc_ocl_runtime_api();
    if (!api) return;
    if (algo->result_buf) {
        api->clReleaseMemObject(algo->result_buf);
        algo->result_buf = NULL;
    }
    if (algo->found_buf) {
        api->clReleaseMemObject(algo->found_buf);
        algo->found_buf = NULL;
    }
    if (algo->dict_buf) {
        api->clReleaseMemObject(algo->dict_buf);
        algo->dict_buf = NULL;
    }
    if (algo->hash_buf) {
        api->clReleaseMemObject(algo->hash_buf);
        algo->hash_buf = NULL;
    }
}

static int hc_ocl_algo_ensure_program(hc_ocl_algo_t* algo, const char* src, const char* kernel_name) {
    if (algo->ready) return 0;

    hc_ocl_api_t* api = hc_ocl_runtime_api();
    cl_context ctx = hc_ocl_runtime_context();
    cl_device_id dev = hc_ocl_runtime_device();
    if (!api || !ctx || !dev) return -1;

    cl_int err = 0;
    size_t src_len = strlen(src);
    algo->program = api->clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS || !algo->program) return -1;

    err = api->clBuildProgram(algo->program, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        api->clGetProgramBuildInfo(algo->program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        if (log_sz > 1) {
            char* log = (char*)malloc(log_sz);
            if (log) {
                api->clGetProgramBuildInfo(algo->program, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
                fprintf(stderr, "OpenCL build log (%s):\n%s\n", kernel_name, log);
                free(log);
            }
        }
        api->clReleaseProgram(algo->program);
        algo->program = NULL;
        return -1;
    }

    algo->kernel = api->clCreateKernel(algo->program, kernel_name, &err);
    if (err != CL_SUCCESS || !algo->kernel) {
        api->clReleaseProgram(algo->program);
        algo->program = NULL;
        return -1;
    }

    algo->ready = 1;
    return 0;
}

int hc_ocl_algo_prepare(hc_ocl_algo_t* algo, const char* src, const char* kernel_name,
                        const unsigned char* dict, size_t dict_len, const unsigned char* hash,
                        size_t hash_len, gpu_tread_ctx_t* ctx) {
    if (!algo || !src || !kernel_name || !dict || !hash || !ctx) return -1;
    if (dict_len == 0 || dict_len > OCL_DICT_MAX || hash_len == 0) return -1;

    hc_ocl_api_t* api = hc_ocl_runtime_api();
    cl_context octx = hc_ocl_runtime_context();
    if (!api || !octx) return -1;
    if (hc_ocl_algo_ensure_program(algo, src, kernel_name) != 0) return -1;

    hc_ocl_algo_release_bufs(algo);

    cl_int err = 0;
    unsigned char dict_storage[OCL_DICT_MAX];
    memset(dict_storage, 0, sizeof(dict_storage));
    memcpy(dict_storage, dict, dict_len);

    algo->dict_buf =
        api->clCreateBuffer(octx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, OCL_DICT_MAX, dict_storage, &err);
    if (err != CL_SUCCESS) return -1;

    unsigned char hash_storage[64];
    if (hash_len > sizeof(hash_storage)) return -1;
    memcpy(hash_storage, hash, hash_len);
    algo->hash_buf =
        api->clCreateBuffer(octx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hash_len, hash_storage, &err);
    if (err != CL_SUCCESS) {
        hc_ocl_algo_release_bufs(algo);
        return -1;
    }

    algo->result_buf = api->clCreateBuffer(octx, CL_MEM_READ_WRITE, GPU_ATTEMPT_SIZE, NULL, &err);
    if (err != CL_SUCCESS) {
        hc_ocl_algo_release_bufs(algo);
        return -1;
    }

    int zero = 0;
    algo->found_buf =
        api->clCreateBuffer(octx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &zero, &err);
    if (err != CL_SUCCESS) {
        hc_ocl_algo_release_bufs(algo);
        return -1;
    }

    algo->hash_len = hash_len;
    ctx->dev_result_ = (unsigned char*)algo->result_buf;
    return 0;
}

void hc_ocl_algo_run(hc_ocl_algo_t* algo, gpu_tread_ctx_t* ctx, size_t dict_len) {
    if (!algo || !ctx || !algo->ready || !algo->kernel) return;

    hc_ocl_api_t* api = hc_ocl_runtime_api();
    cl_command_queue queue = hc_ocl_runtime_queue();
    if (!api || !queue) return;

    const uint32_t threads = (uint32_t)ctx->max_threads_per_block_;
    const uint32_t count = ctx->batch_count_;
    if (threads == 0 || count == 0) return;

    unsigned char zeros[GPU_ATTEMPT_SIZE];
    memset(zeros, 0, sizeof(zeros));
    api->clEnqueueWriteBuffer(queue, algo->result_buf, CL_TRUE, 0, GPU_ATTEMPT_SIZE, zeros, 0, NULL, NULL);
    int zero = 0;
    api->clEnqueueWriteBuffer(queue, algo->found_buf, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);

    const uint32_t min_len = ctx->passmin_ ? ctx->passmin_ : 1u;
    const cl_ulong start = (cl_ulong)ctx->index_start_;
    const cl_uint cl_count = count;
    const cl_uint pass_len = ctx->pass_length_;
    const cl_uint cl_dict_len = (cl_uint)dict_len;
    const cl_uint cl_min_len = min_len;

    api->clSetKernelArg(algo->kernel, 0, sizeof(cl_mem), &algo->result_buf);
    api->clSetKernelArg(algo->kernel, 1, sizeof(cl_mem), &algo->dict_buf);
    api->clSetKernelArg(algo->kernel, 2, sizeof(cl_mem), &algo->hash_buf);
    api->clSetKernelArg(algo->kernel, 3, sizeof(cl_mem), &algo->found_buf);
    api->clSetKernelArg(algo->kernel, 4, sizeof(cl_ulong), &start);
    api->clSetKernelArg(algo->kernel, 5, sizeof(cl_uint), &cl_count);
    api->clSetKernelArg(algo->kernel, 6, sizeof(cl_uint), &pass_len);
    api->clSetKernelArg(algo->kernel, 7, sizeof(cl_uint), &cl_dict_len);
    api->clSetKernelArg(algo->kernel, 8, sizeof(cl_uint), &cl_min_len);
    if (algo->pass_wide_arg) {
        const cl_uint use_wide = ctx->use_wide_pass_ ? 1u : 0u;
        api->clSetKernelArg(algo->kernel, 9, sizeof(cl_uint), &use_wide);
    }

    size_t local = threads;
    size_t global = ((size_t)count + local - 1) / local * local;
    if (api->clEnqueueNDRangeKernel(queue, algo->kernel, 1, NULL, &global, &local, 0, NULL, NULL) !=
        CL_SUCCESS) {
        return;
    }

    api->clEnqueueReadBuffer(queue, algo->result_buf, CL_FALSE, 0, GPU_ATTEMPT_SIZE, ctx->result_, 0, NULL,
                             NULL);
    ctx->launch_in_flight_ = TRUE;
}
