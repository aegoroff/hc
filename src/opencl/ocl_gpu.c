/*!
 * OpenCL backend for the GPU brute-force ABI (single preferred GPU device).
 * Symbols are compiled as ocl_* (see ocl_prefix.h). Public ABI is provided by
 * ocl_shim.c or gpu_dispatch.c (CUDA preferred when both backends are linked).
 */
#include "ocl_api.h"
#include "gpu_abi.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    hc_ocl_api_t api;
    int loaded;
    int usable;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    int max_blocks;
    int max_threads;
    int compute_units;
} ocl_runtime_t;

static ocl_runtime_t g_ocl;

static int ocl_info_str(hc_ocl_api_t* api, cl_device_id dev, cl_device_info info, char* buf, size_t n) {
    if (!buf || n == 0) return -1;
    buf[0] = 0;
    size_t sz = 0;
    if (api->clGetDeviceInfo(dev, info, 0, NULL, &sz) != CL_SUCCESS || sz == 0) return -1;
    if (sz > n) sz = n;
    if (api->clGetDeviceInfo(dev, info, sz, buf, NULL) != CL_SUCCESS) return -1;
    buf[n - 1] = 0;
    return 0;
}

static int ocl_platform_str(hc_ocl_api_t* api, cl_platform_id plat, cl_platform_info info, char* buf,
                            size_t n) {
    if (!buf || n == 0) return -1;
    buf[0] = 0;
    size_t sz = 0;
    if (api->clGetPlatformInfo(plat, info, 0, NULL, &sz) != CL_SUCCESS || sz == 0) return -1;
    if (sz > n) sz = n;
    if (api->clGetPlatformInfo(plat, info, sz, buf, NULL) != CL_SUCCESS) return -1;
    buf[n - 1] = 0;
    return 0;
}

static int ocl_score_device(hc_ocl_api_t* api, cl_platform_id plat, cl_device_id dev) {
    char vendor[256];
    char name[256];
    char pvendor[256];
    ocl_info_str(api, dev, CL_DEVICE_VENDOR, vendor, sizeof(vendor));
    ocl_info_str(api, dev, CL_DEVICE_NAME, name, sizeof(name));
    ocl_platform_str(api, plat, CL_PLATFORM_VENDOR, pvendor, sizeof(pvendor));

    int score = 1;
    /* Prefer Intel Arc / Intel GPU platforms. */
    if (strstr(vendor, "Intel") || strstr(pvendor, "Intel")) score += 100;
    if (strstr(name, "Arc") || strstr(name, "ARC")) score += 50;
    return score;
}

static int ocl_pick_device(hc_ocl_api_t* api, cl_device_id* out_dev) {
    cl_uint nplat = 0;
    if (api->clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) return -1;

    cl_platform_id plats[16];
    if (nplat > 16) nplat = 16;
    if (api->clGetPlatformIDs(nplat, plats, NULL) != CL_SUCCESS) return -1;

    int best_score = -1;
    cl_device_id best = NULL;

    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint ndev = 0;
        if (api->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev) != CL_SUCCESS || ndev == 0) {
            continue;
        }
        cl_device_id devs[16];
        if (ndev > 16) ndev = 16;
        if (api->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ndev, devs, NULL) != CL_SUCCESS) continue;

        for (cl_uint d = 0; d < ndev; ++d) {
            const int score = ocl_score_device(api, plats[p], devs[d]);
            if (score > best_score) {
                best_score = score;
                best = devs[d];
            }
        }
    }

    if (!best) return -1;
    *out_dev = best;
    return 0;
}

static int ocl_ensure(void) {
    if (g_ocl.loaded) return g_ocl.usable ? 0 : -1;

    g_ocl.loaded = 1;
    if (hc_ocl_api_load(&g_ocl.api) != 0) {
        g_ocl.usable = 0;
        return -1;
    }

    if (ocl_pick_device(&g_ocl.api, &g_ocl.device) != 0) {
        hc_ocl_api_unload(&g_ocl.api);
        g_ocl.usable = 0;
        return -1;
    }

    cl_int err = 0;
    g_ocl.context = g_ocl.api.clCreateContext(NULL, 1, &g_ocl.device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !g_ocl.context) {
        hc_ocl_api_unload(&g_ocl.api);
        g_ocl.usable = 0;
        return -1;
    }

    g_ocl.queue = g_ocl.api.clCreateCommandQueue(g_ocl.context, g_ocl.device, 0, &err);
    if (err != CL_SUCCESS || !g_ocl.queue) {
        g_ocl.api.clReleaseContext(g_ocl.context);
        g_ocl.context = NULL;
        hc_ocl_api_unload(&g_ocl.api);
        g_ocl.usable = 0;
        return -1;
    }

    cl_uint cus = 1;
    size_t wg = 256;
    g_ocl.api.clGetDeviceInfo(g_ocl.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    g_ocl.api.clGetDeviceInfo(g_ocl.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, NULL);

    g_ocl.compute_units = cus > 0 ? (int)cus : 1;
    g_ocl.max_threads = (int)(wg > 256 ? 256 : (wg > 0 ? wg : 64));
    /* Rough launch budget analogous to CUDA core*SM grid sizing. */
    g_ocl.max_blocks = g_ocl.compute_units * 32;
    if (g_ocl.max_blocks < 64) g_ocl.max_blocks = 64;

    g_ocl.usable = 1;
    return 0;
}

hc_ocl_api_t* hc_ocl_runtime_api(void) {
    if (ocl_ensure() != 0) return NULL;
    return &g_ocl.api;
}

cl_context hc_ocl_runtime_context(void) {
    if (ocl_ensure() != 0) return NULL;
    return g_ocl.context;
}

cl_command_queue hc_ocl_runtime_queue(void) {
    if (ocl_ensure() != 0) return NULL;
    return g_ocl.queue;
}

cl_device_id hc_ocl_runtime_device(void) {
    if (ocl_ensure() != 0) return NULL;
    return g_ocl.device;
}

void gpu_get_props(device_props_t* prop) {
    memset(prop, 0, sizeof(*prop));
    if (ocl_ensure() != 0) return;
    prop->device_count = 1;
}

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) {
    memset(prop, 0, sizeof(*prop));
    if (device_ix != 0 || ocl_ensure() != 0) return FALSE;
    prop->device_count = 1;
    prop->max_blocks_number = g_ocl.max_blocks;
    prop->max_threads_per_block = g_ocl.max_threads;
    prop->multiprocessor_count = g_ocl.compute_units;
    return TRUE;
}

BOOL gpu_can_use_gpu(void) {
    return ocl_ensure() == 0 ? TRUE : FALSE;
}

int gpu_driver_version(void) {
    /* OpenCL has no CUDA-style driver encoding; keep 0 so bf.zig skips the
     * CUDA version mismatch diagnostic. */
    return 0;
}

int gpu_runtime_version(void) {
    return 0;
}

gpu_versions_t gpu_number_to_version(int version_number) {
    gpu_versions_t version = { 0, 0 };
    version.major = version_number / 1000;
    version.minor = (version_number - version.major * 1000) / 10;
    return version;
}

BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx) {
    if (!ctx || ocl_ensure() != 0) return FALSE;
    ctx->stream_ = g_ocl.queue;
    ctx->launch_in_flight_ = FALSE;
    return TRUE;
}

void gpu_synchronize(gpu_tread_ctx_t* ctx) {
    if (!ctx || !g_ocl.usable || !g_ocl.queue) return;
    g_ocl.api.clFinish(g_ocl.queue);
    if (ctx->result_ && ctx->result_[0]) {
        ctx->found_in_the_thread_ = TRUE;
    }
}

#include "ocl_common.h"

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->launch_in_flight_) {
        gpu_synchronize(ctx);
        ctx->launch_in_flight_ = FALSE;
    }
    hc_ocl_run_active_cleanup();
    ctx->dev_result_ = NULL;
    ctx->stream_ = NULL;
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl)) {
    (void)dict_len;
    if (!ctx || !pfn_kernel) return;
    pfn_kernel(ctx, dict_len);
}
