/*!
 * Runtime OpenCL ICD loader via dlopen — binary has no NEEDED libOpenCL.
 */
#include "ocl_api.h"

#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void* ocl_dlsym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void* ocl_dlopen(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void ocl_dlclose(void* lib) {
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

#ifdef _WIN32
static const char* const k_ocl_libs[] = {
    "OpenCL.dll",
};
#else
static const char* const k_ocl_libs[] = {
    "libOpenCL.so.1",
    "libOpenCL.so",
};
#endif

#define LOAD_FN(field, typ)                                             \
    do {                                                                \
        api->field = (typ)ocl_dlsym(api->lib, #field);                  \
        if (!api->field) {                                              \
            hc_ocl_api_unload(api);                                     \
            return -1;                                                  \
        }                                                               \
    } while (0)

int hc_ocl_api_load(hc_ocl_api_t* api) {
    if (!api) return -1;
    memset(api, 0, sizeof(*api));

    for (size_t i = 0; i < sizeof(k_ocl_libs) / sizeof(k_ocl_libs[0]); ++i) {
        api->lib = ocl_dlopen(k_ocl_libs[i]);
        if (api->lib) break;
    }
    if (!api->lib) return -1;

    LOAD_FN(clGetPlatformIDs, cl_int (*)(cl_uint, cl_platform_id*, cl_uint*));
    LOAD_FN(clGetPlatformInfo, cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*));
    LOAD_FN(clGetDeviceIDs, cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*));
    LOAD_FN(clGetDeviceInfo, cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*));
    LOAD_FN(clCreateContext,
            cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*), void*, cl_int*));
    LOAD_FN(clReleaseContext, cl_int (*)(cl_context));
    LOAD_FN(clCreateCommandQueue,
            cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*));
    LOAD_FN(clReleaseCommandQueue, cl_int (*)(cl_command_queue));
    LOAD_FN(clCreateProgramWithSource,
            cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*));
    LOAD_FN(clBuildProgram,
            cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*),
                       void*));
    LOAD_FN(clGetProgramBuildInfo,
            cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*));
    LOAD_FN(clReleaseProgram, cl_int (*)(cl_program));
    LOAD_FN(clCreateKernel, cl_kernel (*)(cl_program, const char*, cl_int*));
    LOAD_FN(clReleaseKernel, cl_int (*)(cl_kernel));
    LOAD_FN(clSetKernelArg, cl_int (*)(cl_kernel, cl_uint, size_t, const void*));
    LOAD_FN(clCreateBuffer, cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*));
    LOAD_FN(clReleaseMemObject, cl_int (*)(cl_mem));
    LOAD_FN(clEnqueueWriteBuffer,
            cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint,
                       const cl_event*, cl_event*));
    LOAD_FN(clEnqueueReadBuffer,
            cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*,
                       cl_event*));
    LOAD_FN(clEnqueueNDRangeKernel,
            cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*,
                       cl_uint, const cl_event*, cl_event*));
    LOAD_FN(clFinish, cl_int (*)(cl_command_queue));
    return 0;
}

void hc_ocl_api_unload(hc_ocl_api_t* api) {
    if (!api) return;
    if (api->lib) ocl_dlclose(api->lib);
    memset(api, 0, sizeof(*api));
}
