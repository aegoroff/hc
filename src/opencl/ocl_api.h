/*!
 * Minimal OpenCL 1.2 surface for dynamic loading (no link-time libOpenCL).
 * Types mirror the Khronos headers enough for our host glue; kernels use
 * the driver's online compiler.
 */
#ifndef HC_OCL_API_H_
#define HC_OCL_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_program_info;
typedef cl_uint cl_program_build_info;
typedef cl_bitfield cl_context_properties;

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1u << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VENDOR 0x0903
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_TYPE 0x1000
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_TRUE 1
#define CL_FALSE 0

typedef struct hc_ocl_api {
    void* lib;

    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (*clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                  void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
    cl_int (*clReleaseContext)(cl_context);
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    cl_int (*clReleaseCommandQueue)(cl_command_queue);
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                             void (*)(cl_program, void*), void*);
    cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    cl_int (*clReleaseProgram)(cl_program);
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*);
    cl_int (*clReleaseKernel)(cl_kernel);
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int (*clReleaseMemObject)(cl_mem);
    cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint,
                                   const cl_event*, cl_event*);
    cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint,
                                  const cl_event*, cl_event*);
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                     const size_t*, cl_uint, const cl_event*, cl_event*);
    cl_int (*clFinish)(cl_command_queue);
} hc_ocl_api_t;

/** Load ICD loader and resolve entry points. Returns 0 on success. */
int hc_ocl_api_load(hc_ocl_api_t* api);
void hc_ocl_api_unload(hc_ocl_api_t* api);

#ifdef __cplusplus
}
#endif

#endif /* HC_OCL_API_H_ */
