#ifndef HC_OCL_RUNTIME_H_
#define HC_OCL_RUNTIME_H_

#include "ocl_api.h"

hc_ocl_api_t* hc_ocl_runtime_api(void);
cl_context hc_ocl_runtime_context(void);
cl_command_queue hc_ocl_runtime_queue(void);
cl_device_id hc_ocl_runtime_device(void);

#endif /* HC_OCL_RUNTIME_H_ */
