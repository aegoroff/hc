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
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "cuda_runtime.h"
#include "gpu.h"

void gpu_get_props(device_props_t* prop) {
    cudaDeviceProp device_prop;
    int n_dev_count;

    if(cudaSuccess != cudaGetDeviceCount(&n_dev_count)) {
        return;
    }

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
