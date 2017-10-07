/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains GPU related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_GPU_H_
#define LINQ2HASH_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct device_props_t {
        int device_count;
        int max_blocks_number;
        int max_threads_per_block;
    } device_props_t;

    void gpu_get_props(device_props_t* prop);

#ifdef __cplusplus
}
#endif


#endif // LINQ2HASH_GPU_H_