/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains configuration module interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-09-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH_CONFIGURATION_H_
#define LINQ2HASH_CONFIGURATION_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct configuration_ctx_t {
    void (*on_string)(const char* const str);
    void (*on_file)(struct arg_file* files);
    int argc;
    const char* const* argv;
} configuration_ctx_t;

void conf_configure_app(configuration_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_CONFIGURATION_H_
