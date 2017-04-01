/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains common HLINQ definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#ifndef LINQ2HASH__HC_H_
#define LINQ2HASH__HC_H_


#define APP_NAME "Hash Calculator " PRODUCT_VERSION


#ifdef __cplusplus
extern "C" {
#endif

void hc_print_copyright(void);
void hc_print_syntax(void* argtableS, void* argtableH, void* argtableF, void* argtableD);
void hc_print_cmd_syntax(void* argtable, void* end);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH__HC_H_
