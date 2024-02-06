/*!
 * \brief   The file contains common HLINQ definitions and interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#ifndef LINQ2HASH__HC_H_
#define LINQ2HASH__HC_H_

#define APP_NAME "Hash Calculator " PRODUCT_VERSION

#ifdef __cplusplus
extern "C" {
#endif

void hc_print_copyright(void);
void hc_print_syntax(void* argtable_s, void* argtable_h, void* argtable_f, void* argtable_d);
void hc_print_cmd_syntax(void* argtable, void* end);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH__HC_H_
