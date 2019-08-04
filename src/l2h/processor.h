/*!
 * \brief   The file contains l2h processor interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2019-08-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#ifndef LINQ2HASH_PROC_H_
#define LINQ2HASH_PROC_H_

#ifdef __cplusplus
extern "C" {
#endif

    void proc_init(apr_pool_t* pool);
    void proc_complete();
    BOOL proc_match_re(const char* pattern, const char* subject);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_PROC_H_
