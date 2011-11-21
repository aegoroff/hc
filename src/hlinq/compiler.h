/*!
 * \brief   The file contains HLINQ compiler API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2011
 */

#ifndef COMPILER_HCALC_H_
#define COMPILER_HCALC_H_

#include    <antlr3.h>
#include "apr.h"
#include "apr_pools.h"
#include "apr_strings.h"
#include "apr_file_io.h"
#include "apr_hash.h"
#define SPECIAL_STR_ID "__str__"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Digest {
    apr_byte_t* Data;
    apr_size_t Size;
} Digest;

typedef struct OutputContext {
    int         IsPrintSeparator;
    int         IsFinishLine;
    const char* StringToPrint;
} OutputContext;

typedef struct DataContext {
    int         IsPrintLowCase;
    int         IsPrintCalcTime;
    const char* HashToSearch;
    apr_off_t   Limit;
    apr_off_t   Offset;
    apr_file_t* FileToSave;
    void        (* PfnOutput)(OutputContext* ctx);
} DataContext;


typedef enum HASH_ALGORITHM
{
    Md5,
    Sha1,
    Md4,
    Sha256,
    Sha384,
    Sha512,
    Whirlpool,
    Crc32
} HASH_ALGORITHM;

typedef struct StatementContext {
    const char* String;
    HASH_ALGORITHM HashAlgorithm;
    const char* SearchRoot;
    const char* ActionTarget;
    BOOL Recursively;
} StatementContext;

void InitProgram(BOOL onlyValidate, apr_pool_t* root);
void OpenStatement();
void CloseStatement(const char* identifier);
void CreateStatementContext(const char* identifier);
BOOL CallAttiribute(pANTLR3_UINT8 identifier);
void SetActionTarget(pANTLR3_UINT8 str, const char* identifier);
char* Trim(pANTLR3_UINT8 str);
void SetSearchRoot(pANTLR3_UINT8 str, const char* identifier);
void SetString(const char* str);
void SetHashAlgorithm(HASH_ALGORITHM algorithm);
void SetRecursively(const char* identifier);

const char* HashToString(apr_byte_t* digest, int isPrintLowCase, apr_size_t sz);
void        OutputDigest(apr_byte_t* digest, DataContext* ctx, apr_size_t sz);

void CalculateStringHash(
    const char* string, 
    apr_byte_t* digest, 
    apr_status_t (*fn)(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
    );

void CalculateStringHashMD5(const char* string,  apr_byte_t* digest);
void CalculateStringHashSHA1(const char* string,  apr_byte_t* digest);

Digest* HashMD5(const char* string);
Digest* HashSHA1(const char* string);
void OutputToConsole(OutputContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_