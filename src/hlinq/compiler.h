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
#include "apr_hash.h"
#include "..\srclib\lib.h"
#include "..\srclib\bf.h"
#include "..\srclib\traverse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Digest {
    apr_byte_t* Data;
    apr_size_t Size;
} Digest;

typedef enum HASH_ALGORITHM
{
    Undefined = -1,
    Md5,
    Sha1,
    Md4,
    Sha256,
    Sha384,
    Sha512,
    Whirlpool,
    Crc32
} HASH_ALGORITHM;

typedef enum ContextType
{
    Undefined = -1,
    File,
    String,
    Dir,
    Hash
} ContextType;

typedef struct StatementCtx {
    const char* Id;
    const char* Source;
    HASH_ALGORITHM HashAlgorithm;
    ContextType Type;
} StatementCtx;

typedef struct StringStatementContext {
    const char* String;
    HASH_ALGORITHM HashAlgorithm;
    BOOL BruteForce;
    int Min;
    int Max;
    apr_size_t HashLength;
    const char* Dictionary;
} StringStatementContext;

typedef struct FileStatementContext {
    const char* SearchRoot;
    const char* HashToSearch;
    const char* NameFilter;
    BOOL Recursively;
    int Limit;
    int Offset;
    HASH_ALGORITHM HashAlgorithm;
} FileStatementContext;

void InitProgram(BOOL onlyValidate, apr_pool_t* root);
void OpenStatement();
void CloseStatement();
void RegisterIdentifier(pANTLR3_UINT8 identifier, ContextType type);
BOOL CallAttiribute(pANTLR3_UINT8 identifier);
char* Trim(pANTLR3_UINT8 str);
void SetSource(pANTLR3_UINT8 str);
void AssignStrAttribute(int code, pANTLR3_UINT8 value);
void AssignIntAttribute(int code, pANTLR3_UINT8 value);
void SetHashAlgorithm(HASH_ALGORITHM algorithm);
void SetRecursively();
void SetBruteForce();
void* GetContext();
FileStatementContext* GetFileContext();
StringStatementContext* GetStringContext();

void RunString(DataContext* dataCtx);
void RunFile(DataContext* dataCtx);
apr_status_t CalculateFile(const char* pathToFile, DataContext* ctx, apr_pool_t* pool);

void SetMin(int value);
void SetMax(int value);
void SetLimit(int value);
void SetOffset(int value);
void SetDictionary(const char* value);
void SetName(const char* value);

void SetHashToSearch(const char* value, HASH_ALGORITHM algorithm);
void SetMd5ToSearch(const char* value);
void SetSha1ToSearch(const char* value);
void SetSha256ToSearch(const char* value);
void SetSha384ToSearch(const char* value);
void SetSha512ToSearch(const char* value);
void SetShaMd4ToSearch(const char* value);
void SetShaCrc32ToSearch(const char* value);
void SetShaWhirlpoolToSearch(const char* value);

Digest* Hash(
    const char* string,
    apr_size_t size, 
    apr_status_t (*fn)(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
    );
Digest* HashMD4(const char* string);
Digest* HashMD5(const char* string);
Digest* HashSHA1(const char* string);
Digest* HashSHA256(const char* string);
Digest* HashSHA384(const char* string);
Digest* HashSHA512(const char* string);
Digest* HashWhirlpool(const char* string);
Digest* HashCrc32(const char* string);
void OutputToConsole(OutputContext* ctx);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax);

#ifdef __cplusplus
}
#endif

#endif // COMPILER_HCALC_H_