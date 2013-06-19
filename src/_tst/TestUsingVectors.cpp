#include "TestUsingVectors.h"
#include "libtom.h"
#include "lib.h"
#include "output.h"
#include <stdio.h>
#include <tchar.h>

#define BYTE_CHARS_SIZE 2   // byte representation string length

bool TestUsingVectors::CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2, size_t sz)
{
    size_t i = 0;

    for (; i <= sz - (sz >> 2); i += 4) {
        if (digest1[i] != digest2[i]) {
            return FALSE;
        }
        if (digest1[i + 1] != digest2[i + 1]) {
            return FALSE;
        }
        if (digest1[i + 2] != digest2[i + 2]) {
            return FALSE;
        }
        if (digest1[i + 3] != digest2[i + 3]) {
            return FALSE;
        }
    }
    return TRUE;
}

void TestUsingVectors::ToDigest(const char* hash, apr_byte_t* digest, size_t sz)
{
    size_t i = 0;
    size_t to = MIN(sz, strlen(hash) / BYTE_CHARS_SIZE);

    for (; i < to; ++i) {
        digest[i] = (apr_byte_t)htoi(hash + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
    }
}


TEST_F(TestUsingVectors, Md2) {
    char* hashes[] = {
        "8350E5A3E24C153DF2275C9F80692773",
        "EE8DBAE3BC62BDC94EA63F69C1BC26C9",
        "1EAA4F494D81BC570FED4440EF3AC1C3",
        "54CDB6D1BF893171E7814DB84DF63A3A",
        "F71A82F8083CD9ABA3D0D651E2577EDA",
        "2F708334DBD1FE8F71CEE77E54B470F9",
        "E014DF2DF43498495056E7A437476A34",
        "9C410644446400B0F2C1B4697C443E19",
        "0944DEC40367AC855117012204018C9F",
        "CE8A6E797AC79D82D2C6D151F740CB33",
        "06DB4C310570268754114F747E1F0946",
        "9F323D5FC6DA86307BEBC0371A733787",
        "3C1C7E741794D3D4022DE17FCE72B283",
        "035D71AA96F782A9EB8D431E431672EE",
        "7ABE4067ED6CA42C79B542829434559C",
        "5E8D0D6F6F8E07C226AE9DD32609035A",
        "2B1632FF487D6C98AA3773B9D3FCD2AB",
        "D3D894482F7541BC0948B19842B479D9",
        "CFE6B872AC98304524CC6A88B6C45881",
        "1573DD015C8629DE9664CA0721473888",
        "ACFE2D3BB3CCAD8AEF6E37D0D8FBD634",
        "F5F83499AA172BE8344F7F39BA708AAA",
        "1D1C71FF6321B685D26F7FA620DA6C22",
        "4D7E74B6C8321775A34F7EFF38AAE5DF",
        "351A988C86AC5A10D0AB8E9071795181",
        "970F511C12E9CCD526EFF8574CF1467F",
        "0A68F53A476F7499EF79278A4EE8DAA3",
        "D458CF9C8CD0ABA23BD9A8C5ABE495CE",
        "C8002E85C3AD9B8B4AFD23378165C54B",
        "0B4788B157ED150A34D0E6E96BB4789C",
        "B14F4E31DE09281E07248A17939BE5B9",
        "803EEB99231526D6A33C8D4FCA537A6F",
        "51FE5D6637D2F0F09E48CE2A7F5030EA"
    };
    apr_byte_t expected[SZ_MD2];
    apr_byte_t bytes[SZ_MD2];
    int length = sizeof(hashes)/sizeof(hashes[0]);
    for (int i = 0; i < length; i++)
    {
        apr_byte_t* data = (apr_byte_t*)apr_pcalloc(pool_, 1);
        data[0] = (apr_byte_t)i;

        MD2CalculateDigest(bytes, data, 1);
        const char* result = HashToString(bytes, FALSE, SZ_MD2, pool_);
        ToDigest(hashes[i], expected, SZ_MD2);
        std::cout << std::endl;
        EXPECT_STREQ(hashes[i], result);
    }
}