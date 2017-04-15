/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains hashes test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#pragma once

#include "gtest.h"
#include <tchar.h>
#include <apr_pools.h>
#include "hashes.h"
#include <apr_hash.h>

#ifdef __cplusplus

extern "C" {
#endif

    static apr_pool_t* pool_;
    static apr_hash_t* htest_algorithms = NULL;

    class HashTest : public ::testing::TestWithParam<const char*> {
    protected:
        virtual void SetUp() override;
        virtual void TearDown() override;
    public:
        static const char* GetHash(const char* algorithm);

        static void TearDownTestCase() {
            apr_pool_destroy(pool_);
            apr_terminate();
        }

        static void SetUpTestCase() {
            auto argc = 1;

            const char* const argv[] = { "1" };

            auto status = apr_app_initialize(&argc, (const char *const **)&argv, nullptr);

            if(status != APR_SUCCESS) {
                throw status;
            }
            apr_pool_create(&pool_, NULL);
            hsh_initialize_hashes(pool_);
            htest_algorithms = apr_hash_make(pool_);

            apr_hash_set(htest_algorithms, "crc32", APR_HASH_KEY_STRING, "884863D2");
            apr_hash_set(htest_algorithms, "edonr256", APR_HASH_KEY_STRING, "2DBADC39B5189B24479A766F87AC68DA5CB0C0AFF5D692DF3CECAB7B4F423CF1");
            apr_hash_set(htest_algorithms, "edonr512", APR_HASH_KEY_STRING, "9A40FA8740E3E0E6475B83BABF1B78B1A38AC3F8DB081723C53E611F2513D68C52BDF641BCC856D7321ACE59FC5181ECC0D5CA6A311D7DF4C7FA80CE4DF8FBA5");
            apr_hash_set(htest_algorithms, "gost", APR_HASH_KEY_STRING, "5EF18489617BA2D8D2D7E0DA389AAA4FF022AD01A39512A4FEA1A8C45E439148");
            apr_hash_set(htest_algorithms, "haval-128-3", APR_HASH_KEY_STRING, "BDC9FC6D0E82C40FA3DE3FD54803DBD1");
            apr_hash_set(htest_algorithms, "haval-128-4", APR_HASH_KEY_STRING, "7FD91A17538880FB2007F59A49B1C5A5");
            apr_hash_set(htest_algorithms, "haval-128-5", APR_HASH_KEY_STRING, "092356CE125C84828EA26E633328EF0B");
            apr_hash_set(htest_algorithms, "haval-160-3", APR_HASH_KEY_STRING, "9AA8070C350A5B8E9EF84D50C501488DCD209D89");
            apr_hash_set(htest_algorithms, "haval-160-4", APR_HASH_KEY_STRING, "7F21296963CC57E11A3DF4EC10BC79A4489125B8");
            apr_hash_set(htest_algorithms, "haval-160-5", APR_HASH_KEY_STRING, "8FF0C07890BE1CD2388DB65C85DA7B6C34E8A3D1");
            apr_hash_set(htest_algorithms, "haval-192-3", APR_HASH_KEY_STRING, "B00150CCD88C4404BBB4DE1D044D22CDE1D0AF78BFCFE911");
            apr_hash_set(htest_algorithms, "haval-192-4", APR_HASH_KEY_STRING, "47E4674075CB59C43DFF566B98B40F62F2652B5697B89C28");
            apr_hash_set(htest_algorithms, "haval-192-5", APR_HASH_KEY_STRING, "575C8E28A5BCFBC10179020D70C6C367280B40FC7AD806C3");
            apr_hash_set(htest_algorithms, "haval-224-3", APR_HASH_KEY_STRING, "A294D60D7351B4BC2E5962F5FF5A620B430B5069F27923E70D8AFBF0");
            apr_hash_set(htest_algorithms, "haval-224-4", APR_HASH_KEY_STRING, "B9E3BCFBC5EA72626CACFBEB0E055CB89ADF2CE9B0E24A3C8A32CB34");
            apr_hash_set(htest_algorithms, "haval-224-5", APR_HASH_KEY_STRING, "FC2D1B6F27FB775D8E7030715AF85B646239C9D9D675CCFF309B49B7");
            apr_hash_set(htest_algorithms, "haval-256-3", APR_HASH_KEY_STRING, "E3891CB6FD1A883A1AE723F13BA336F586FA8C10506C4799C209D10113675BC1");
            apr_hash_set(htest_algorithms, "haval-256-4", APR_HASH_KEY_STRING, "A16D7FCD48CED7B612FF2C35D78241EB89A752EFF2931647A32C2C3C22F8D747");
            apr_hash_set(htest_algorithms, "haval-256-5", APR_HASH_KEY_STRING, "386DBED5748A4B9E9409D8CE94ACFE8DF324A166EAC054E9817F85F7AEC8AED5");
            apr_hash_set(htest_algorithms, "md2", APR_HASH_KEY_STRING, "EF1FEDF5D32EAD6B7AAF687DE4ED1B71");
            apr_hash_set(htest_algorithms, "md4", APR_HASH_KEY_STRING, "C58CDA49F00748A3BC0FCFA511D516CB");
            apr_hash_set(htest_algorithms, "md5", APR_HASH_KEY_STRING, "202CB962AC59075B964B07152D234B70");
            apr_hash_set(htest_algorithms, "ntlm", APR_HASH_KEY_STRING, "C58CDA49F00748A3BC0FCFA511D516CB");
            apr_hash_set(htest_algorithms, "ripemd128", APR_HASH_KEY_STRING, "781F357C35DF1FEF3138F6D29670365A");
            apr_hash_set(htest_algorithms, "ripemd160", APR_HASH_KEY_STRING, "E3431A8E0ADBF96FD140103DC6F63A3F8FA343AB");
            apr_hash_set(htest_algorithms, "ripemd256", APR_HASH_KEY_STRING, "8536753AD7BFACE2DBA89FB318C95B1B42890016057D4C3A2F351CEC3ACBB28B");
            apr_hash_set(htest_algorithms, "ripemd320", APR_HASH_KEY_STRING, "BFA11B73AD4E6421A8BA5A1223D9C9F58A5AD456BE98BEE5BFCD19A3ECDC6140CE4C700BE860FDA9");
            apr_hash_set(htest_algorithms, "sha-3-224", APR_HASH_KEY_STRING, "602BDC204140DB016BEE5374895E5568CE422FABE17E064061D80097");
            apr_hash_set(htest_algorithms, "sha-3-256", APR_HASH_KEY_STRING, "A03AB19B866FC585B5CB1812A2F63CA861E7E7643EE5D43FD7106B623725FD67");
            apr_hash_set(htest_algorithms, "sha-3-384", APR_HASH_KEY_STRING, "9BD942D1678A25D029B114306F5E1DAE49FE8ABEEACD03CFAB0F156AA2E363C988B1C12803D4A8C9BA38FDC873E5F007");
            apr_hash_set(htest_algorithms, "sha-3-512", APR_HASH_KEY_STRING, "48C8947F69C054A5CAA934674CE8881D02BB18FB59D5A63EEADDFF735B0E9801E87294783281AE49FC8287A0FD86779B27D7972D3E84F0FA0D826D7CB67DFEFC");
            apr_hash_set(htest_algorithms, "sha-3k-224", APR_HASH_KEY_STRING, "5C52615361CE4C5469F9D8C90113C7A543A4BF43490782D291CB32D8");
            apr_hash_set(htest_algorithms, "sha-3k-256", APR_HASH_KEY_STRING, "64E604787CBF194841E7B68D7CD28786F6C9A0A3AB9F8B0A0E87CB4387AB0107");
            apr_hash_set(htest_algorithms, "sha-3k-384", APR_HASH_KEY_STRING, "7DD34CCAAE92BFC7EB541056D200DB23B6BBEEFE95BE0D2BB43625113361906F0AFC701DBEF1CFB615BF98B1535A84C1");
            apr_hash_set(htest_algorithms, "sha-3k-512", APR_HASH_KEY_STRING, "8CA32D950873FD2B5B34A7D79C4A294B2FD805ABE3261BEB04FAB61A3B4B75609AFD6478AA8D34E03F262D68BB09A2BA9D655E228C96723B2854838A6E613B9D");
            apr_hash_set(htest_algorithms, "sha1", APR_HASH_KEY_STRING, "40BD001563085FC35165329EA1FF5C5ECBDBBEEF");
            apr_hash_set(htest_algorithms, "sha224", APR_HASH_KEY_STRING, "78D8045D684ABD2EECE923758F3CD781489DF3A48E1278982466017F");
            apr_hash_set(htest_algorithms, "sha256", APR_HASH_KEY_STRING, "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3");
            apr_hash_set(htest_algorithms, "sha384", APR_HASH_KEY_STRING, "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F");
            apr_hash_set(htest_algorithms, "sha512", APR_HASH_KEY_STRING, "3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2");
            apr_hash_set(htest_algorithms, "snefru128", APR_HASH_KEY_STRING, "ED592424402DBDC9190D700A696EEB6A");
            apr_hash_set(htest_algorithms, "snefru256", APR_HASH_KEY_STRING, "9A26D1977B322678918E6C3EF1D8291A5A1DCF1AF2FC363DA1666D5422D0A1DE");
            apr_hash_set(htest_algorithms, "tiger", APR_HASH_KEY_STRING, "A86807BB96A714FE9B22425893E698334CD71E36B0EEF2BE");
            apr_hash_set(htest_algorithms, "tiger2", APR_HASH_KEY_STRING, "598B54A953F0ABF9BA647793A3C7C0C4EB8A68698F3594F4");
            apr_hash_set(htest_algorithms, "tth", APR_HASH_KEY_STRING, "E091CFC8F2BC148030F99CBF276B45481ED525CA31EB2EB5");
            apr_hash_set(htest_algorithms, "whirlpool", APR_HASH_KEY_STRING, "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F");
            apr_hash_set(htest_algorithms, "blake2b", APR_HASH_KEY_STRING, "E64CB91C7C1819BDCDA4DCA47A2AAE98E737DF75DDB0287083229DC0695064616DF676A0C95AE55109FE0A27BA9DEE79EA9A5C9D90CCEB0CF8AE80B4F61AB4A3");
            apr_hash_set(htest_algorithms, "blake2s", APR_HASH_KEY_STRING, "E906644AD861B58D47500E6C636EE3BF4CB4BB00016BB352B1D2D03D122C1605");
        }
    };

#ifdef __cplusplus
}
#endif
