/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2013 Alexander Egorov
*/

namespace _tst.net
{

    #region Abstracts

    public abstract class HashCalculatorFileTestsWin64<T> : HashCalculatorFileTests<ArchWin64, T> where T : Hash, new()
    {
    }

    public abstract class HashQueryFileTestsWin64<T> : HashQueryFileTests<ArchWin64, T> where T : Hash, new()
    {
    }

    public abstract class HashCalculatorFileTestsWin32<T> : HashCalculatorFileTests<ArchWin32, T> where T : Hash, new()
    {
    }

    public abstract class HashQueryFileTestsWin32<T> : HashQueryFileTests<ArchWin32, T> where T : Hash, new()
    {
    }

    public abstract class HashCalculatorStringTestsWin64<T> : HashCalculatorStringTests<ArchWin64, T> where T : Hash, new()
    {
    }

    public abstract class HashCalculatorStringTestsWin32<T> : HashCalculatorStringTests<ArchWin32, T> where T : Hash, new()
    {
    }

    public abstract class HashQueryStringTestsWin64<T> : HashQueryStringTests<ArchWin64, T> where T : Hash, new()
    {
    }

    public abstract class HashQueryStringTestsWin32<T> : HashQueryStringTests<ArchWin32, T> where T : Hash, new()
    {
    }

    #endregion

    #region HashQueryStringTests32

    public class HashQueryStringTestsMd4Win32 : HashQueryStringTestsWin32<Md4>
    {
    }

    public class HashQueryStringTestsMd5Win32 : HashQueryStringTestsWin32<Md5>
    {
    }

    public class HashQueryStringTestsSha1Win32 : HashQueryStringTestsWin32<Sha1>
    {
    }

    public class HashQueryStringTestsSha224Win32 : HashQueryStringTestsWin32<Sha224>
    {
    }

    public class HashQueryStringTestsSha256Win32 : HashQueryStringTestsWin32<Sha256>
    {
    }

    public class HashQueryStringTestsSha384Win32 : HashQueryStringTestsWin32<Sha384>
    {
    }

    public class HashQueryStringTestsSha512Win32 : HashQueryStringTestsWin32<Sha512>
    {
    }

    public class HashQueryStringTestsWhirlpoolWin32 : HashQueryStringTestsWin32<Whirlpool>
    {
    }

    public class HashQueryStringTestsCrc32Win32 : HashQueryStringTestsWin32<Crc32>
    {
    }

    public class HashQueryStringTestsMd2Win32 : HashQueryStringTestsWin32<Md2>
    {
    }

    public class HashQueryStringTestsTigerWin32 : HashQueryStringTestsWin32<Tiger>
    {
    }

    public class HashQueryStringTestsTiger2Win32 : HashQueryStringTestsWin32<Tiger2>
    {
    }

    public class HashQueryStringTestsRmd128Win32 : HashQueryStringTestsWin32<Rmd128>
    {
    }

    public class HashQueryStringTestsRmd160Win32 : HashQueryStringTestsWin32<Rmd160>
    {
    }

    public class HashQueryStringTestsRmd256Win32 : HashQueryStringTestsWin32<Rmd256>
    {
    }

    public class HashQueryStringTestsRmd320Win32 : HashQueryStringTestsWin32<Rmd320>
    {
    }

    public class HashQueryStringTestsGostWin32 : HashQueryStringTestsWin32<Gost>
    {
    }

    public class HashQueryStringTestsSnefru128Win32 : HashQueryStringTestsWin32<Snefru128>
    {
    }

    public class HashQueryStringTestsSnefru256Win32 : HashQueryStringTestsWin32<Snefru256>
    {
    }

    public class HashQueryStringTestsTthWin32 : HashQueryStringTestsWin32<Tth>
    {
    }

    public class HashQueryStringTestsHaval_128_3Win32 : HashQueryStringTestsWin32<Haval_128_3>
    {
    }

    public class HashQueryStringTestsHaval_128_4Win32 : HashQueryStringTestsWin32<Haval_128_4>
    {
    }

    public class HashQueryStringTestsHaval_128_5Win32 : HashQueryStringTestsWin32<Haval_128_5>
    {
    }

    public class HashQueryStringTestsHaval_160_3Win32 : HashQueryStringTestsWin32<Haval_160_3>
    {
    }

    public class HashQueryStringTestsHaval_160_4Win32 : HashQueryStringTestsWin32<Haval_160_4>
    {
    }

    public class HashQueryStringTestsHaval_160_5Win32 : HashQueryStringTestsWin32<Haval_160_5>
    {
    }

    public class HashQueryStringTestsHaval_192_3Win32 : HashQueryStringTestsWin32<Haval_192_3>
    {
    }

    public class HashQueryStringTestsHaval_192_4Win32 : HashQueryStringTestsWin32<Haval_192_4>
    {
    }

    public class HashQueryStringTestsHaval_192_5Win32 : HashQueryStringTestsWin32<Haval_192_5>
    {
    }

    public class HashQueryStringTestsHaval_224_3Win32 : HashQueryStringTestsWin32<Haval_224_3>
    {
    }

    public class HashQueryStringTestsHaval_224_4Win32 : HashQueryStringTestsWin32<Haval_224_4>
    {
    }

    public class HashQueryStringTestsHaval_224_5Win32 : HashQueryStringTestsWin32<Haval_224_5>
    {
    }

    public class HashQueryStringTestsHaval_256_3Win32 : HashQueryStringTestsWin32<Haval_256_3>
    {
    }

    public class HashQueryStringTestsHaval_256_4Win32 : HashQueryStringTestsWin32<Haval_256_4>
    {
    }

    public class HashQueryStringTestsHaval_256_5Win32 : HashQueryStringTestsWin32<Haval_256_5>
    {
    }

    public class HashQueryStringTestsEdonr256Win32 : HashQueryStringTestsWin32<Edonr256>
    {
    }

    public class HashQueryStringTestsEdonr512Win32 : HashQueryStringTestsWin32<Edonr512>
    {
    }

    public class HashQueryStringTestsNtlmWin32 : HashQueryStringTestsWin32<Ntlm>
    {
    }

    public class HashQueryStringTestsSha_3_224Win32 : HashQueryStringTestsWin32<Sha_3_224>
    {
    }

    public class HashQueryStringTestsSha_3_256Win32 : HashQueryStringTestsWin32<Sha_3_256>
    {
    }

    public class HashQueryStringTestsSha_3_384Win32 : HashQueryStringTestsWin32<Sha_3_384>
    {
    }

    public class HashQueryStringTestsSha_3_512Win32 : HashQueryStringTestsWin32<Sha_3K_512>
    {
    }

    public class HashQueryStringTestsSha_3K_224Win32 : HashQueryStringTestsWin32<Sha_3K_224>
    {
    }

    public class HashQueryStringTestsSha_3K_256Win32 : HashQueryStringTestsWin32<Sha_3K_256>
    {
    }

    public class HashQueryStringTestsSha_3K_384Win32 : HashQueryStringTestsWin32<Sha_3K_384>
    {
    }

    public class HashQueryStringTestsSha_3K_512Win32 : HashQueryStringTestsWin32<Sha_3K_512>
    {
    }

    #endregion

    #region HashQueryStringTests64

    public class HashQueryStringTestsMd4Win64 : HashQueryStringTestsWin64<Md4>
    {
    }

    public class HashQueryStringTestsMd5Win64 : HashQueryStringTestsWin64<Md5>
    {
    }

    public class HashQueryStringTestsSha1Win64 : HashQueryStringTestsWin64<Sha1>
    {
    }

    public class HashQueryStringTestsSha224Win64 : HashQueryStringTestsWin64<Sha224>
    {
    }

    public class HashQueryStringTestsSha256Win64 : HashQueryStringTestsWin64<Sha256>
    {
    }

    public class HashQueryStringTestsSha384Win64 : HashQueryStringTestsWin64<Sha384>
    {
    }

    public class HashQueryStringTestsSha512Win64 : HashQueryStringTestsWin64<Sha512>
    {
    }

    public class HashQueryStringTestsWhirlpoolWin64 : HashQueryStringTestsWin64<Whirlpool>
    {
    }

    public class HashQueryStringTestsCrc32Win64 : HashQueryStringTestsWin64<Crc32>
    {
    }

    public class HashQueryStringTestsMd2Win64 : HashQueryStringTestsWin64<Md2>
    {
    }

    public class HashQueryStringTestsTigerWin64 : HashQueryStringTestsWin64<Tiger>
    {
    }

    public class HashQueryStringTestsTiger2Win64 : HashQueryStringTestsWin64<Tiger2>
    {
    }

    public class HashQueryStringTestsRmd128Win64 : HashQueryStringTestsWin64<Rmd128>
    {
    }

    public class HashQueryStringTestsRmd160Win64 : HashQueryStringTestsWin64<Rmd160>
    {
    }

    public class HashQueryStringTestsRmd256Win64 : HashQueryStringTestsWin64<Rmd256>
    {
    }

    public class HashQueryStringTestsRmd320Win64 : HashQueryStringTestsWin64<Rmd320>
    {
    }

    public class HashQueryStringTestsGostWin64 : HashQueryStringTestsWin64<Gost>
    {
    }

    public class HashQueryStringTestsSnefru128Win64 : HashQueryStringTestsWin64<Snefru128>
    {
    }

    public class HashQueryStringTestsSnefru256Win64 : HashQueryStringTestsWin64<Snefru256>
    {
    }

    public class HashQueryStringTestsTthWin64 : HashQueryStringTestsWin64<Tth>
    {
    }

    public class HashQueryStringTestsHaval_128_3Win64 : HashQueryStringTestsWin64<Haval_128_3>
    {
    }

    public class HashQueryStringTestsHaval_128_4Win64 : HashQueryStringTestsWin64<Haval_128_4>
    {
    }

    public class HashQueryStringTestsHaval_128_5Win64 : HashQueryStringTestsWin64<Haval_128_5>
    {
    }

    public class HashQueryStringTestsHaval_160_3Win64 : HashQueryStringTestsWin64<Haval_160_3>
    {
    }

    public class HashQueryStringTestsHaval_160_4Win64 : HashQueryStringTestsWin64<Haval_160_4>
    {
    }

    public class HashQueryStringTestsHaval_160_5Win64 : HashQueryStringTestsWin64<Haval_160_5>
    {
    }

    public class HashQueryStringTestsHaval_192_3Win64 : HashQueryStringTestsWin64<Haval_192_3>
    {
    }

    public class HashQueryStringTestsHaval_192_4Win64 : HashQueryStringTestsWin64<Haval_192_4>
    {
    }

    public class HashQueryStringTestsHaval_192_5Win64 : HashQueryStringTestsWin64<Haval_192_5>
    {
    }

    public class HashQueryStringTestsHaval_224_3Win64 : HashQueryStringTestsWin64<Haval_224_3>
    {
    }

    public class HashQueryStringTestsHaval_224_4Win64 : HashQueryStringTestsWin64<Haval_224_4>
    {
    }

    public class HashQueryStringTestsHaval_224_5Win64 : HashQueryStringTestsWin64<Haval_224_5>
    {
    }

    public class HashQueryStringTestsHaval_256_3Win64 : HashQueryStringTestsWin64<Haval_256_3>
    {
    }

    public class HashQueryStringTestsHaval_256_4Win64 : HashQueryStringTestsWin64<Haval_256_4>
    {
    }

    public class HashQueryStringTestsHaval_256_5Win64 : HashQueryStringTestsWin64<Haval_256_5>
    {
    }

    public class HashQueryStringTestsEdonr256Win64 : HashQueryStringTestsWin64<Edonr256>
    {
    }

    public class HashQueryStringTestsEdonr512Win64 : HashQueryStringTestsWin64<Edonr512>
    {
    }

    public class HashQueryStringTestsNtlmWin64 : HashQueryStringTestsWin64<Ntlm>
    {
    }

    public class HashQueryStringTestsSha_3_224Win64 : HashQueryStringTestsWin64<Sha_3_224>
    {
    }

    public class HashQueryStringTestsSha_3_256Win64 : HashQueryStringTestsWin64<Sha_3_256>
    {
    }

    public class HashQueryStringTestsSha_3_384Win64 : HashQueryStringTestsWin64<Sha_3_384>
    {
    }

    public class HashQueryStringTestsSha_3_512Win64 : HashQueryStringTestsWin64<Sha_3K_512>
    {
    }

    public class HashQueryStringTestsSha_3K_224Win64 : HashQueryStringTestsWin64<Sha_3K_224>
    {
    }

    public class HashQueryStringTestsSha_3K_256Win64 : HashQueryStringTestsWin64<Sha_3K_256>
    {
    }

    public class HashQueryStringTestsSha_3K_384Win64 : HashQueryStringTestsWin64<Sha_3K_384>
    {
    }

    public class HashQueryStringTestsSha_3K_512Win64 : HashQueryStringTestsWin64<Sha_3K_512>
    {
    }

    #endregion

    #region HashCalculatorString32

    public class HashCalculatorStringTestsMd4Win32 : HashCalculatorStringTestsWin32<Md4>
    {
    }

    public class HashCalculatorStringTestsMd5Win32 : HashCalculatorStringTestsWin32<Md5>
    {
    }

    public class HashCalculatorStringTestsSha1Win32 : HashCalculatorStringTestsWin32<Sha1>
    {
    }

    public class HashCalculatorStringTestsSha224Win32 : HashCalculatorStringTestsWin32<Sha224>
    {
    }

    public class HashCalculatorStringTestsSha256Win32 : HashCalculatorStringTestsWin32<Sha256>
    {
    }

    public class HashCalculatorStringTestsSha384Win32 : HashCalculatorStringTestsWin32<Sha384>
    {
    }

    public class HashCalculatorStringTestsSha512Win32 : HashCalculatorStringTestsWin32<Sha512>
    {
    }

    public class HashCalculatorStringTestsWhirlpoolWin32 : HashCalculatorStringTestsWin32<Whirlpool>
    {
    }

    public class HashCalculatorStringTestsCrc32Win32 : HashCalculatorStringTestsWin32<Crc32>
    {
    }

    public class HashCalculatorStringTestsMd2Win32 : HashCalculatorStringTestsWin32<Md2>
    {
    }

    public class HashCalculatorStringTestsTigerWin32 : HashCalculatorStringTestsWin32<Tiger>
    {
    }

    public class HashCalculatorStringTestsTiger2Win32 : HashCalculatorStringTestsWin32<Tiger2>
    {
    }

    public class HashCalculatorStringTestsRmd128Win32 : HashCalculatorStringTestsWin32<Rmd128>
    {
    }

    public class HashCalculatorStringTestsRmd160Win32 : HashCalculatorStringTestsWin32<Rmd160>
    {
    }

    public class HashCalculatorStringTestsRmd256Win32 : HashCalculatorStringTestsWin32<Rmd256>
    {
    }

    public class HashCalculatorStringTestsRmd320Win32 : HashCalculatorStringTestsWin32<Rmd320>
    {
    }

    public class HashCalculatorStringTestsGostWin32 : HashCalculatorStringTestsWin32<Gost>
    {
    }

    public class HashCalculatorStringTestsSnefru128Win32 : HashCalculatorStringTestsWin32<Snefru128>
    {
    }

    public class HashCalculatorStringTestsSnefru256Win32 : HashCalculatorStringTestsWin32<Snefru256>
    {
    }

    public class HashCalculatorStringTestsTthWin32 : HashCalculatorStringTestsWin32<Tth>
    {
    }

    public class HashCalculatorStringTestsHaval_128_3Win32 : HashCalculatorStringTestsWin32<Haval_128_3>
    {
    }

    public class HashCalculatorStringTestsHaval_128_4Win32 : HashCalculatorStringTestsWin32<Haval_128_4>
    {
    }

    public class HashCalculatorStringTestsHaval_128_5Win32 : HashCalculatorStringTestsWin32<Haval_128_5>
    {
    }

    public class HashCalculatorStringTestsHaval_160_3Win32 : HashCalculatorStringTestsWin32<Haval_160_3>
    {
    }

    public class HashCalculatorStringTestsHaval_160_4Win32 : HashCalculatorStringTestsWin32<Haval_160_4>
    {
    }

    public class HashCalculatorStringTestsHaval_160_5Win32 : HashCalculatorStringTestsWin32<Haval_160_5>
    {
    }

    public class HashCalculatorStringTestsHaval_192_3Win32 : HashCalculatorStringTestsWin32<Haval_192_3>
    {
    }

    public class HashCalculatorStringTestsHaval_192_4Win32 : HashCalculatorStringTestsWin32<Haval_192_4>
    {
    }

    public class HashCalculatorStringTestsHaval_192_5Win32 : HashCalculatorStringTestsWin32<Haval_192_5>
    {
    }

    public class HashCalculatorStringTestsHaval_224_3Win32 : HashCalculatorStringTestsWin32<Haval_224_3>
    {
    }

    public class HashCalculatorStringTestsHaval_224_4Win32 : HashCalculatorStringTestsWin32<Haval_224_4>
    {
    }

    public class HashCalculatorStringTestsHaval_224_5Win32 : HashCalculatorStringTestsWin32<Haval_224_5>
    {
    }

    public class HashCalculatorStringTestsHaval_256_3Win32 : HashCalculatorStringTestsWin32<Haval_256_3>
    {
    }

    public class HashCalculatorStringTestsHaval_256_4Win32 : HashCalculatorStringTestsWin32<Haval_256_4>
    {
    }

    public class HashCalculatorStringTestsHaval_256_5Win32 : HashCalculatorStringTestsWin32<Haval_256_5>
    {
    }

    public class HashCalculatorStringTestsEdonr256Win32 : HashCalculatorStringTestsWin32<Edonr256>
    {
    }

    public class HashCalculatorStringTestsEdonr512Win32 : HashCalculatorStringTestsWin32<Edonr512>
    {
    }

    public class HashCalculatorStringTestsNtlmWin32 : HashCalculatorStringTestsWin32<Ntlm>
    {
    }

    public class HashCalculatorStringTestsSha_3_224Win32 : HashCalculatorStringTestsWin32<Sha_3_224>
    {
    }

    public class HashCalculatorStringTestsSha_3_256Win32 : HashCalculatorStringTestsWin32<Sha_3_256>
    {
    }

    public class HashCalculatorStringTestsSha_3_384Win32 : HashCalculatorStringTestsWin32<Sha_3_384>
    {
    }

    public class HashCalculatorStringTestsSha_3_512Win32 : HashCalculatorStringTestsWin32<Sha_3K_512>
    {
    }

    public class HashCalculatorStringTestsSha_3K_224Win32 : HashCalculatorStringTestsWin32<Sha_3K_224>
    {
    }

    public class HashCalculatorStringTestsSha_3K_256Win32 : HashCalculatorStringTestsWin32<Sha_3K_256>
    {
    }

    public class HashCalculatorStringTestsSha_3K_384Win32 : HashCalculatorStringTestsWin32<Sha_3K_384>
    {
    }

    public class HashCalculatorStringTestsSha_3K_512Win32 : HashCalculatorStringTestsWin32<Sha_3K_512>
    {
    }

    #endregion

    #region HashCalculatorString64

    public class HashCalculatorStringTestsMd4Win64 : HashCalculatorStringTestsWin64<Md4>
    {
    }

    public class HashCalculatorStringTestsMd5Win64 : HashCalculatorStringTestsWin64<Md5>
    {
    }

    public class HashCalculatorStringTestsSha1Win64 : HashCalculatorStringTestsWin64<Sha1>
    {
    }

    public class HashCalculatorStringTestsSha224Win64 : HashCalculatorStringTestsWin64<Sha224>
    {
    }

    public class HashCalculatorStringTestsSha256Win64 : HashCalculatorStringTestsWin64<Sha256>
    {
    }

    public class HashCalculatorStringTestsSha384Win64 : HashCalculatorStringTestsWin64<Sha384>
    {
    }

    public class HashCalculatorStringTestsSha512Win64 : HashCalculatorStringTestsWin64<Sha512>
    {
    }

    public class HashCalculatorStringTestsWhirlpoolWin64 : HashCalculatorStringTestsWin64<Whirlpool>
    {
    }

    public class HashCalculatorStringTestsCrc32Win64 : HashCalculatorStringTestsWin64<Crc32>
    {
    }

    public class HashCalculatorStringTestsMd2Win64 : HashCalculatorStringTestsWin64<Md2>
    {
    }

    public class HashCalculatorStringTestsTigerWin64 : HashCalculatorStringTestsWin64<Tiger>
    {
    }

    public class HashCalculatorStringTestsTiger2Win64 : HashCalculatorStringTestsWin64<Tiger2>
    {
    }

    public class HashCalculatorStringTestsRmd128Win64 : HashCalculatorStringTestsWin64<Rmd128>
    {
    }

    public class HashCalculatorStringTestsRmd160Win64 : HashCalculatorStringTestsWin64<Rmd160>
    {
    }

    public class HashCalculatorStringTestsRmd256Win64 : HashCalculatorStringTestsWin64<Rmd256>
    {
    }

    public class HashCalculatorStringTestsRmd320Win64 : HashCalculatorStringTestsWin64<Rmd320>
    {
    }

    public class HashCalculatorStringTestsGostWin64 : HashCalculatorStringTestsWin64<Gost>
    {
    }

    public class HashCalculatorStringTestsSnefru128Win64 : HashCalculatorStringTestsWin64<Snefru128>
    {
    }

    public class HashCalculatorStringTestsSnefru256Win64 : HashCalculatorStringTestsWin64<Snefru256>
    {
    }

    public class HashCalculatorStringTestsTthWin64 : HashCalculatorStringTestsWin64<Tth>
    {
    }

    public class HashCalculatorStringTestsHaval_128_3Win64 : HashCalculatorStringTestsWin64<Haval_128_3>
    {
    }

    public class HashCalculatorStringTestsHaval_128_4Win64 : HashCalculatorStringTestsWin64<Haval_128_4>
    {
    }

    public class HashCalculatorStringTestsHaval_128_5Win64 : HashCalculatorStringTestsWin64<Haval_128_5>
    {
    }

    public class HashCalculatorStringTestsHaval_160_3Win64 : HashCalculatorStringTestsWin64<Haval_160_3>
    {
    }

    public class HashCalculatorStringTestsHaval_160_4Win64 : HashCalculatorStringTestsWin64<Haval_160_4>
    {
    }

    public class HashCalculatorStringTestsHaval_160_5Win64 : HashCalculatorStringTestsWin64<Haval_160_5>
    {
    }

    public class HashCalculatorStringTestsHaval_192_3Win64 : HashCalculatorStringTestsWin64<Haval_192_3>
    {
    }

    public class HashCalculatorStringTestsHaval_192_4Win64 : HashCalculatorStringTestsWin64<Haval_192_4>
    {
    }

    public class HashCalculatorStringTestsHaval_192_5Win64 : HashCalculatorStringTestsWin64<Haval_192_5>
    {
    }

    public class HashCalculatorStringTestsHaval_224_3Win64 : HashCalculatorStringTestsWin64<Haval_224_3>
    {
    }

    public class HashCalculatorStringTestsHaval_224_4Win64 : HashCalculatorStringTestsWin64<Haval_224_4>
    {
    }

    public class HashCalculatorStringTestsHaval_224_5Win64 : HashCalculatorStringTestsWin64<Haval_224_5>
    {
    }

    public class HashCalculatorStringTestsHaval_256_3Win64 : HashCalculatorStringTestsWin64<Haval_256_3>
    {
    }

    public class HashCalculatorStringTestsHaval_256_4Win64 : HashCalculatorStringTestsWin64<Haval_256_4>
    {
    }

    public class HashCalculatorStringTestsHaval_256_5Win64 : HashCalculatorStringTestsWin64<Haval_256_5>
    {
    }

    public class HashCalculatorStringTestsEdonr256Win64 : HashCalculatorStringTestsWin64<Edonr256>
    {
    }

    public class HashCalculatorStringTestsEdonr512Win64 : HashCalculatorStringTestsWin64<Edonr512>
    {
    }

    public class HashCalculatorStringTestsNtlmWin64 : HashCalculatorStringTestsWin64<Ntlm>
    {
    }

    public class HashCalculatorStringTestsSha_3_224Win64 : HashCalculatorStringTestsWin64<Sha_3_224>
    {
    }

    public class HashCalculatorStringTestsSha_3_256Win64 : HashCalculatorStringTestsWin64<Sha_3_256>
    {
    }

    public class HashCalculatorStringTestsSha_3_384Win64 : HashCalculatorStringTestsWin64<Sha_3_384>
    {
    }

    public class HashCalculatorStringTestsSha_3_512Win64 : HashCalculatorStringTestsWin64<Sha_3K_512>
    {
    }

    public class HashCalculatorStringTestsSha_3K_224Win64 : HashCalculatorStringTestsWin64<Sha_3K_224>
    {
    }

    public class HashCalculatorStringTestsSha_3K_256Win64 : HashCalculatorStringTestsWin64<Sha_3K_256>
    {
    }

    public class HashCalculatorStringTestsSha_3K_384Win64 : HashCalculatorStringTestsWin64<Sha_3K_384>
    {
    }

    public class HashCalculatorStringTestsSha_3K_512Win64 : HashCalculatorStringTestsWin64<Sha_3K_512>
    {
    }

    #endregion

    #region HashQueryFileTests32

    public class HashQueryFileTestsMd4Win32 : HashQueryFileTestsWin32<Md4>
    {
    }

    public class HashQueryFileTestsMd5Win32 : HashQueryFileTestsWin32<Md5>
    {
    }

    public class HashQueryFileTestsSha1Win32 : HashQueryFileTestsWin32<Sha1>
    {
    }

    public class HashQueryFileTestsSha224Win32 : HashQueryFileTestsWin32<Sha224>
    {
    }

    public class HashQueryFileTestsSha256Win32 : HashQueryFileTestsWin32<Sha256>
    {
    }

    public class HashQueryFileTestsSha384Win32 : HashQueryFileTestsWin32<Sha384>
    {
    }

    public class HashQueryFileTestsSha512Win32 : HashQueryFileTestsWin32<Sha512>
    {
    }

    public class HashQueryFileTestsWhirlpoolWin32 : HashQueryFileTestsWin32<Whirlpool>
    {
    }

    public class HashQueryFileTestsCrc32Win32 : HashQueryFileTestsWin32<Crc32>
    {
    }

    public class HashQueryFileTestsMd2Win32 : HashQueryFileTestsWin32<Md2>
    {
    }

    public class HashQueryFileTestsTigerWin32 : HashQueryFileTestsWin32<Tiger>
    {
    }

    public class HashQueryFileTestsTiger2Win32 : HashQueryFileTestsWin32<Tiger2>
    {
    }

    public class HashQueryFileTestsRmd128Win32 : HashQueryFileTestsWin32<Rmd128>
    {
    }

    public class HashQueryFileTestsRmd160Win32 : HashQueryFileTestsWin32<Rmd160>
    {
    }

    public class HashQueryFileTestsRmd256Win32 : HashQueryFileTestsWin32<Rmd256>
    {
    }

    public class HashQueryFileTestsRmd320Win32 : HashQueryFileTestsWin32<Rmd320>
    {
    }

    public class HashQueryFileTestsGostWin32 : HashQueryFileTestsWin32<Gost>
    {
    }

    public class HashQueryFileTestsSnefru128Win32 : HashQueryFileTestsWin32<Snefru128>
    {
    }

    public class HashQueryFileTestsSnefru256Win32 : HashQueryFileTestsWin32<Snefru256>
    {
    }

    public class HashQueryFileTestsTthWin32 : HashQueryFileTestsWin32<Tth>
    {
    }

    public class HashQueryFileTestsHaval_128_3Win32 : HashQueryFileTestsWin32<Haval_128_3>
    {
    }

    public class HashQueryFileTestsHaval_128_4Win32 : HashQueryFileTestsWin32<Haval_128_4>
    {
    }

    public class HashQueryFileTestsHaval_128_5Win32 : HashQueryFileTestsWin32<Haval_128_5>
    {
    }

    public class HashQueryFileTestsHaval_160_3Win32 : HashQueryFileTestsWin32<Haval_160_3>
    {
    }

    public class HashQueryFileTestsHaval_160_4Win32 : HashQueryFileTestsWin32<Haval_160_4>
    {
    }

    public class HashQueryFileTestsHaval_160_5Win32 : HashQueryFileTestsWin32<Haval_160_5>
    {
    }

    public class HashQueryFileTestsHaval_192_3Win32 : HashQueryFileTestsWin32<Haval_192_3>
    {
    }

    public class HashQueryFileTestsHaval_192_4Win32 : HashQueryFileTestsWin32<Haval_192_4>
    {
    }

    public class HashQueryFileTestsHaval_192_5Win32 : HashQueryFileTestsWin32<Haval_192_5>
    {
    }

    public class HashQueryFileTestsHaval_224_3Win32 : HashQueryFileTestsWin32<Haval_224_3>
    {
    }

    public class HashQueryFileTestsHaval_224_4Win32 : HashQueryFileTestsWin32<Haval_224_4>
    {
    }

    public class HashQueryFileTestsHaval_224_5Win32 : HashQueryFileTestsWin32<Haval_224_5>
    {
    }

    public class HashQueryFileTestsHaval_256_3Win32 : HashQueryFileTestsWin32<Haval_256_3>
    {
    }

    public class HashQueryFileTestsHaval_256_4Win32 : HashQueryFileTestsWin32<Haval_256_4>
    {
    }

    public class HashQueryFileTestsHaval_256_5Win32 : HashQueryFileTestsWin32<Haval_256_5>
    {
    }

    public class HashQueryFileTestsEdonr256Win32 : HashQueryFileTestsWin32<Edonr256>
    {
    }

    public class HashQueryFileTestsEdonr512Win32 : HashQueryFileTestsWin32<Edonr512>
    {
    }

    public class HashQueryFileTestsSha_3_224Win32 : HashQueryFileTestsWin32<Sha_3_224>
    {
    }

    public class HashQueryFileTestsSha_3_256Win32 : HashQueryFileTestsWin32<Sha_3_256>
    {
    }

    public class HashQueryFileTestsSha_3_384Win32 : HashQueryFileTestsWin32<Sha_3_384>
    {
    }

    public class HashQueryFileTestsSha_3_512Win32 : HashQueryFileTestsWin32<Sha_3K_512>
    {
    }

    public class HashQueryFileTestsSha_3K_224Win32 : HashQueryFileTestsWin32<Sha_3K_224>
    {
    }

    public class HashQueryFileTestsSha_3K_256Win32 : HashQueryFileTestsWin32<Sha_3K_256>
    {
    }

    public class HashQueryFileTestsSha_3K_384Win32 : HashQueryFileTestsWin32<Sha_3K_384>
    {
    }

    public class HashQueryFileTestsSha_3K_512Win32 : HashQueryFileTestsWin32<Sha_3K_512>
    {
    }

    #endregion

    #region HashQueryFileTests64

    public class HashQueryFileTestsMd4Win64 : HashQueryFileTestsWin64<Md4>
    {
    }

    public class HashQueryFileTestsMd5Win64 : HashQueryFileTestsWin64<Md5>
    {
    }

    public class HashQueryFileTestsSha1Win64 : HashQueryFileTestsWin64<Sha1>
    {
    }

    public class HashQueryFileTestsSha224Win64 : HashQueryFileTestsWin64<Sha224>
    {
    }

    public class HashQueryFileTestsSha256Win64 : HashQueryFileTestsWin64<Sha256>
    {
    }

    public class HashQueryFileTestsSha384Win64 : HashQueryFileTestsWin64<Sha384>
    {
    }

    public class HashQueryFileTestsSha512Win64 : HashQueryFileTestsWin64<Sha512>
    {
    }

    public class HashQueryFileTestsWhirlpoolWin64 : HashQueryFileTestsWin64<Whirlpool>
    {
    }

    public class HashQueryFileTestsCrc32Win64 : HashQueryFileTestsWin64<Crc32>
    {
    }

    public class HashQueryFileTestsMd2Win64 : HashQueryFileTestsWin64<Md2>
    {
    }

    public class HashQueryFileTestsTigerWin64 : HashQueryFileTestsWin64<Tiger>
    {
    }

    public class HashQueryFileTestsTiger2Win64 : HashQueryFileTestsWin64<Tiger2>
    {
    }

    public class HashQueryFileTestsRmd128Win64 : HashQueryFileTestsWin64<Rmd128>
    {
    }

    public class HashQueryFileTestsRmd160Win64 : HashQueryFileTestsWin64<Rmd160>
    {
    }

    public class HashQueryFileTestsRmd256Win64 : HashQueryFileTestsWin64<Rmd256>
    {
    }

    public class HashQueryFileTestsRmd320Win64 : HashQueryFileTestsWin64<Rmd320>
    {
    }

    public class HashQueryFileTestsGostWin64 : HashQueryFileTestsWin64<Gost>
    {
    }

    public class HashQueryFileTestsSnefru128Win64 : HashQueryFileTestsWin64<Snefru128>
    {
    }

    public class HashQueryFileTestsSnefru256Win64 : HashQueryFileTestsWin64<Snefru256>
    {
    }

    public class HashQueryFileTestsTthWin64 : HashQueryFileTestsWin64<Tth>
    {
    }

    public class HashQueryFileTestsHaval_128_3Win64 : HashQueryFileTestsWin64<Haval_128_3>
    {
    }

    public class HashQueryFileTestsHaval_128_4Win64 : HashQueryFileTestsWin64<Haval_128_4>
    {
    }

    public class HashQueryFileTestsHaval_128_5Win64 : HashQueryFileTestsWin64<Haval_128_5>
    {
    }

    public class HashQueryFileTestsHaval_160_3Win64 : HashQueryFileTestsWin64<Haval_160_3>
    {
    }

    public class HashQueryFileTestsHaval_160_4Win64 : HashQueryFileTestsWin64<Haval_160_4>
    {
    }

    public class HashQueryFileTestsHaval_160_5Win64 : HashQueryFileTestsWin64<Haval_160_5>
    {
    }

    public class HashQueryFileTestsHaval_192_3Win64 : HashQueryFileTestsWin64<Haval_192_3>
    {
    }

    public class HashQueryFileTestsHaval_192_4Win64 : HashQueryFileTestsWin64<Haval_192_4>
    {
    }

    public class HashQueryFileTestsHaval_192_5Win64 : HashQueryFileTestsWin64<Haval_192_5>
    {
    }

    public class HashQueryFileTestsHaval_224_3Win64 : HashQueryFileTestsWin64<Haval_224_3>
    {
    }

    public class HashQueryFileTestsHaval_224_4Win64 : HashQueryFileTestsWin64<Haval_224_4>
    {
    }

    public class HashQueryFileTestsHaval_224_5Win64 : HashQueryFileTestsWin64<Haval_224_5>
    {
    }

    public class HashQueryFileTestsHaval_256_3Win64 : HashQueryFileTestsWin64<Haval_256_3>
    {
    }

    public class HashQueryFileTestsHaval_256_4Win64 : HashQueryFileTestsWin64<Haval_256_4>
    {
    }

    public class HashQueryFileTestsHaval_256_5Win64 : HashQueryFileTestsWin64<Haval_256_5>
    {
    }

    public class HashQueryFileTestsEdonr256Win64 : HashQueryFileTestsWin64<Edonr256>
    {
    }

    public class HashQueryFileTestsEdonr512Win64 : HashQueryFileTestsWin64<Edonr512>
    {
    }

    public class HashQueryFileTestsSha_3_224Win64 : HashQueryFileTestsWin64<Sha_3_224>
    {
    }

    public class HashQueryFileTestsSha_3_256Win64 : HashQueryFileTestsWin64<Sha_3_256>
    {
    }

    public class HashQueryFileTestsSha_3_384Win64 : HashQueryFileTestsWin64<Sha_3_384>
    {
    }

    public class HashQueryFileTestsSha_3_512Win64 : HashQueryFileTestsWin64<Sha_3K_512>
    {
    }

    public class HashQueryFileTestsSha_3K_224Win64 : HashQueryFileTestsWin64<Sha_3K_224>
    {
    }

    public class HashQueryFileTestsSha_3K_256Win64 : HashQueryFileTestsWin64<Sha_3K_256>
    {
    }

    public class HashQueryFileTestsSha_3K_384Win64 : HashQueryFileTestsWin64<Sha_3K_384>
    {
    }

    public class HashQueryFileTestsSha_3K_512Win64 : HashQueryFileTestsWin64<Sha_3K_512>
    {
    }

    #endregion

    #region HashCalculatorFileTests32

    public class HashCalculatorFileTestsMd4Win32 : HashCalculatorFileTestsWin32<Md4>
    {
    }

    public class HashCalculatorFileTestsMd5Win32 : HashCalculatorFileTestsWin32<Md5>
    {
    }

    public class HashCalculatorFileTestsSha1Win32 : HashCalculatorFileTestsWin32<Sha1>
    {
    }

    public class HashCalculatorFileTestsSha224Win32 : HashCalculatorFileTestsWin32<Sha224>
    {
    }

    public class HashCalculatorFileTestsSha256Win32 : HashCalculatorFileTestsWin32<Sha256>
    {
    }

    public class HashCalculatorFileTestsSha384Win32 : HashCalculatorFileTestsWin32<Sha384>
    {
    }

    public class HashCalculatorFileTestsSha512Win32 : HashCalculatorFileTestsWin32<Sha512>
    {
    }

    public class HashCalculatorFileTestsWhirlpoolWin32 : HashCalculatorFileTestsWin32<Whirlpool>
    {
    }

    public class HashCalculatorFileTestsCrc32Win32 : HashCalculatorFileTestsWin32<Crc32>
    {
    }

    public class HashCalculatorFileTestsMd2Win32 : HashCalculatorFileTestsWin32<Md2>
    {
    }

    public class HashCalculatorFileTestsTigerWin32 : HashCalculatorFileTestsWin32<Tiger>
    {
    }

    public class HashCalculatorFileTestsTiger2Win32 : HashCalculatorFileTestsWin32<Tiger2>
    {
    }

    public class HashCalculatorFileTestsRmd128Win32 : HashCalculatorFileTestsWin32<Rmd128>
    {
    }

    public class HashCalculatorFileTestsRmd160Win32 : HashCalculatorFileTestsWin32<Rmd160>
    {
    }

    public class HashCalculatorFileTestsRmd256Win32 : HashCalculatorFileTestsWin32<Rmd256>
    {
    }

    public class HashCalculatorFileTestsRmd320Win32 : HashCalculatorFileTestsWin32<Rmd320>
    {
    }

    public class HashCalculatorFileTestsGostWin32 : HashCalculatorFileTestsWin32<Gost>
    {
    }

    public class HashCalculatorFileTestsSnefru128Win32 : HashCalculatorFileTestsWin32<Snefru128>
    {
    }

    public class HashCalculatorFileTestsSnefru256Win32 : HashCalculatorFileTestsWin32<Snefru256>
    {
    }

    public class HashCalculatorFileTestsTthWin32 : HashCalculatorFileTestsWin32<Tth>
    {
    }

    public class HashCalculatorFileTestsHaval_128_3Win32 : HashCalculatorFileTestsWin32<Haval_128_3>
    {
    }

    public class HashCalculatorFileTestsHaval_128_4Win32 : HashCalculatorFileTestsWin32<Haval_128_4>
    {
    }

    public class HashCalculatorFileTestsHaval_128_5Win32 : HashCalculatorFileTestsWin32<Haval_128_5>
    {
    }

    public class HashCalculatorFileTestsHaval_160_3Win32 : HashCalculatorFileTestsWin32<Haval_160_3>
    {
    }

    public class HashCalculatorFileTestsHaval_160_4Win32 : HashCalculatorFileTestsWin32<Haval_160_4>
    {
    }

    public class HashCalculatorFileTestsHaval_160_5Win32 : HashCalculatorFileTestsWin32<Haval_160_5>
    {
    }

    public class HashCalculatorFileTestsHaval_192_3Win32 : HashCalculatorFileTestsWin32<Haval_192_3>
    {
    }

    public class HashCalculatorFileTestsHaval_192_4Win32 : HashCalculatorFileTestsWin32<Haval_192_4>
    {
    }

    public class HashCalculatorFileTestsHaval_192_5Win32 : HashCalculatorFileTestsWin32<Haval_192_5>
    {
    }

    public class HashCalculatorFileTestsHaval_224_3Win32 : HashCalculatorFileTestsWin32<Haval_224_3>
    {
    }

    public class HashCalculatorFileTestsHaval_224_4Win32 : HashCalculatorFileTestsWin32<Haval_224_4>
    {
    }

    public class HashCalculatorFileTestsHaval_224_5Win32 : HashCalculatorFileTestsWin32<Haval_224_5>
    {
    }

    public class HashCalculatorFileTestsHaval_256_3Win32 : HashCalculatorFileTestsWin32<Haval_256_3>
    {
    }

    public class HashCalculatorFileTestsHaval_256_4Win32 : HashCalculatorFileTestsWin32<Haval_256_4>
    {
    }

    public class HashCalculatorFileTestsHaval_256_5Win32 : HashCalculatorFileTestsWin32<Haval_256_5>
    {
    }

    public class HashCalculatorFileTestsEdonr256Win32 : HashCalculatorFileTestsWin32<Edonr256>
    {
    }

    public class HashCalculatorFileTestsEdonr512Win32 : HashCalculatorFileTestsWin32<Edonr512>
    {
    }

    public class HashCalculatorFileTestsSha_3_224Win32 : HashCalculatorFileTestsWin32<Sha_3_224>
    {
    }

    public class HashCalculatorFileTestsSha_3_256Win32 : HashCalculatorFileTestsWin32<Sha_3_256>
    {
    }

    public class HashCalculatorFileTestsSha_3_384Win32 : HashCalculatorFileTestsWin32<Sha_3_384>
    {
    }

    public class HashCalculatorFileTestsSha_3_512Win32 : HashCalculatorFileTestsWin32<Sha_3K_512>
    {
    }

    public class HashCalculatorFileTestsSha_3K_224Win32 : HashCalculatorFileTestsWin32<Sha_3K_224>
    {
    }

    public class HashCalculatorFileTestsSha_3K_256Win32 : HashCalculatorFileTestsWin32<Sha_3K_256>
    {
    }

    public class HashCalculatorFileTestsSha_3K_384Win32 : HashCalculatorFileTestsWin32<Sha_3K_384>
    {
    }

    public class HashCalculatorFileTestsSha_3K_512Win32 : HashCalculatorFileTestsWin32<Sha_3K_512>
    {
    }

    #endregion

    #region HashCalculatorFileTests64

    public class HashCalculatorFileTestsMd4Win64 : HashCalculatorFileTestsWin64<Md4>
    {
    }

    public class HashCalculatorFileTestsMd5Win64 : HashCalculatorFileTestsWin64<Md5>
    {
    }

    public class HashCalculatorFileTestsSha1Win64 : HashCalculatorFileTestsWin64<Sha1>
    {
    }

    public class HashCalculatorFileTestsSha224Win64 : HashCalculatorFileTestsWin64<Sha224>
    {
    }

    public class HashCalculatorFileTestsSha256Win64 : HashCalculatorFileTestsWin64<Sha256>
    {
    }

    public class HashCalculatorFileTestsSha384Win64 : HashCalculatorFileTestsWin64<Sha384>
    {
    }

    public class HashCalculatorFileTestsSha512Win64 : HashCalculatorFileTestsWin64<Sha512>
    {
    }

    public class HashCalculatorFileTestsWhirlpoolWin64 : HashCalculatorFileTestsWin64<Whirlpool>
    {
    }

    public class HashCalculatorFileTestsCrc32Win64 : HashCalculatorFileTestsWin64<Crc32>
    {
    }

    public class HashCalculatorFileTestsMd2Win64 : HashCalculatorFileTestsWin64<Md2>
    {
    }

    public class HashCalculatorFileTestsTigerWin64 : HashCalculatorFileTestsWin64<Tiger>
    {
    }

    public class HashCalculatorFileTestsTiger2Win64 : HashCalculatorFileTestsWin64<Tiger2>
    {
    }

    public class HashCalculatorFileTestsRmd128Win64 : HashCalculatorFileTestsWin64<Rmd128>
    {
    }

    public class HashCalculatorFileTestsRmd160Win64 : HashCalculatorFileTestsWin64<Rmd160>
    {
    }

    public class HashCalculatorFileTestsRmd256Win64 : HashCalculatorFileTestsWin64<Rmd256>
    {
    }

    public class HashCalculatorFileTestsRmd320Win64 : HashCalculatorFileTestsWin64<Rmd320>
    {
    }

    public class HashCalculatorFileTestsGostWin64 : HashCalculatorFileTestsWin64<Gost>
    {
    }

    public class HashCalculatorFileTestsSnefru128Win64 : HashCalculatorFileTestsWin64<Snefru128>
    {
    }

    public class HashCalculatorFileTestsSnefru256Win64 : HashCalculatorFileTestsWin64<Snefru256>
    {
    }

    public class HashCalculatorFileTestsTthWin64 : HashCalculatorFileTestsWin64<Tth>
    {
    }

    public class HashCalculatorFileTestsHaval_128_3Win64 : HashCalculatorFileTestsWin64<Haval_128_3>
    {
    }

    public class HashCalculatorFileTestsHaval_128_4Win64 : HashCalculatorFileTestsWin64<Haval_128_4>
    {
    }

    public class HashCalculatorFileTestsHaval_128_5Win64 : HashCalculatorFileTestsWin64<Haval_128_5>
    {
    }

    public class HashCalculatorFileTestsHaval_160_3Win64 : HashCalculatorFileTestsWin64<Haval_160_3>
    {
    }

    public class HashCalculatorFileTestsHaval_160_4Win64 : HashCalculatorFileTestsWin64<Haval_160_4>
    {
    }

    public class HashCalculatorFileTestsHaval_160_5Win64 : HashCalculatorFileTestsWin64<Haval_160_5>
    {
    }

    public class HashCalculatorFileTestsHaval_192_3Win64 : HashCalculatorFileTestsWin64<Haval_192_3>
    {
    }

    public class HashCalculatorFileTestsHaval_192_4Win64 : HashCalculatorFileTestsWin64<Haval_192_4>
    {
    }

    public class HashCalculatorFileTestsHaval_192_5Win64 : HashCalculatorFileTestsWin64<Haval_192_5>
    {
    }

    public class HashCalculatorFileTestsHaval_224_3Win64 : HashCalculatorFileTestsWin64<Haval_224_3>
    {
    }

    public class HashCalculatorFileTestsHaval_224_4Win64 : HashCalculatorFileTestsWin64<Haval_224_4>
    {
    }

    public class HashCalculatorFileTestsHaval_224_5Win64 : HashCalculatorFileTestsWin64<Haval_224_5>
    {
    }

    public class HashCalculatorFileTestsHaval_256_3Win64 : HashCalculatorFileTestsWin64<Haval_256_3>
    {
    }

    public class HashCalculatorFileTestsHaval_256_4Win64 : HashCalculatorFileTestsWin64<Haval_256_4>
    {
    }

    public class HashCalculatorFileTestsHaval_256_5Win64 : HashCalculatorFileTestsWin64<Haval_256_5>
    {
    }

    public class HashCalculatorFileTestsEdonr256Win64 : HashCalculatorFileTestsWin64<Edonr256>
    {
    }

    public class HashCalculatorFileTestsEdonr512Win64 : HashCalculatorFileTestsWin64<Edonr512>
    {
    }

    public class HashCalculatorFileTestsSha_3_224Win64 : HashCalculatorFileTestsWin64<Sha_3_224>
    {
    }

    public class HashCalculatorFileTestsSha_3_256Win64 : HashCalculatorFileTestsWin64<Sha_3_256>
    {
    }

    public class HashCalculatorFileTestsSha_3_384Win64 : HashCalculatorFileTestsWin64<Sha_3_384>
    {
    }

    public class HashCalculatorFileTestsSha_3_512Win64 : HashCalculatorFileTestsWin64<Sha_3K_512>
    {
    }

    public class HashCalculatorFileTestsSha_3K_224Win64 : HashCalculatorFileTestsWin64<Sha_3K_224>
    {
    }

    public class HashCalculatorFileTestsSha_3K_256Win64 : HashCalculatorFileTestsWin64<Sha_3K_256>
    {
    }

    public class HashCalculatorFileTestsSha_3K_384Win64 : HashCalculatorFileTestsWin64<Sha_3K_384>
    {
    }

    public class HashCalculatorFileTestsSha_3K_512Win64 : HashCalculatorFileTestsWin64<Sha_3K_512>
    {
    }

    #endregion
}