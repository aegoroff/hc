/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2013 Alexander Egorov
*/

using System;
using NUnit.Framework;

namespace _tst.net
{
    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Md2))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    public class HashCalculatorFileTests64<THash> : HashCalculatorFileTests<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\x64\Release\{0}"; }
        }
    }
    
    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Md2))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    public class HashQuery64<THash> : HashQuery<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\x64\Release\{0}"; }
        }
    }
	
	[TestFixture(typeof(Md2))]
	[TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    public class HashCalculatorFileTests32<THash> : HashCalculatorFileTests<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }
    }
    
    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Md2))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    public class HashQuery32<THash> : HashQuery<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }
    }

    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Md2))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    [TestFixture(typeof(Ntlm))]
    public class HashCalculatorStringTests64<THash> : HashCalculatorStringTests<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\x64\Release\{0}"; }
        }
    }

    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    [TestFixture(typeof(Md2))]
    [TestFixture(typeof(Tiger))]
    [TestFixture(typeof(Tiger2))]
    [TestFixture(typeof(Rmd128))]
    [TestFixture(typeof(Rmd160))]
    [TestFixture(typeof(Rmd256))]
    [TestFixture(typeof(Rmd320))]
    [TestFixture(typeof(Sha224))]
    [TestFixture(typeof(Gost))]
    [TestFixture(typeof(Snefru128))]
    [TestFixture(typeof(Snefru256))]
    [TestFixture(typeof(Tth))]
    [TestFixture(typeof(Haval_128_3))]
    [TestFixture(typeof(Haval_128_4))]
    [TestFixture(typeof(Haval_128_5))]
    [TestFixture(typeof(Haval_160_3))]
    [TestFixture(typeof(Haval_160_4))]
    [TestFixture(typeof(Haval_160_5))]
    [TestFixture(typeof(Haval_192_3))]
    [TestFixture(typeof(Haval_192_4))]
    [TestFixture(typeof(Haval_192_5))]
    [TestFixture(typeof(Haval_224_3))]
    [TestFixture(typeof(Haval_224_4))]
    [TestFixture(typeof(Haval_224_5))]
    [TestFixture(typeof(Haval_256_3))]
    [TestFixture(typeof(Haval_256_4))]
    [TestFixture(typeof(Haval_256_5))]
    [TestFixture(typeof(Edonr256))]
    [TestFixture(typeof(Edonr512))]
    [TestFixture(typeof(Ntlm))]
    public class HashCalculatorStringTests32<THash> : HashCalculatorStringTests<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }
    }
}