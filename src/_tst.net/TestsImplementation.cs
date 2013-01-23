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
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    public class HashCalculator64<THash> : HashCalculator<THash> where THash : Hash, new()
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
    public class HashQuery64<THash> : HashQuery<THash> where THash : Hash, new()
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
    public class HashCalculator32<THash> : HashCalculator<THash> where THash : Hash, new()
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
    public class HashQuery32<THash> : HashQuery<THash> where THash : Hash, new()
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }
    }
}