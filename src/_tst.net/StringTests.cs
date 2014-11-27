/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Extensions;

namespace _tst.net
{
    [Trait("Category", "string")]
    public abstract class StringTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        protected const string RestoredStringTemplate = "Initial string is: {0}";

        protected override string Executable
        {
            get { return "hc.exe"; }
        }

        protected static IEnumerable<object[]> CreateProperty(object[] data)
        {
            return from h in Hashes from item in data select new[] { h[0], item };
        }

        public static IEnumerable<object[]> HashesAndNonDefaultDict
        {
            get { return CreateProperty(new object[] { "123", "0-9", "0-9a-z", "0-9A-Z", "0-9a-zA-Z" }); }
        }

        public static IEnumerable<object[]> HashesAndNonDefaultDictFailure
        {
            get { return CreateProperty(new object[] { "a-zA-Z", "a-z", "A-Z", "abc" }); }
        }

        protected abstract IList<string> RunEmptyStringCrack(Hash h);
        
        protected abstract IList<string> RunStringCrack(Hash h);
        
        protected abstract IList<string> RunStringHash(Hash h);
        
        protected abstract IList<string> RunStringHashLowCase(Hash h);
        
        protected abstract IList<string> RunEmptyStringHash(Hash h);

        [Theory, PropertyData("Hashes")]
        public void CalcString(Hash h)
        {
            IList<string> results = this.RunStringHash(h);
            Assert.Equal(1, results.Count);
            Assert.Equal(h.HashString, results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcStringLowCaseOutput(Hash h)
        {
            IList<string> results = this.RunStringHashLowCase(h);
            Assert.Equal(1, results.Count);
            Assert.Equal(h.HashString.ToLowerInvariant(), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcEmptyString(Hash h)
        {
            IList<string> results = this.RunEmptyStringHash(h);
            Assert.Equal(1, results.Count);
            Assert.Equal(h.EmptyStringHash, results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CrackString(Hash h)
        {
            IList<string> results = RunStringCrack(h);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[2]);
        }

        [Theory, PropertyData("Hashes")]
        public void CrackEmptyString(Hash h)
        {
            IList<string> results = RunEmptyStringCrack(h);
            Assert.Equal(3, results.Count);
            Assert.Equal("Attempts: 0 Time 00:00:0.000 Speed: 0 attempts/second", results[1]);
            Assert.Equal(string.Format(RestoredStringTemplate, "Empty string"), results[2]);
        }
        
        public static IEnumerable<object[]> Hashes
        {
            get
            {
                return new[]
                {
                    new object[] {new Md4()},
                    new object[] {new Md5()},
                    new object[] {new Md2()},
                    new object[] {new Sha1()},
                    new object[] {new Sha224()},
                    new object[] {new Sha256()},
                    new object[] {new Sha384()},
                    new object[] {new Sha512()},
                    new object[] {new Whirlpool()},
                    new object[] {new Crc32()},
                    new object[] {new Tiger()},
                    new object[] {new Tiger2()},
                    new object[] {new Rmd128()},
                    new object[] {new Rmd160()},
                    new object[] {new Rmd256()},
                    new object[] {new Rmd320()},
                    new object[] {new Gost()},
                    new object[] {new Snefru128()},
                    new object[] {new Snefru256()},
                    new object[] {new Tth()},
                    new object[] {new Haval_128_3()},
                    new object[] {new Haval_128_4()},
                    new object[] {new Haval_128_5()},
                    new object[] {new Haval_160_3()},
                    new object[] {new Haval_160_4()},
                    new object[] {new Haval_160_5()},
                    new object[] {new Haval_192_3()},
                    new object[] {new Haval_192_4()},
                    new object[] {new Haval_192_5()},
                    new object[] {new Haval_224_3()},
                    new object[] {new Haval_224_4()},
                    new object[] {new Haval_224_5()},
                    new object[] {new Haval_256_3()},
                    new object[] {new Haval_256_4()},
                    new object[] {new Haval_256_5()},
                    new object[] {new Edonr256()},
                    new object[] {new Edonr512()},
                    new object[] {new Sha_3_224()},
                    new object[] {new Sha_3_256()},
                    new object[] {new Sha_3_384()},
                    new object[] {new Sha_3_512()},
                    new object[] {new Sha_3K_224()},
                    new object[] {new Sha_3K_256()},
                    new object[] {new Sha_3K_384()},
                    new object[] {new Sha_3K_512()},
                    new object[] {new Ntlm()}
                };
            }
        }
    }
}