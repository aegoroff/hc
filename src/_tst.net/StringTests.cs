/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2015 Alexander Egorov
 */

using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace _tst.net
{
    [Trait("Group", "string")]
    [Trait("Category", "string")]
    public abstract class StringTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        protected const string RestoredStringTemplate = "Initial string is: {0}";
        protected const string NothingFound = "Nothing found";

        protected StringTests(T data) : base(data)
        {
        }

        protected override string Executable => "hc.exe";

        protected static IEnumerable<object[]> CreateProperty(object[] data)
        {
            return from h in Hashes from item in data select new[] { h[0], item };
        }

        public static IEnumerable<object[]> HashesAndNonDefaultDict => CreateProperty(new object[] { "123", "0-9", "0-9a-z", "0-9A-Z" });

        public static IEnumerable<object[]> HashesAndNonDefaultDictFailure => CreateProperty(new object[] { "a-zA-Z", "a-z", "A-Z", "abc" });

        protected abstract IList<string> RunEmptyStringCrack(Hash h);
        
        protected abstract IList<string> RunStringCrack(Hash h);
        
        protected abstract IList<string> RunStringCrackTooShort(Hash h);
        
        protected abstract IList<string> RunStringCrackTooMinLength(Hash h);
        
        protected abstract IList<string> RunStringHash(Hash h);
        
        protected abstract IList<string> RunStringHashLowCase(Hash h);
        
        protected abstract IList<string> RunEmptyStringHash(Hash h);

        protected abstract IList<string> RunStringCrackLowCaseHash(Hash h);
        
        protected abstract IList<string> RunCrackStringUsingNonDefaultDictionary(Hash h, string dict);

        [Theory, MemberData(nameof(Hashes))]
        public void CalcString(Hash h)
        {
            var results = this.RunStringHash(h);
            Assert.Equal(h.HashString, results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData(nameof(Hashes))]
        public void CalcStringLowCaseOutput(Hash h)
        {
            var results = this.RunStringHashLowCase(h);
            Assert.Equal(h.HashString.ToLowerInvariant(), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData(nameof(Hashes))]
        public void CalcEmptyString(Hash h)
        {
            var results = this.RunEmptyStringHash(h);
            Assert.Equal(h.EmptyStringHash, results[0]);
            Assert.Equal(1, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString(Hash h)
        {
            var results = this.RunStringCrack(h);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackEmptyString(Hash h)
        {
            var results = this.RunEmptyStringCrack(h);
            Assert.Equal("Attempts: 0 Time 00:00:0.000 Speed: 0 attempts/second", results[0]);
            Assert.Equal(string.Format(RestoredStringTemplate, "Empty string"), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackStringUsingLowCaseHash(Hash h)
        {
            var results = this.RunStringCrackLowCaseHash(h);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(HashesAndNonDefaultDict))]
        public void CrackStringSuccessUsingNonDefaultDictionary(Hash h, string dict)
        {
            var results = this.RunCrackStringUsingNonDefaultDictionary(h, dict);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString.Substring(0,2)), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(HashesAndNonDefaultDictFailure))]
        public void CrackStringFailureUsingNonDefaultDictionary(Hash h, string dict)
        {
            var results = this.RunCrackStringUsingNonDefaultDictionary(h, dict);
            Assert.Equal(NothingFound, results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackStringTooShortLength(Hash h)
        {
            var results = this.RunStringCrackTooShort(h);
            Assert.Equal(NothingFound, results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackStringTooLongMinLength(Hash h)
        {
            var results = this.RunStringCrackTooMinLength(h);
            Assert.Equal(NothingFound, results[1]);
            Assert.Equal(2, results.Count);
        }
        
        public static IEnumerable<object[]> Hashes => new[]
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
            new object[] {new Ripemd128()},
            new object[] {new Ripemd160()},
            new object[] {new Ripemd256()},
            new object[] {new Ripemd320()},
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