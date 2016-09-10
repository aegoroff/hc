/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2016 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using FluentAssertions;
using Xunit;

namespace _tst.net
{
    [Trait("Mode", "cmd")]
    public abstract class CmdStringTests<T> : StringTests<T>
        where T : Architecture, new()
    {
        private const string StringOpt = "-s";
        private const string LowCaseOpt = "-l";
        private const string NoProbeOpt = "--noprobe";
        private const string CrackOpt = "-c";
        private const string HashOpt = "-m";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string DictOpt = "-a";
        private const string PerfOpt = "-p";
        private const string EmptyStr = "\"\"";


        protected CmdStringTests() : base(new T())
        {
        }

        protected override IList<string> RunEmptyStringCrack(Hash h)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.EmptyStringHash);
        }

        protected override IList<string> RunStringCrack(Hash h)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3");
        }

        protected override IList<string> RunStringCrackTooShort(Hash h)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt,
                                                (h.InitialString.Length - 1).ToString(), DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringCrackTooMinLength(Hash h)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MinOpt,
                                                (h.InitialString.Length + 1).ToString(), MaxOpt,
                                                (h.InitialString.Length + 2).ToString(), DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringCrackLowCaseHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString.ToLowerInvariant(), MaxOpt, "3", DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringOpt, h.InitialString);
        }

        protected override IList<string> RunStringHashLowCase(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringOpt, h.InitialString, LowCaseOpt);
        }

        protected override IList<string> RunEmptyStringHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringOpt, EmptyStr);
        }

        protected override IList<string> RunCrackStringUsingNonDefaultDictionary(Hash h, string dict)
        {
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.StartPartStringHash, DictOpt, dict, MaxOpt, "2", MinOpt, "2");
        }

        private static byte[] StringToByteArray(string hex)
        {
            var numberChars = hex.Length;
            var bytes = new byte[numberChars / 2];
            for (var i = 0; i < numberChars; i += 2)
                bytes[i / 2] = Convert.ToByte(hex.Substring(i, 2), 16);
            return bytes;
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_Base64_Success(Hash h)
        {
            // Arrange
            var bytes = StringToByteArray(h.HashString);
            var base64 = Convert.ToBase64String(bytes);
            
            // Act
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, "-b", base64, MaxOpt, "3");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString), $"Because {base64} must be restored to {h.InitialString}");
            results.Should().HaveCount(2);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_SingleThread_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", "1");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString));
            results.Should().HaveCount(2);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(HashesAndBadThreads))]
        public void CrackString_BadThreadCountNumber_Success(Hash h, string threads)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", threads);

            // Assert
            results[2].Should().Be(string.Format(RestoredStringTemplate, h.InitialString));
            results.Should().HaveCount(3);
        }

        public static IEnumerable<object[]> HashesAndBadThreads => CreateProperty(new object[] {"-1", "10000"});

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_SingleCharStringWithMaxOpt_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, "2"));
            results.Should().HaveCount(2);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackStringSingleCharStringWithMaxOptOnSingleThread(Hash h)
        {
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", "-T", "1");
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackStringSingleCharStringWithMaxOptAndNonDefaultDict(Hash h)
        {
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", DictOpt, "[0-9]");
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void TestPerformance(Hash h)
        {
            var results = this.Runner.Run(h.Algorithm, PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");
            Assert.Equal(string.Format(RestoredStringTemplate, "12345"), results[2]);
            Assert.Equal(3, results.Count);
        }

        [Trait("Type", "crack")]
        [Fact]
        public void CrackNonAsciiString()
        {
            Hash h = new Md5();
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, "327108899019B3BCFFF1683FBFDAF226", DictOpt, "еграб", MinOpt, "6", MaxOpt, "6");
            Asserts.StringMatching(results[1], "Initial string is: *");
            Assert.Equal(2, results.Count);
        }
    }
}