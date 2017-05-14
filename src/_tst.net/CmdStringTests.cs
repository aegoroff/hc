/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2017 Alexander Egorov
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
        private const string Base64Opt = "-b";
        private const string LowCaseOpt = "-l";
        private const string NoProbeOpt = "--noprobe";
        private const string HashOpt = "-m";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string DictOpt = "-a";
        private const string PerfOpt = "-p";
        private const string EmptyStr = "\"\"";
        private const string StringCmd = "string";
        private const string HashCmd = "hash";


        protected CmdStringTests() : base(new T())
        {
        }

        protected override IList<string> RunEmptyStringCrack(Hash h)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.EmptyStringHash);
        }

        protected override IList<string> RunStringCrack(Hash h)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3");
        }

        protected override IList<string> RunStringCrackTooShort(Hash h)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString, MaxOpt,
                                                (h.InitialString.Length - 1).ToString(), DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringCrackTooMinLength(Hash h)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString, MinOpt,
                                                (h.InitialString.Length + 1).ToString(), MaxOpt,
                                                (h.InitialString.Length + 2).ToString(), DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringCrackLowCaseHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString.ToLowerInvariant(), MaxOpt, "3", DictOpt, h.InitialString);
        }

        protected override IList<string> RunStringHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringCmd, StringOpt, h.InitialString);
        }

        protected override IList<string> RunStringHashLowCase(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringCmd, StringOpt, h.InitialString, LowCaseOpt);
        }

        protected override IList<string> RunEmptyStringHash(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringCmd, StringOpt, EmptyStr);
        }

        protected override IList<string> RunStringHashAsBase64(Hash h)
        {
            return this.Runner.Run(h.Algorithm, StringCmd, Base64Opt, StringOpt, h.InitialString);
        }

        protected override IList<string> RunCrackStringUsingNonDefaultDictionary(Hash h, string dict)
        {
            return this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.StartPartStringHash, DictOpt, dict, MaxOpt, "2", MinOpt, "2");
        }

#if DEBUG
        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_Base64_Success(Hash h)
        {
            // Arrange
            var bytes = StringToByteArray(h.HashString);
            var base64 = Convert.ToBase64String(bytes);
            
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, "-b", HashOpt, base64, MaxOpt, "3");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString), $"Because {base64} must be restored to {h.InitialString}");
            results.Should().HaveCount(2);
        }
#endif

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_SingleThread_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", "1");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString));
            results.Should().HaveCount(2);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(HashesAndBadThreads))]
        public void CrackString_BadThreadCountNumber_Success(Hash h, string threads)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", threads);

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
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2");

            // Assert
            results[1].Should().Be(string.Format(RestoredStringTemplate, "2"));
            results.Should().HaveCount(2);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_SingleCharStringWithMaxOptOnSingleThread_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", "-T", "1");

            // Assert
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_SingleCharStringWithMaxOptAndNonDefaultDict_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", DictOpt, "[0-9]");

            // Assert
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData(nameof(Hashes))]
        public void CrackString_TestPerformance_Success(Hash h)
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");

            // Assert
            Assert.Equal(string.Format(RestoredStringTemplate, "12345"), results[2]);
            Assert.Equal(3, results.Count);
        }

        [Trait("Type", "crack")]
        [Fact]
        public void CrackString_NonAscii_Success()
        {
            // Arrange
            Hash h = new Md5();

            // Act
            var results = this.Runner.Run(h.Algorithm, HashCmd, NoProbeOpt, HashOpt, "327108899019B3BCFFF1683FBFDAF226", DictOpt, "еграб", MinOpt, "6", MaxOpt, "6");

            // Assert
            Asserts.StringMatching(results[1], "Initial string is: *");
            Assert.Equal(2, results.Count);
        }
    }
}