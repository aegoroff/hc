﻿/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using System.Collections.Generic;
using Xunit;
using Xunit.Extensions;

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
            return this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.StartPartStringHash, DictOpt, dict, MaxOpt, "2");
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("Hashes")]
        public void CrackStringSingleThread(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", "1");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[2]);
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("HashesAndBadThreads")]
        public void CrackStringBadThreads(Hash h, string threads)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", "-T", threads);
            Assert.Equal(4, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[3]);
        }

        public static IEnumerable<object[]> HashesAndBadThreads
        {
            get { return CreateProperty(new object[] {"-1", "10000"}); }
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("Hashes")]
        public void CrackStringSingleCharStringWithMaxOpt(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("Hashes")]
        public void CrackStringSingleCharStringWithMaxOptOnSingleThread(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", "-T", "1");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("Hashes")]
        public void CrackStringSingleCharStringWithMaxOptAndNonDefaultDict(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.MiddlePartStringHash, MaxOpt, "2", DictOpt, "[0-9]");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }

        [Trait("Type", "crack")]
        [Theory, PropertyData("Hashes")]
        public void TestPerformance(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "12345"), results[2]);
        }
    }
}