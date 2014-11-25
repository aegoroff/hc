/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using System.Collections.Generic;
using Xunit;
using Xunit.Extensions;

namespace _tst.net
{
    public abstract class HashCalculatorStringTests<T, THash> : StringTests<T, THash>
        where T : Architecture, new()
        where THash : Hash, new()
    {
        private const string StringOpt = "-s";
        private const string EmptyStr = "\"\"";
        private const string LowCaseOpt = "-l";
        private const string NoProbeOpt = "--noprobe";
        private const string CrackOpt = "-c";
        private const string HashOpt = "-m";
        private const string RestoredStringTemplate = "Initial string is: {0}";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string NothingFound = "Nothing found";
        private const string DictOpt = "-a";
        private const string PerfOpt = "-p";

        protected override string Executable
        {
            get { return base.Executable + " " + this.Hash.Algorithm; }
        }

        [Fact]
        public void CalcString()
        {
            IList<string> results = this.Runner.Run(StringOpt, this.InitialString);
            Assert.Equal(1, results.Count);
            Assert.Equal(this.HashString, results[0]);
        }

        [Fact]
        public void CalcStringLowCaseOutput()
        {
            IList<string> results = this.Runner.Run(StringOpt, this.InitialString, LowCaseOpt);
            Assert.Equal(1, results.Count);
            Assert.Equal(this.HashString.ToLowerInvariant(), results[0]);
        }

        [Fact]
        public void CalcEmptyString()
        {
            IList<string> results = this.Runner.Run(StringOpt, EmptyStr);
            Assert.Equal(1, results.Count);
            Assert.Equal(this.EmptyStringHash, results[0]);
        }

        [Fact]
        public void CrackString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, MaxOpt, "3");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Fact]
        public void CrackStringSingleThread()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, MaxOpt, "3", "-T", "1");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Theory]
        [InlineData("-1")]
        [InlineData("10000")]
        public void CrackStringBadThreads(string threads)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, MaxOpt, "3", "-T", threads);
            Assert.Equal(4, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, this.InitialString), results[3]);
        }

        [Fact]
        public void CrackEmptyString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.EmptyStringHash);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "Empty string"), results[2]);
        }

        [Fact]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString.ToLowerInvariant(), MaxOpt, "3", DictOpt, this.InitialString);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Theory]
        [InlineData("123")]
        [InlineData("0-9")]
        [InlineData("0-9a-z")]
        [InlineData("0-9A-Z")]
        [InlineData("0-9a-zA-Z")]
        public void CrackStringSuccessUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, DictOpt, dict, MaxOpt, "3");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Theory]
        [InlineData("a-zA-Z")]
        [InlineData("a-z")]
        [InlineData("A-Z")]
        [InlineData("abc")]
        public void CrackStringFailureUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, DictOpt, dict, MaxOpt, "3");
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }


        [Fact]
        public void CrackStringTooShortLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, MaxOpt,
                                                (this.InitialString.Length - 1).ToString(), DictOpt, this.InitialString);
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }

        [Fact]
        public void CrackStringTooLongMinLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.HashString, MinOpt,
                                                (this.InitialString.Length + 1).ToString(), MaxOpt,
                                                (this.InitialString.Length + 1).ToString(), DictOpt, this.InitialString);
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }

        [Fact]
        public void CrackStringSingleCharStringWithMaxOpt()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.Hash.MiddlePartStringHash, MaxOpt, "2");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }
        
        [Fact]
        public void CrackStringSingleCharStringWithMaxOptOnSingleThread()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.Hash.MiddlePartStringHash, MaxOpt, "2", "-T", "1");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }
        
        [Fact]
        public void CrackStringSingleCharStringWithMaxOptAndNonDefaultDict()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, this.Hash.MiddlePartStringHash, MaxOpt, "2", DictOpt, "[0-9]");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "2"), results[2]);
        }

        [Fact]
        public void TestPerformance()
        {
            IList<string> results = this.Runner.Run(PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "12345"), results[2]);
        }
    }
}