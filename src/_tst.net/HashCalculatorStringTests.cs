/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using System.Collections.Generic;
using NUnit.Framework;

namespace _tst.net
{
    public abstract class HashCalculatorStringTests<THash> where THash : Hash, new()
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

        protected abstract string PathTemplate { get; }

        protected Hash Hash { get; set; }

        protected ProcessRunner Runner { get; set; }

        protected string InitialString
        {
            get { return this.Hash.InitialString; }
        }

        protected virtual string Executable
        {
            get { return this.Hash.Executable; }
        }

        protected string HashString
        {
            get { return this.Hash.HashString; }
        }

        protected string EmptyStringHash
        {
            get { return this.Hash.EmptyStringHash; }
        }

        [SetUp]
        public void Setup()
        {
            this.Runner = new ProcessRunner(string.Format(PathTemplate, Executable));
        }

        [TestFixtureSetUp]
        public void TestFixtureSetup()
        {
            this.Hash = new THash();
        }

        [Test]
        public void CalcString()
        {
            IList<string> results = this.Runner.Run(StringOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString));
        }

        [Test]
        public void CalcStringLowCaseOutput()
        {
            IList<string> results = this.Runner.Run(StringOpt, InitialString, LowCaseOpt);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
        }

        [Test]
        public void CalcEmptyString()
        {
            IList<string> results = this.Runner.Run(StringOpt, EmptyStr);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(EmptyStringHash));
        }

        [Test]
        public void CrackString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [Test]
        public void CrackStringSingleThread()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt, "3", "-T", "1");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("-1")]
        [TestCase("10000")]
        public void CrackStringBadThreads(string threads)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt, "3", "-T", threads);
            Assert.That(results.Count, Is.EqualTo(4));
            Assert.That(results[3], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [Test]
        public void CrackEmptyString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, EmptyStringHash);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "Empty string")));
        }

        [Test]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString.ToLowerInvariant(), MaxOpt, "3", DictOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("123")]
        [TestCase("0-9")]
        [TestCase("0-9a-z")]
        [TestCase("0-9A-Z")]
        [TestCase("0-9a-zA-Z")]
        public void CrackStringSuccessUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, DictOpt, dict, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("a-zA-Z")]
        [TestCase("a-z")]
        [TestCase("A-Z")]
        [TestCase("abc")]
        public void CrackStringFailureUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, DictOpt, dict, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }


        [Test]
        public void CrackStringTooShortLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt,
                                                (InitialString.Length - 1).ToString(), DictOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [Test]
        public void CrackStringTooLongMinLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MinOpt,
                                                (InitialString.Length + 1).ToString(), MaxOpt,
                                                (InitialString.Length + 1).ToString(), DictOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [Test]
        public void TestPerformance()
        {
            IList<string> results = this.Runner.Run(PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "12345")));
        }
    }
}