/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using System.Collections.Generic;
using NUnit.Framework;

namespace _tst.net
{
    public abstract class HashQueryStringTests<THash> where THash : Hash, new()
    {
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        private const string HashStringCrackQueryTpl = "for string s from hash '{0}' do crack {1};";
        private const string RestoredStringTemplate = "Initial string is: {0}";
        private const string NothingFound = "Nothing found";


        private const string SyntaxOnlyOpt = "--syntaxonly";
        private const string QueryOpt = "-C";
        private const string LowerOpt = "-l";
        private const string NoProbeOpt = "--noprobe";
        
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

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return Runner.Run(QueryOpt, string.Format(template, parameters), NoProbeOpt);
        }

        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        [Test]
        public void ValidateSyntaxOption()
        {
            IList<string> results = Runner.Run(QueryOpt, string.Format(HashStringQueryTpl, InitialString, Hash.Algorithm), SyntaxOnlyOpt);
            Assert.That(results.Count, Is.EqualTo(0));
        }

        [Test]
        public void CalcString()
        {
            IList<string> results = RunQuery(HashStringQueryTpl, InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString));
        }

        [Test]
        public void CalcStringLowCase()
        {
            IList<string> results = RunQueryWithOpt(HashStringQueryTpl, LowerOpt, InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
        }

        [Test]
        public void CalcEmptyString()
        {
            IList<string> results = RunQuery(HashStringQueryTpl, string.Empty, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(EmptyStringHash));
        }

        [Test]
        public void CrackString()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, HashString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [Test]
        public void CrackNonAsciiString()
        {
            IList<string> results = RunQuery("for string s from hash '327108899019B3BCFFF1683FBFDAF226' let s.dict ='еграб' do crack md5;");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo("Initial string is: егр"));
        }

        [Test]
        public void VariableRedefinition()
        {
            IList<string> results = RunQuery("let h = '202CB962AC59075B964B07152D234B70';let x = '0-9';for string s from hash h let s.dict = x do crack md5;let x = '45';for string s from hash h let s.dict = x do crack md5;");
            Assert.That(results.Count, Is.EqualTo(6));
            Assert.That(results[2], Is.EqualTo("Initial string is: 123"));
            Assert.That(results[5], Is.EqualTo("Nothing found"));
        }

        [Test]
        public void CrackEmptyString()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, EmptyStringHash, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "Empty string")));
        }

        [Test]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, HashString.ToLowerInvariant(), Hash.Algorithm);
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
            IList<string> results = RunQuery("for string s from hash '{0}' let s.dict = '{1}' do crack {2};", HashString, dict, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("123")]
        [TestCase("0-9")]
        public void CrackStringSuccessUsingNonDefaultDictionaryWithVar(string dict)
        {
            IList<string> results = RunQuery("let x = '{1}';for string s from hash '{0}' let s.dict = x do crack {2};", HashString, dict, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("a-zA-Z")]
        [TestCase("a-z")]
        [TestCase("A-Z")]
        [TestCase("abc")]
        public void CrackStringFailureUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = RunQuery("for string s from hash '{0}' let s.dict = '{1}', s.max = 3 do crack {2};", HashString, dict, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [Test]
        public void CrackStringTooShortLength()
        {
            IList<string> results = RunQuery("for string s from hash '{0}' let s.max = {1} do crack {2};", HashString, InitialString.Length - 1, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [Test]
        public void CrackStringTooLongMinLength()
        {
            IList<string> results = RunQuery("for string s from hash '{0}' let s.min = {1}, s.max = {2}, s.dict = '123' do crack {3};", HashString, InitialString.Length + 1, InitialString.Length + 2, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }
    }
}