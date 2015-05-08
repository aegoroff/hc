/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2015 Alexander Egorov
 */

using System.Collections.Generic;
using Xunit;

namespace _tst.net
{
    [Trait("Mode", "query")]
    public abstract class QueryStringTests<T> : StringTests<T>
        where T : Architecture, new()
    {
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        private const string HashStringCrackQueryTpl = "for string s from hash '{0}' do crack {1};";

        private const string SyntaxOnlyOpt = "--syntaxonly";
        private const string QueryOpt = "-C";
        private const string LowerOpt = "-l";
        private const string NoProbeOpt = "--noprobe";

        protected QueryStringTests() : base(new T())
        {
        }

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), NoProbeOpt);
        }

        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        protected override IList<string> RunEmptyStringCrack(Hash h)
        {
            return this.RunQuery(HashStringCrackQueryTpl, h.EmptyStringHash, h.Algorithm);
        }

        protected override IList<string> RunStringCrack(Hash h)
        {
            return RunQuery(HashStringCrackQueryTpl, h.HashString, h.Algorithm);
        }

        protected override IList<string> RunStringCrackTooShort(Hash h)
        {
            return RunQuery("for string s from hash '{0}' let s.max = {1} do crack {2};", h.HashString, h.InitialString.Length - 1, h.Algorithm);
        }

        protected override IList<string> RunStringCrackTooMinLength(Hash h)
        {
            return RunQuery("for string s from hash '{0}' let s.min = {1}, s.max = {2}, s.dict = '123' do crack {3};", h.HashString, h.InitialString.Length + 1, h.InitialString.Length + 2, h.Algorithm);
        }
        
        protected override IList<string> RunStringCrackLowCaseHash(Hash h)
        {
            return RunQuery("for string s from hash '{0}' let s.dict = '{2}' do crack {1};", h.HashString.ToLowerInvariant(), h.Algorithm, h.InitialString);
        }

        protected override IList<string> RunStringHash(Hash h)
        {
            return this.RunQuery(HashStringQueryTpl, h.InitialString, h.Algorithm);
        }

        protected override IList<string> RunStringHashLowCase(Hash h)
        {
            return RunQueryWithOpt(HashStringQueryTpl, LowerOpt, h.InitialString, h.Algorithm);
        }

        protected override IList<string> RunEmptyStringHash(Hash h)
        {
            return this.RunQuery(HashStringQueryTpl, string.Empty, h.Algorithm);
        }

        protected override IList<string> RunCrackStringUsingNonDefaultDictionary(Hash h, string dict)
        {
            return RunQuery("for string s from hash '{0}' let s.dict = '{1}', s.max = 2, s.min = 2 do crack {2};", h.StartPartStringHash, dict, h.Algorithm);
        }

        [Theory, MemberData("Hashes")]
        public void ValidateSyntaxOption(Hash h)
        {
            IList<string> results = this.Runner.Run(QueryOpt, string.Format(HashStringQueryTpl, h.InitialString, h.Algorithm), SyntaxOnlyOpt);
            Assert.Equal(0, results.Count);
        }

        [Trait("Type", "crack")]
        [Fact]
        public void CrackNonAsciiString()
        {
            IList<string> results = RunQuery("for string s from hash '327108899019B3BCFFF1683FBFDAF226' let s.dict ='еграб' do crack md5;");
            Asserts.StringMatching(results[1], "Initial string is: *");
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void VariableRedefinition()
        {
            IList<string> results = RunQuery("let h = '202CB962AC59075B964B07152D234B70';let x = '0-9';for string s from hash h let s.dict = x do crack md5;let x = '45';for string s from hash h let s.dict = x do crack md5;");
            Assert.Equal("Initial string is: 123", results[1]);
            Assert.Equal("Nothing found", results[3]);
            Assert.Equal(4, results.Count);
        }

        [Trait("Type", "crack")]
        [Theory, MemberData("HashesAndNonDefaultDictSmall")]
        public void CrackStringSuccessUsingNonDefaultDictionaryWithVar(Hash h, string dict)
        {
            IList<string> results = RunQuery("let x = '{1}';for string s from hash '{0}' let s.dict = x do crack {2};", h.HashString, dict, h.Algorithm);
            Assert.Equal(string.Format(RestoredStringTemplate, h.InitialString), results[1]);
            Assert.Equal(2, results.Count);
        }

        public static IEnumerable<object[]> HashesAndNonDefaultDictSmall
        {
            get { return CreateProperty(new object[] { "123", "0-9" }); }
        }
    }
}