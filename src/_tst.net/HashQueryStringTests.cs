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
    public abstract class HashQueryStringTests<T, THash> : StringTests<T, THash>
        where T : Architecture, new()
        where THash : Hash, new()
    {
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        private const string HashStringCrackQueryTpl = "for string s from hash '{0}' do crack {1};";
        private const string RestoredStringTemplate = "Initial string is: {0}";
        private const string NothingFound = "Nothing found";


        private const string SyntaxOnlyOpt = "--syntaxonly";
        private const string QueryOpt = "-C";
        private const string LowerOpt = "-l";
        private const string NoProbeOpt = "--noprobe";

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), NoProbeOpt);
        }

        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        [Fact]
        public void ValidateSyntaxOption()
        {
            IList<string> results = this.Runner.Run(QueryOpt, string.Format(HashStringQueryTpl, InitialString, Hash.Algorithm), SyntaxOnlyOpt);
            Assert.Equal(0, results.Count);
        }

        [Fact]
        public void CalcString()
        {
            IList<string> results = RunQuery(HashStringQueryTpl, InitialString, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(HashString, results[0]);
        }

        [Fact]
        public void CalcStringLowCase()
        {
            IList<string> results = RunQueryWithOpt(HashStringQueryTpl, LowerOpt, InitialString, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(HashString.ToLowerInvariant(), results[0]);
        }

        [Fact]
        public void CalcEmptyString()
        {
            IList<string> results = RunQuery(HashStringQueryTpl, string.Empty, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(EmptyStringHash, results[0]);
        }

        [Fact]
        public void CrackString()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, HashString, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Fact]
        public void CrackNonAsciiString()
        {
            IList<string> results = RunQuery("for string s from hash '327108899019B3BCFFF1683FBFDAF226' let s.dict ='еграб' do crack md5;");
            Assert.Equal(3, results.Count);
            Asserts.StringMatching(results[2], "Initial string is: .*");
        }

        [Fact]
        public void VariableRedefinition()
        {
            IList<string> results = RunQuery("let h = '202CB962AC59075B964B07152D234B70';let x = '0-9';for string s from hash h let s.dict = x do crack md5;let x = '45';for string s from hash h let s.dict = x do crack md5;");
            Assert.Equal(6, results.Count);
            Assert.Equal("Initial string is: 123", results[2]);
            Assert.Equal("Nothing found", results[5]);
        }

        [Fact]
        public void CrackEmptyString()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, EmptyStringHash, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, "Empty string"), results[2]);
        }

        [Fact]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, HashString.ToLowerInvariant(), Hash.Algorithm);
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
            IList<string> results = RunQuery("for string s from hash '{0}' let s.dict = '{1}' do crack {2};", HashString, dict, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(string.Format(RestoredStringTemplate, InitialString), results[2]);
        }

        [Theory]
        [InlineData("123")]
        [InlineData("0-9")]
        public void CrackStringSuccessUsingNonDefaultDictionaryWithVar(string dict)
        {
            IList<string> results = RunQuery("let x = '{1}';for string s from hash '{0}' let s.dict = x do crack {2};", HashString, dict, Hash.Algorithm);
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
            IList<string> results = RunQuery("for string s from hash '{0}' let s.dict = '{1}', s.max = 3 do crack {2};", HashString, dict, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }

        [Fact]
        public void CrackStringTooShortLength()
        {
            IList<string> results = RunQuery("for string s from hash '{0}' let s.max = {1} do crack {2};", HashString, InitialString.Length - 1, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }

        [Fact]
        public void CrackStringTooLongMinLength()
        {
            IList<string> results = RunQuery("for string s from hash '{0}' let s.min = {1}, s.max = {2}, s.dict = '123' do crack {3};", HashString, InitialString.Length + 1, InitialString.Length + 2, Hash.Algorithm);
            Assert.Equal(3, results.Count);
            Assert.Equal(NothingFound, results[2]);
        }
    }
}