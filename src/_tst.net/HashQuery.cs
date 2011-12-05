/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2007-2011 Alexander Egorov
 */

using System.Collections.Generic;
using NUnit.Framework;

namespace _tst.net
{
    [TestFixture(typeof(Md4))]
    [TestFixture(typeof(Md5))]
    [TestFixture(typeof(Sha1))]
    [TestFixture(typeof(Sha256))]
    [TestFixture(typeof(Sha384))]
    [TestFixture(typeof(Sha512))]
    [TestFixture(typeof(Whirlpool))]
    [TestFixture(typeof(Crc32))]
    public class HashQuery<THash> : HashBase<THash> where THash : Hash, new()
    {
        private const string EmptyFileName = "empty";
        private const string NotEmptyFileName = "notempty";
        private const string Slash = @"\";
        private const string BaseTestDir = @"C:\_tst.net";
        private const string NotEmptyFile = BaseTestDir + Slash + NotEmptyFileName;
        private const string EmptyFile = BaseTestDir + Slash + EmptyFileName;
        private const string SubDir = BaseTestDir + Slash + "sub";
        private const string QueryOpt = "-q";
        private const string HashStringTpl = "for string '{0}' do {1};";
        private const string HashStringCrackTpl = "for string s from hash '{0}' do crack {1};";
        private const string RestoredStringTemplate = "Initial string is: {0}";

        protected override string EmptyFileNameProp
        {
            get { return EmptyFileName; }
        }

        protected override string EmptyFileProp
        {
            get { return EmptyFile; }
        }

        protected override string NotEmptyFileNameProp
        {
            get { return NotEmptyFileName; }
        }

        protected override string NotEmptyFileProp
        {
            get { return NotEmptyFile; }
        }

        protected override string BaseTestDirProp
        {
            get { return BaseTestDir; }
        }

        protected override string SubDirProp
        {
            get { return SubDir; }
        }

        protected override string SlashProp
        {
            get { return Slash; }
        }

        protected override string Executable
        {
            get { return "hl.exe"; }
        }

        IList<string> RunQuery(string template, params string[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters));
        }

        [Test]
        public void CalcString()
        {
            IList<string> results = RunQuery(HashStringTpl, InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString));
        }
        
        [Test]
        public void CalcStringLowCase()
        {
            IList<string> results = this.Runner.Run("-l", QueryOpt, string.Format(HashStringTpl, InitialString, Hash.Algorithm));
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
        }

        [Test]
        public void CalcEmptyString()
        {
            IList<string> results = RunQuery(HashStringTpl, string.Empty, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(EmptyStringHash));
        }

        [Test]
        public void CrackString()
        {
            IList<string> results = RunQuery(HashStringCrackTpl, HashString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }
        
        [Test]
        public void CrackEmptyString()
        {
            IList<string> results = RunQuery(HashStringCrackTpl, HashEmptyString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "Empty string")));
        }

        [Test]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = RunQuery(HashStringCrackTpl, HashString.ToLowerInvariant(), Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }
    }
}