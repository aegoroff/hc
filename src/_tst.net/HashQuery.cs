/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2007-2011 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
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
        private const string TimeOpt = "-t";
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        private const string HashStringCrackQueryTpl = "for string s from hash '{0}' do crack {1};";
        private const string RestoredStringTemplate = "Initial string is: {0}";
        
        private const string FileResultTpl = @"{0} | {2} bytes | {1}";
        private const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec$";
        private const string HashFileQueryTpl = "for file f from '{0}' do {1};";
        
        private const string QueryFile = BaseTestDir + Slash + "hl.hlq";

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

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters));
        }
        
        IList<string> RunFileQuery(string template, params object[] parameters)
        {
            File.WriteAllText(QueryFile, string.Format(template, parameters));
            return this.Runner.Run("-f", QueryFile);
        }
        
        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        [TearDown]
        public void Teardown()
        {
            if (File.Exists(QueryFile))
            {
                File.Delete(QueryFile);
            }
        }

        [Test]
        public void FileQuery()
        {
            IList<string> results = RunFileQuery(HashStringQueryTpl, InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString));
        }
        
        [Test]
        public void FileWithSeveralQueries()
        {
            IList<string> results = RunFileQuery(HashStringQueryTpl + Environment.NewLine + HashStringQueryTpl, InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(2));
            Assert.That(results[0], Is.EqualTo(HashString));
            Assert.That(results[1], Is.EqualTo(HashString));
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
            IList<string> results = RunQueryWithOpt(HashStringQueryTpl, "-l", InitialString, Hash.Algorithm);
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
        public void CrackEmptyString()
        {
            IList<string> results = RunQuery(HashStringCrackQueryTpl, HashEmptyString, Hash.Algorithm);
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

        [Test]
        public void CalcFile()
        {
            IList<string> results = RunQuery(HashFileQueryTpl, NotEmptyFile, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [Test]
        public void CalcFileTime()
        {
            IList<string> results = RunQueryWithOpt(HashFileQueryTpl, TimeOpt, NotEmptyFile, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.StringMatching(FileResultTimeTpl));
        }

        [Test]
        public void CalcFileLimit()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1} do {2};", NotEmptyFile, 2, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, StartPartStringHash, InitialString.Length)));
        }

        [Test]
        public void CalcFileOffset()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", NotEmptyFile, 1, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, TrailPartStringHash, InitialString.Length)));
        }

        [Test]
        public void CalcFileLimitAndOffset()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {1} do {2};", NotEmptyFile, 1, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, MiddlePartStringHash, InitialString.Length)));
        }

        [Test]
        public void CalcFileOffsetGreaterThenFileSIze()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", NotEmptyFile, 4, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, "Offset is greater then file size",
                                                 InitialString.Length)));
        }

        [Test]
        public void CalcBigFile()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' do {1};", file, Hash.Algorithm);
                Assert.That(results.Count, Is.EqualTo(1));
                StringAssert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Test]
        public void CalcBigFileWithOffset()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", file, 1024, Hash.Algorithm);
                Assert.That(results.Count, Is.EqualTo(1));
                StringAssert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Test]
        public void CalcBigFileWithLimitAndOffset()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {2} do {3};", file, 1048500, 1024, Hash.Algorithm);
                Assert.That(results.Count, Is.EqualTo(1));
                StringAssert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Test]
        public void CalcUnexistFile()
        {
            const string unexist = "u";
            IList<string> results = RunQuery("for file f from '{0}' do {1};", unexist, Hash.Algorithm); ;
            Assert.That(results.Count, Is.EqualTo(1));
            string en = string.Format("{0} | The system cannot find the file specified.  ", unexist);
            string ru = string.Format("{0} | Не удается найти указанный файл.  ", unexist);
            Assert.That(results[0], Is.InRange(en, ru));
        }

        [Test]
        public void CalcEmptyFile()
        {
            IList<string> results = RunQuery("for file f from '{0}' do {1};", EmptyFile, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
        }
    }
}