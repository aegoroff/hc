/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2007-2011 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
        private const string EmptyFileName = "e_mpty";
        private const string NotEmptyFileName = "n_otempty";
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
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes$";
        
        private const string QueryFile = BaseTestDir + Slash + "hl.hlq";
        private const string ValidationQueryTemplate = "for file f from '{0}' let f.{1} = '{2}' do validate;";
        private const string SearchFileQueryTemplate = "for file f from dir '{0}' where f.{1} == '{2}' do find;";
        private const string CalculateFileQueryTemplate = "for file f from '{0}' do {1};";
        private const string NothingFound = "Nothing found";

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
            get { return "hq.exe"; }
        }

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return Runner.Run(QueryOpt, string.Format(template, parameters));
        }
        
        IList<string> RunFileQuery(string template, params object[] parameters)
        {
            File.WriteAllText(QueryFile, string.Format(template, parameters));
            return Runner.Run("-f", QueryFile);
        }
        
        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
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
            const int count = 2;
            IList<string> results = RunFileQuery(SeveralQueries(count), InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(count));
            Assert.That(results[0], Is.EqualTo(HashString));
            Assert.That(results[1], Is.EqualTo(HashString));
        }
        
        [Test]
        public void TooBigFileWithSeveralQueries()
        {
            IList<string> results = RunFileQuery(SeveralQueries(15000), InitialString, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo("Too much statements. Max allowed 10000"));
        }

        private static string SeveralQueries( int count )
        {
            return string.Join(Environment.NewLine, CreateSeveralStringQueries(count));
        }

        static IEnumerable<string> CreateSeveralStringQueries(int count)
        {
            for (int i = 0; i < count; i++)
            {
                yield return HashStringQueryTpl;
            }
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
        public void CrackNonAsciiString()
        {
            IList<string> results = RunQuery("for string s from hash '327108899019B3BCFFF1683FBFDAF226' let s.dict ='еграб' do crack md5;");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo("Initial string is: егр"));
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

        [Test]
        public void CalcFile()
        {
            IList<string> results = RunQuery(CalculateFileQueryTemplate, NotEmptyFile, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [Test]
        public void CalcFileTime()
        {
            IList<string> results = RunQueryWithOpt(CalculateFileQueryTemplate, TimeOpt, NotEmptyFile, Hash.Algorithm);
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
                IList<string> results = RunQuery(CalculateFileQueryTemplate, file, Hash.Algorithm);
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
            IList<string> results = RunQuery(CalculateFileQueryTemplate, unexist, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            string en = string.Format("{0} | The system cannot find the file specified.  ", unexist);
            string ru = string.Format("{0} | Не удается найти указанный файл.  ", unexist);
            Assert.That(results[0], Is.InRange(en, ru));
        }

        [Test]
        public void CalcEmptyFile()
        {
            IList<string> results = RunQuery(CalculateFileQueryTemplate, EmptyFile, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
        }

        [Test]
        public void CalcDir()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' do {1};", BaseTestDir, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(2));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
            Assert.That(results[1],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [Test]
        public void CalcDirRecursivelyManySubs()
        {
            const string sub2Suffix = "2";
            Directory.CreateDirectory(SubDir + sub2Suffix);

            CreateEmptyFile(SubDir + sub2Suffix + Slash + EmptyFileName);
            CreateNotEmptyFile(SubDir + sub2Suffix + Slash + NotEmptyFileName);

            try
            {
                IList<string> results = RunQuery("for file f from dir '{0}' do {1} withsubs;", BaseTestDir, Hash.Algorithm);
                Assert.That(results.Count, Is.EqualTo(6));
            }
            finally
            {
                Directory.Delete(SubDir + sub2Suffix, true);
            }
        }

        [Test]
        public void CalcDirIncludeFilter()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.name ~ '{1}' do {2};", BaseTestDir, EmptyFileName, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
        }

        [Test]
        public void CalcDirExcludeFilter()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.name !~ '{1}' do {2};", BaseTestDir, EmptyFileName, Hash.Algorithm);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [TestCase(0, "for file f from dir '{0}' where f.name ~ '{1}' and f.name !~ '{1}' do {2};", BaseTestDir, EmptyFileName)]
        [TestCase(0, "for file f from dir '{0}' where f.name !~ '{1}' and f.name !~ '{2}' do {3};", BaseTestDir, EmptyFileName, NotEmptyFileName)]
        [TestCase(2, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3};", BaseTestDir, EmptyFileName, NotEmptyFileName)]
        [TestCase(2, "for file f from dir '{0}' where f.name ~ '{1}' do {2} withsubs;", BaseTestDir, EmptyFileName)]
        [TestCase(2, "for file f from dir '{0}' where f.name !~ '{1}' do {2} withsubs;", BaseTestDir, EmptyFileName)]
        [TestCase(4, "for file f from dir '{0}' do {1} withsubs;", new object[] { BaseTestDir })]
        [TestCase(2, "for file f from dir '{0}' where f.size == 0 do {1} withsubs;", new object[] { BaseTestDir })]
        [TestCase(1, "for file f from dir '{0}' where f.size == 0 do {1};", new object[] { BaseTestDir })]
        [TestCase(2, "for file f from dir '{0}' where f.size != 0 do {1} withsubs;", new object[] { BaseTestDir })]
        [TestCase(1, "for file f from dir '{0}' where f.size != 0 do {1};", new object[] { BaseTestDir })]
        [TestCase(4, "for file f from dir '{0}' where f.size == 0 or f.name ~ '{1}' do {2} withsubs;", BaseTestDir, NotEmptyFileName)]
        [TestCase(2, "for file f from dir '{0}' where f.size == 0 or not f.name ~ '{1}' do {2} withsubs;", BaseTestDir, NotEmptyFileName)]
        [TestCase(0, "for file f from dir '{0}' where f.size == 0 and f.name ~ '{1}' do {2} withsubs;", BaseTestDir, NotEmptyFileName)]
        [TestCase(4, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3} withsubs;", BaseTestDir, EmptyFileName, NotEmptyFileName)]
        [TestCase(2, "for file f from dir '{0}' where f.name ~ '{1}' or (f.name ~ '{2}' and f.size == 0) do {3} withsubs;", BaseTestDir, EmptyFileName, NotEmptyFileName)]
        [TestCase(3, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) or f.path ~ '{3}' do {4} withsubs;", BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*")]
        [TestCase(1, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) and f.path ~ '{3}' do {4} withsubs;", BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*")]
        [TestCase(0, "for file f from dir '{0}' where f.name ~ '' do {1} withsubs;", new object[] { BaseTestDir })]
        [TestCase(4, "for file f from dir '{0}' where f.name !~ '' do {1} withsubs;", new object[] { BaseTestDir })]
        [TestCase(4, "for file f from dir '{0}' where f.name ~ 'mpty' do {1} withsubs;", new object[] { BaseTestDir })]
        public void CalcDir(int countResults, string template, params object[] parameters)
        {
            List<object> p = parameters.ToList();
            p.Add(Hash.Algorithm);
            IList<string> results = RunQuery(template, p.ToArray());
            Assert.That(results.Count, Is.EqualTo(countResults));
        }

        [Test]
        public void SearchFile()
        {
            IList<string> results = RunQuery(SearchFileQueryTemplate, BaseTestDir, Hash.Algorithm, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length)));
        }
        
        [Test]
        public void SearshFileNotEq()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} != '{2}' do find;", BaseTestDir, Hash.Algorithm, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileSearchTpl, EmptyFile, 0)));
        }

        [Test]
        public void SearchFileTimed()
        {
            IList<string> results = RunQueryWithOpt(SearchFileQueryTemplate, TimeOpt, BaseTestDir, Hash.Algorithm, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.StringMatching(FileSearchTimeTpl));
        }

        [Test]
        public void SearchFileRecursively()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} == '{2}' do find withsubs;", BaseTestDir, Hash.Algorithm, HashString);
            Assert.That(results.Count, Is.EqualTo(2));
        }
        
        [Test]
        public void SearchFileOffsetMoreThenFileSize()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 100 and f.{1} == '{2}' do find withsubs;", BaseTestDir, Hash.Algorithm, HashString);
            Assert.That(results, Is.Empty);
        }

        [Test]
        public void SearchFileSeveralHashes()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", BaseTestDir, Hash.Algorithm, TrailPartStringHash, MiddlePartStringHash);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length)));
        }
        
        [Test]
        public void SearchFileSeveralHashesNoResults()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", BaseTestDir, Hash.Algorithm, TrailPartStringHash, HashString);
            Assert.That(results, Is.Empty);
        }

        [Test]
        public void ValidateFileSuccess()
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, NotEmptyFile, Hash.Algorithm, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, "File is valid", InitialString.Length)));
        }

        [Test]
        public void ValidateFileFailure()
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, NotEmptyFile, Hash.Algorithm, TrailPartStringHash);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, "File is invalid", InitialString.Length)));
        }
    }
}