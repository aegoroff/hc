/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2013 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Extensions;

namespace _tst.net
{
    public abstract class HashQueryFileTests<T, THash> : FileTests<T, THash>, IDisposable
        where T : Architecture, new()
        where THash : Hash, new()
    {
        private const string EmptyFileName = "e_mpty";
        private const string NotEmptyFileName = "n_otempty";
        private const string NotEmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + NotEmptyFileName;
        private const string EmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + EmptyFileName;
        private const string QueryOpt = "-C";
        private const string FileOpt = "-F";
        private const string ParamOpt = "-P";
        private const string TimeOpt = "-t";
        private const string NoProbeOpt = "--noprobe";
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        
        
        private const string FileResultTpl = @"{0} | {2} bytes | {1}";
        private const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes$";
        
        private const string QueryFile = FileFixture.BaseTestDir + FileFixture.Slash + "hl.hlq";
        private const string ValidationQueryTemplate = "for file f from '{0}' let f.{1} = '{2}' do validate;";
        private const string SearchFileQueryTemplate = "for file f from dir '{0}' where f.{1} == '{2}' do find;";
        private const string CalculateFileQueryTemplate = "for file f from '{0}' do {1};";
        

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

        IList<string> RunQuery(string template, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), NoProbeOpt);
        }
        
        IList<string> RunValidatingQuery(string path, string template, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), ParamOpt, path);
        }
        
        IList<string> RunFileQuery(string template, params object[] parameters)
        {
            File.WriteAllText(QueryFile, string.Format(template, parameters));
            return this.Runner.Run(FileOpt, QueryFile);
        }
        
        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        public void Dispose()
        {
            if (File.Exists(QueryFile))
            {
                File.Delete(QueryFile);
            }
        }

        [Fact]
        public void FileQuery()
        {
            IList<string> results = RunFileQuery(HashStringQueryTpl, InitialString, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(HashString, results[0]);
        }
        
        [Fact]
        public void FileWithSeveralQueries()
        {
            const int count = 2;
            IList<string> results = RunFileQuery(SeveralQueries(count), InitialString, Hash.Algorithm);
            Assert.Equal(count, results.Count);
            Assert.Equal(HashString, results[0]);
            Assert.Equal(HashString, results[1]);
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

        [Fact]
        public void CalcFile()
        {
            IList<string> results = RunQuery(CalculateFileQueryTemplate, NotEmptyFile, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[0]);
        }

        [Fact]
        public void CalcFileTime()
        {
            IList<string> results = RunQueryWithOpt(CalculateFileQueryTemplate, TimeOpt, NotEmptyFile, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileResultTimeTpl);
        }

        [Theory]
        [InlineData(1, "File is valid")]
        [InlineData(0, "File is invalid")]
        public void ValidateParameterFile(int offset, string result)
        {
            IList<string> results = RunValidatingQuery(NotEmptyFile, "for file f from parameter where f.offset == {0} and f.{1} == '{2}' do validate;", offset, Hash.Algorithm, TrailPartStringHash);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, result, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileLimit()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1} do {2};", NotEmptyFile, 2, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, StartPartStringHash, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileOffset()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", NotEmptyFile, 1, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, TrailPartStringHash, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileLimitAndOffset()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {1} do {2};", NotEmptyFile, 1, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, MiddlePartStringHash, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileOffsetGreaterThenFileSIze()
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", NotEmptyFile, 4, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "Offset is greater then file size", InitialString.Length), results[0]);
        }

        [Fact]
        public void CalcBigFile()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery(CalculateFileQueryTemplate, file, Hash.Algorithm);
                Assert.Equal(1, results.Count);
                Assert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Fact]
        public void CalcBigFileWithOffset()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", file, 1024, Hash.Algorithm);
                Assert.Equal(1, results.Count);
                Assert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Fact]
        public void CalcBigFileWithLimitAndOffset()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {2} do {3};", file, 1048500, 1024, Hash.Algorithm);
                Assert.Equal(1, results.Count);
                Assert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Fact]
        public void CalcUnexistFile()
        {
            const string unexist = "u";
            IList<string> results = RunQuery(CalculateFileQueryTemplate, unexist, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            string en = string.Format("{0} | The system cannot find the file specified.  ", unexist);
            string ru = string.Format("{0} | Не удается найти указанный файл.  ", unexist);
            Assert.Contains(results[0], new[] { en, ru });
        }

        [Fact]
        public void CalcEmptyFile()
        {
            IList<string> results = RunQuery(CalculateFileQueryTemplate, EmptyFile, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
        }

        [Fact]
        public void CalcDir()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' do {1};", FileFixture.BaseTestDir, Hash.Algorithm);
            Assert.Equal(2, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[1]);
        }

        [Fact]
        public void CalcDirRecursivelyManySubs()
        {
            const string sub2Suffix = "2";
            Directory.CreateDirectory(FileFixture.SubDir + sub2Suffix);

            CreateEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + EmptyFileName);
            CreateNotEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + NotEmptyFileName);

            try
            {
                IList<string> results = RunQuery("for file f from dir '{0}' do {1} withsubs;", FileFixture.BaseTestDir, Hash.Algorithm);
                Assert.Equal(6, results.Count);
            }
            finally
            {
                Directory.Delete(FileFixture.SubDir + sub2Suffix, true);
            }
        }

        [Fact] // TODO: Make theory
        public void CalcDirIncludeFilter()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.name ~ '{1}' do {2};", FileFixture.BaseTestDir, EmptyFileName, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcDirIncludeFilterWithVar()
        {
            IList<string> results = RunQuery("let x = '{0}';let y = '{1}';for file f from dir x where f.name ~ y do {2};", FileFixture.BaseTestDir, EmptyFileName, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
        }

        [Fact]
        public void CalcDirExcludeFilter()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.name !~ '{1}' do {2};", FileFixture.BaseTestDir, EmptyFileName, Hash.Algorithm);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[0]);
        }

        [Theory]
        [InlineData(0, "for file f from dir '{0}' where f.name ~ '{1}' and f.name !~ '{1}' do {2};", new object[] { FileFixture.BaseTestDir, EmptyFileName })]
        [InlineData(0, "for file f from dir '{0}' where f.name !~ '{1}' and f.name !~ '{2}' do {3};", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName })]
        [InlineData(2, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3};", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName })]
        [InlineData(2, "for file f from dir '{0}' where f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName })]
        [InlineData(2, "for file f from dir '{0}' where f.name !~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName })]
        [InlineData(4, "for file f from dir '{0}' do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        [InlineData(2, "for file f from dir '{0}' where f.size == 0 do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        [InlineData(1, "for file f from dir '{0}' where f.size == 0 do {1};", new object[] { FileFixture.BaseTestDir })]
        [InlineData(2, "for file f from dir '{0}' where f.size != 0 do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        [InlineData(1, "for file f from dir '{0}' where f.size != 0 do {1};", new object[] { FileFixture.BaseTestDir })]
        [InlineData(4, "for file f from dir '{0}' where f.size == 0 or f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName })]
        [InlineData(2, "for file f from dir '{0}' where f.size == 0 or not f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName })]
        [InlineData(0, "for file f from dir '{0}' where f.size == 0 and f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName })]
        [InlineData(4, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName })]
        [InlineData(2, "for file f from dir '{0}' where f.name ~ '{1}' or (f.name ~ '{2}' and f.size == 0) do {3} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName })]
        [InlineData(3, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) or f.path ~ '{3}' do {4} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*" })]
        [InlineData(1, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) and f.path ~ '{3}' do {4} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*" })]
        [InlineData(0, "for file f from dir '{0}' where f.name ~ '' do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        [InlineData(4, "for file f from dir '{0}' where f.name !~ '' do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        [InlineData(4, "for file f from dir '{0}' where f.name ~ 'mpty' do {1} withsubs;", new object[] { FileFixture.BaseTestDir })]
        public void CalcDirTheory(int countResults, string template, params object[] parameters)
        {
            List<object> p = parameters.ToList();
            p.Add(Hash.Algorithm);
            IList<string> results = RunQuery(template, p.ToArray());
            Assert.Equal(countResults, results.Count);
        }

        [Fact] // TODO: Make theory
        public void SearchFile()
        {
            IList<string> results = RunQuery(SearchFileQueryTemplate, FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void SearshFileNotEq()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} != '{2}' do find;", FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, EmptyFile, 0), results[0]);
        }

        [Fact]
        public void SearchFileTimed()
        {
            IList<string> results = RunQueryWithOpt(SearchFileQueryTemplate, TimeOpt, FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileSearchTimeTpl);
        }

        [Fact]
        public void SearchFileRecursively()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Equal(2, results.Count);
        }
        
        [Fact]
        public void SearchFileOffsetMoreThenFileSize()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 100 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Empty(results);
        }
        
        [Fact]
        public void SearchFileOffsetNegative()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == -10 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Empty(results);
        }
        
        [Fact]
        public void SearchFileOffsetOverflow()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 9223372036854775808 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, Hash.Algorithm, HashString);
            Assert.Empty(results);
        }

        [Fact]
        public void SearchFileSeveralHashes()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", FileFixture.BaseTestDir, Hash.Algorithm, TrailPartStringHash, MiddlePartStringHash);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length), results[0]);
        }
        
        [Fact]
        public void SearchFileSeveralHashesOrOperator()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} == '{2}' or f.{1} == '{3}' do find withsubs;", FileFixture.BaseTestDir, Hash.Algorithm, HashString, EmptyStringHash);
            Assert.Equal(4, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, EmptyFile, 0), results[0]);
            Assert.Equal(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length), results[1]);
        }
        
        [Fact]
        public void SearchFileSeveralHashesNoResults()
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", FileFixture.BaseTestDir, Hash.Algorithm, TrailPartStringHash, HashString);
            Assert.Empty(results);
        }

        [Fact] // TODO: Make theory
        public void ValidateFileSuccess()
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, NotEmptyFile, Hash.Algorithm, HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is valid", InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void ValidateFileFailure()
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, NotEmptyFile, Hash.Algorithm, TrailPartStringHash);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is invalid", InitialString.Length), results[0]);
        }
    }
}