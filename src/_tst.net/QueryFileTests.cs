/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2015 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;

namespace _tst.net
{
    [Trait("Mode", "query")]
    public abstract class QueryFileTests<T> : FileTests<T>, IDisposable
        where T : Architecture, new()
    {
        private const string EmptyFileName = "e_mpty";
        private const string NotEmptyFileName = "n_otempty";
        private static string notEmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + NotEmptyFileName;
        private static string emptyFile = FileFixture.BaseTestDir + FileFixture.Slash + EmptyFileName;
        private const string QueryOpt = "-C";
        private const string FileOpt = "-F";
        private const string ParamOpt = "-P";
        private const string TimeOpt = "-t";
        private const string NoProbeOpt = "--noprobe";
        private const string HashStringQueryTpl = "for string '{0}' do {1};";
        
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes$";

        private static string queryFile = FileFixture.BaseTestDir + FileFixture.Slash + "hl.hlq";
        private const string ValidationQueryTemplate = "for file f from '{0}' let f.{1} = '{2}' do validate;";
        private const string SearchFileQueryTemplate = "for file f from dir '{0}' where f.{1} == '{2}' do find;";
        private const string CalculateFileQueryTemplate = "for file f from '{0}' do {1};";

        protected override string EmptyFileNameProp
        {
            get { return EmptyFileName; }
        }

        protected override string EmptyFileProp
        {
            get { return emptyFile; }
        }

        protected override string NotEmptyFileNameProp
        {
            get { return NotEmptyFileName; }
        }

        protected override string NotEmptyFileProp
        {
            get { return notEmptyFile; }
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
            File.WriteAllText(queryFile, string.Format(template, parameters));
            return this.Runner.Run(FileOpt, queryFile);
        }
        
        IList<string> RunQueryWithOpt(string template, string additionalOptions, params object[] parameters)
        {
            return this.Runner.Run(QueryOpt, string.Format(template, parameters), additionalOptions);
        }

        protected override IList<string> RunFileHashCalculation(Hash h, string file)
        {
            return RunQuery(CalculateFileQueryTemplate, file, h.Algorithm);
        }

        protected override IList<string> RunDirWithSpecialOption(Hash h, string option)
        {
            return this.Runner.Run(QueryOpt, string.Format("for file f from dir '{0}' do {1};", FileFixture.BaseTestDir, h.Algorithm), option);
        }

        public void Dispose()
        {
            if (File.Exists(queryFile))
            {
                File.Delete(queryFile);
            }
        }

        [Theory, MemberData("Hashes")]
        public void FileQuery(Hash h)
        {
            IList<string> results = RunFileQuery(HashStringQueryTpl, h.InitialString, h.Algorithm);
            Assert.Equal(h.HashString, results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void FileWithSeveralQueries(Hash h)
        {
            const int count = 2;
            IList<string> results = RunFileQuery(SeveralQueries(count), h.InitialString, h.Algorithm);
            Assert.Equal(h.HashString, results[0]);
            Assert.Equal(h.HashString, results[1]);
            Assert.Equal(count, results.Count);
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

        [Theory, MemberData("Hashes")]
        public void CalcFileTime(Hash h)
        {
            IList<string> results = RunQueryWithOpt(CalculateFileQueryTemplate, TimeOpt, notEmptyFile, h.Algorithm);
            Asserts.StringMatching(results[0], FileResultTimeTpl);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("HashesForValidateParameterFile")]
        [Trait("Category", "hanging")]
        public void ValidateParameterFile(Hash h, int offset, string result)
        {
            IList<string> results = RunValidatingQuery(notEmptyFile, "for file f from parameter where f.offset == {0} and f.{1} == '{2}' do validate;", offset, h.Algorithm, h.TrailPartStringHash);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, result, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        public static IEnumerable<object[]> HashesForValidateParameterFile
        {
            get { return CreateProperty(new object[] {new object[] {1, "File is valid"}, new object[] {0, "File is invalid"}}); }
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileLimit(Hash h)
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1} do {2};", notEmptyFile, 2, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.StartPartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileOffset(Hash h)
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", notEmptyFile, 1, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.TrailPartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileLimitAndOffset(Hash h)
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {1} do {2};", notEmptyFile, 1, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.MiddlePartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileOffsetGreaterThenFileSIze(Hash h)
        {
            IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", notEmptyFile, 4, h.Algorithm);
            Assert.Equal(string.Format(FileErrorTpl, notEmptyFile, "Offset is greater then file size"), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcBigFileWithOffset(Hash h)
        {
            string file = notEmptyFile + "_big";
            CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.offset = {1} do {2};", file, 1024, h.Algorithm);
                Assert.Contains(" Mb (2", results[0]);
                Assert.Equal(1, results.Count);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Theory, MemberData("Hashes")]
        public void CalcBigFileWithLimitAndOffset(Hash h)
        {
            string file = notEmptyFile + "_big";
            CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunQuery("for file f from '{0}' let f.limit = {1}, f.offset = {2} do {3};", file, 1048500, 1024, h.Algorithm);
                Assert.Contains(" Mb (2", results[0]);
                Assert.Equal(1, results.Count);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Theory, MemberData("Hashes")]
        public void CalcUnexistFile(Hash h)
        {
            const string unexist = "u";
            IList<string> results = RunQuery(CalculateFileQueryTemplate, unexist, h.Algorithm);
            var success = string.Format("{0} \\| .+ bytes \\| .+", unexist);
            Asserts.StringNotMatching(results[0], success);
            Asserts.StringMatching(results[0], string.Format("{0} \\| .+?", unexist));
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileLowerOption(Hash h)
        {
            IList<string> results = this.RunQueryWithOpt(CalculateFileQueryTemplate, "-l", NotEmptyFileProp, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFileProp, h.HashString.ToLowerInvariant(), h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcEmptyFile(Hash h)
        {
            IList<string> results = RunQuery(CalculateFileQueryTemplate, emptyFile, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcDir(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' do {1};", FileFixture.BaseTestDir, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[1]);
            Assert.Equal(2, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void CalcDirLowerOption(Hash h)
        {
            IList<string> results = RunQueryWithOpt("for file f from dir '{0}' do {1};", "-l", FileFixture.BaseTestDir, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash.ToLowerInvariant(), 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString.ToLowerInvariant(), h.InitialString.Length), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcDirRecursivelyManySubs(Hash h)
        {
            const string sub2Suffix = "2";
            Directory.CreateDirectory(FileFixture.SubDir + sub2Suffix);

            CreateEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + EmptyFileName);
            CreateNotEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + NotEmptyFileName, h.InitialString);

            try
            {
                IList<string> results = RunQuery("for file f from dir '{0}' do {1} withsubs;", FileFixture.BaseTestDir, h.Algorithm);
                Assert.Equal(6, results.Count);
            }
            finally
            {
                Directory.Delete(FileFixture.SubDir + sub2Suffix, true);
            }
        }

        [Theory, MemberData("HashesForCalcDirIncludeFilter")]
        public void CalcDirIncludeFilter(Hash h, string template)
        {
            IList<string> results = RunQuery(template, FileFixture.BaseTestDir, EmptyFileName, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(1, results.Count);
        }


        public static IEnumerable<object[]> HashesForCalcDirIncludeFilter
        {
            get
            {
                return CreateProperty(new object[]
                {
                    "for file f from dir '{0}' where f.name ~ '{1}' do {2};", 
                    "let x = '{0}';let y = '{1}';for file f from dir x where f.name ~ y do {2};"
                });
            }
        }

        [Theory, MemberData("Hashes")]
        public void CalcDirExcludeFilter(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.name !~ '{1}' do {2};", FileFixture.BaseTestDir, EmptyFileName, h.Algorithm);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("HashesForCalcDirTheory")]
        public void CalcDirTheory(Hash h, int countResults, string template, object[] parameters)
        {
            List<object> p = parameters.ToList();
            p.Add(h.Algorithm);
            IList<string> results = RunQuery(template, p.ToArray());
            Assert.Equal(countResults, results.Count);
        }

        public static IEnumerable<object[]> HashesForCalcDirTheory
        {
            get
            {
                return CreateProperty(new object[]
                {
                    new object[] { 0, "for file f from dir '{0}' where f.name ~ '{1}' and f.name !~ '{1}' do {2};", new object[] { FileFixture.BaseTestDir, EmptyFileName } }, 
                    new object[] { 0, "for file f from dir '{0}' where f.name !~ '{1}' and f.name !~ '{2}' do {3};", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName } },
                    new object[] { 2, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3};", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName } },
                    new object[] { 2, "for file f from dir '{0}' where f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName } },
                    new object[] { 2, "for file f from dir '{0}' where f.name !~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName } },
                    new object[] { 4, "for file f from dir '{0}' do {1} withsubs;", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 2, "for file f from dir '{0}' where f.size == 0 do {1} withsubs;", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 1, "for file f from dir '{0}' where f.size == 0 do {1};", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 2, "for file f from dir '{0}' where f.size != 0 do {1} withsubs;", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 1, "for file f from dir '{0}' where f.size != 0 do {1};", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 4, "for file f from dir '{0}' where f.size == 0 or f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName } },
                    new object[] { 2, "for file f from dir '{0}' where f.size == 0 or not f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName } },
                    new object[] { 0, "for file f from dir '{0}' where f.size == 0 and f.name ~ '{1}' do {2} withsubs;", new object[] { FileFixture.BaseTestDir, NotEmptyFileName } },
                    new object[] { 4, "for file f from dir '{0}' where f.name ~ '{1}' or f.name ~ '{2}' do {3} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName } },
                    new object[] { 2, "for file f from dir '{0}' where f.name ~ '{1}' or (f.name ~ '{2}' and f.size == 0) do {3} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName } },
                    new object[] { 3, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) or f.path ~ '{3}' do {4} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*" } },
                    new object[] { 1, "for file f from dir '{0}' where (f.name ~ '{1}' and (f.name ~ '{2}' or f.size == 0)) and f.path ~ '{3}' do {4} withsubs;", new object[] { FileFixture.BaseTestDir, EmptyFileName, NotEmptyFileName, @".*sub.*" } },
                    new object[] { 0, "for file f from dir '{0}' where f.name ~ '' do {1} withsubs;", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 4, "for file f from dir '{0}' where f.name !~ '' do {1} withsubs;", new object[] { FileFixture.BaseTestDir } },
                    new object[] { 4, "for file f from dir '{0}' where f.name ~ 'mpty' do {1} withsubs;", new object[] { FileFixture.BaseTestDir } }
                });
            }
        }

        [Theory, MemberData("HashesForFileSearch")]
        public void SearchFile(Hash h, string template, int length, string file)
        {
            IList<string> results = RunQuery(template, FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Assert.Equal(string.Format(FileSearchTpl, file, length), results[0]);
            Assert.Equal(1, results.Count);
        }

        public static IEnumerable<object[]> HashesForFileSearch
        {
            get
            {
                return CreateProperty(new object[]
                {
                    new object[] { SearchFileQueryTemplate, 3, notEmptyFile }, 
                    new object[] { "for file f from dir '{0}' where f.{1} != '{2}' do find;", 0, emptyFile }
                });
            }
        }

        [Theory, MemberData("Hashes")]
        public void SearchFileTimed(Hash h)
        {
            IList<string> results = RunQueryWithOpt(SearchFileQueryTemplate, TimeOpt, FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Asserts.StringMatching(results[0], FileSearchTimeTpl);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void SearchFileRecursively(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Assert.Equal(2, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileOffsetMoreThenFileSize(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 100 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Assert.Empty(results);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileOffsetNegative(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == -10 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Assert.Empty(results);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileOffsetOverflow(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 9223372036854775808 and f.{1} == '{2}' do find withsubs;", FileFixture.BaseTestDir, h.Algorithm, h.HashString);
            Assert.Empty(results);
        }

        [Theory, MemberData("Hashes")]
        public void SearchFileSeveralHashes(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", FileFixture.BaseTestDir, h.Algorithm, h.TrailPartStringHash, h.MiddlePartStringHash);
            Assert.Equal(string.Format(FileSearchTpl, notEmptyFile, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileSeveralHashesOrOperator(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.{1} == '{2}' or f.{1} == '{3}' do find withsubs;", FileFixture.BaseTestDir, h.Algorithm, h.HashString, h.EmptyStringHash);
            Assert.Equal(string.Format(FileSearchTpl, emptyFile, 0), results[0]);
            Assert.Equal(string.Format(FileSearchTpl, notEmptyFile, h.InitialString.Length), results[1]);
            Assert.Equal(4, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileSeveralHashesNoResults(Hash h)
        {
            IList<string> results = RunQuery("for file f from dir '{0}' where f.offset == 1 and f.{1} == '{2}' and f.offset == 1 and f.limit == 1 and f.{1} == '{3}' do find;", FileFixture.BaseTestDir, h.Algorithm, h.TrailPartStringHash, h.HashString);
            Assert.Empty(results);
        }

        [Theory, MemberData("Hashes")]
        public void ValidateFileSuccess(Hash h)
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, notEmptyFile, h.Algorithm, h.HashString);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, "File is valid", h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void ValidateFileFailure(Hash h)
        {
            IList<string> results = RunQuery(ValidationQueryTemplate, notEmptyFile, h.Algorithm, h.TrailPartStringHash);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, "File is invalid", h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }
    }
}