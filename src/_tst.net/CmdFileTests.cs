/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2013 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using Xunit.Extensions;

namespace _tst.net
{
    [Trait("Mode", "cmd")]
    public abstract class CmdFileTests<T> : FileTests<T>
        where T : Architecture, new()
    {
        private const string EmptyFileName = "empty";
        private const string NotEmptyFileName = "notempty";
        private const string FileResultSfvTpl = @"{0}    {1}";
        
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec$";
        private const string NotEmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + NotEmptyFileName;
        private const string EmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + EmptyFileName;
        private const string IncludeOpt = "-i";
        private const string ExcludeOpt = "-e";
        private const string RecurseOpt = "-r";
        private const string HashOpt = "-m";
        private const string DirOpt = "-d";
        private const string FileOpt = "-f";
        private const string SearchOpt = "-H";
        private const string LimitOpt = "-z";
        private const string OffsetOpt = "-q";
        private const string TimeOpt = "-t";
        private const string InvalidNumberTpl = @"Invalid parameter --\w{3,6} (\w+)\. Must be number";

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

        protected override IList<string> RunFileHashCalculation(Hash h, string file)
        {
            return this.Runner.Run(h.Algorithm, FileOpt, file);
        }

        [Theory, PropertyData("HashesForCalcFile")]
        public void CalcFileWithLimitThatBiggerThenFileSize(Hash h, string limitOptions)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, limitOptions);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.HashString, h.InitialString.Length), results[0]);
        }

        public static IEnumerable<object[]> HashesForCalcFile
        {
            get { return CreateProperty(new object[] { LimitOpt + " 10" }); }
        }

        [Theory, PropertyData("Hashes")]
        public void ValidateFileSuccess(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, HashOpt, h.HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is valid", h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void ValidateFileFailure(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, HashOpt, h.TrailPartStringHash);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is invalid", h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcFileTime(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, TimeOpt);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileResultTimeTpl);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcFileLimit(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, LimitOpt, "2");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.StartPartStringHash, h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcFileOffset(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, OffsetOpt, "1");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.TrailPartStringHash, h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcFileLimitAndOffset(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, LimitOpt, "1", OffsetOpt, "1");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.MiddlePartStringHash, h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("HashesForFileNumbericOptionsNegativeTest")]
        public void CalcFileNumbericOptionsNegativeTest(Hash h, string limit, string expectation, string option, string optionName)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, option, limit);
            Assert.Equal(5, results.Count);
            Assert.Equal("Invalid " + optionName + " option must be positive but was " + expectation, results[4]);
        }

        public static IEnumerable<object[]> HashesForFileNumbericOptionsNegativeTest
        {
            get
            {
                return CreateProperty(new object[]
                {
                    new object[] { "9223372036854775808", "-9223372036854775808", LimitOpt, "limit" }, 
                    new object[] { "-10", "-10", LimitOpt, "limit" },
                    new object[] { "9223372036854775808", "-9223372036854775808", OffsetOpt, "offset" },
                    new object[] { "-10", "-10", OffsetOpt, "offset" }
                });
            }
        }

        [Theory, PropertyData("HashesForFileLimitAndOffsetIncorrectNumbers")]
        public void CalcFileLimitAndOffsetIncorrectNumbers(Hash h, string limit, string offset)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, LimitOpt, limit, OffsetOpt, offset);
            Asserts.StringMatching(results[0], InvalidNumberTpl);
        }

        public static IEnumerable<object[]> HashesForFileLimitAndOffsetIncorrectNumbers
        {
            get
            {
                return CreateProperty(new object[]
                {
                    new object[] { "a", "1" }, 
                    new object[] { "a", "0" },
                    new object[] { "a", "a" }
                });
            }
        }

        [Theory, PropertyData("Hashes")]
        public void CalcFileOffsetGreaterThenFileSize(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, NotEmptyFile, OffsetOpt, "4");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "Offset is greater then file size",
                                                 h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcBigFileWithOffset(Hash h)
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
            try
            {
                IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, file, OffsetOpt, "1024");
                Assert.Equal(1, results.Count);
                Assert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Theory, PropertyData("Hashes")]
        public void CalcBigFileWithLimitAndOffset(Hash h)
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
            try
            {
                IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, file, OffsetOpt, "1024", LimitOpt, "1048500");
                Assert.Equal(1, results.Count);
                Assert.Contains(" Mb (2", results[0]);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Theory, PropertyData("Hashes")]
        public void CalcUnexistFile(Hash h)
        {
            const string unexist = "u";
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, unexist);
            Assert.Equal(1, results.Count);
            var success = string.Format("{0} \\| .+ bytes \\| .+", unexist);
            Asserts.StringNotMatching(results[0], success);
            Asserts.StringMatching(results[0], string.Format("{0} \\| .+?", unexist));
        }

        [Theory, PropertyData("Hashes")]
        public void CalcEmptyFile(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, EmptyFile);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, h.EmptyStringHash, 0), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcDir(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir);
            Assert.Equal(2, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.HashString, h.InitialString.Length), results[1]);
        }
        
        [Theory, PropertyData("Hashes")]
        public void CalcDirOutputToFile(Hash h)
        {
            const string save = "result";
            var dir = Path.GetDirectoryName(this.Runner.TestExePath);
            var result = Path.Combine(dir, save);

            try
            {
                IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, "-o", save);
                Assert.Equal(2, results.Count);
                Assert.Equal(string.Format(FileResultTpl, EmptyFile, h.EmptyStringHash, 0), results[0]);
                Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.HashString, h.InitialString.Length), results[1]);

                Assert.True(File.Exists(result));
                var content = File.ReadAllText(result);
                var fromConsole = string.Join(Environment.NewLine, results) + Environment.NewLine;
                Assert.Equal(fromConsole, content);
            }
            finally
            {
                if (File.Exists(result))
                {
                    File.Delete(result);
                }
            }
        }
        
        [Theory, PropertyData("Hashes")]
        public void CalcDirSfv(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, "--sfv");
            Assert.Equal(2, results.Count);
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(EmptyFile), h.EmptyStringHash), results[0]);
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(NotEmptyFile), h.HashString), results[1]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcDirRecursivelyManySubs(Hash h)
        {
            const string sub2Suffix = "2";
            Directory.CreateDirectory(FileFixture.SubDir + sub2Suffix);

            CreateEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + EmptyFileName);
            CreateNotEmptyFile(FileFixture.SubDir + sub2Suffix + FileFixture.Slash + NotEmptyFileName, h.InitialString);

            try
            {
                IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, RecurseOpt);
                Assert.Equal(6, results.Count);
            }
            finally
            {
                Directory.Delete(FileFixture.SubDir + sub2Suffix, true);
            }
        }

        [Theory, PropertyData("Hashes")]
        public void CalcDirIncludeFilter(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, h.EmptyStringHash, 0), results[0]);
        }

        [Theory, PropertyData("Hashes")]
        public void CalcDirExcludeFilter(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, h.HashString, h.InitialString.Length), results[0]);
        }

        [Theory, PropertyData("HashesForCalcDirTheory")]
        public void CalcDirTheory(Hash h, int countResults, params string[] commandLine)
        {
            var cmd = new List<string> { h.Algorithm };
            cmd.AddRange(commandLine);
            IList<string> results = this.Runner.Run(cmd.ToArray());
            Assert.Equal(countResults, results.Count);
        }

        public static IEnumerable<object[]> HashesForCalcDirTheory
        {
            get
            {
                return CreateProperty(new object[]
                {
                    new object[] {0, new[] { DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName, ExcludeOpt, EmptyFileName } }, 
                    new object[] { 0, new[] { DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName + ";" + NotEmptyFileName } },
                    new object[] { 2, new[] {DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName + ";" + NotEmptyFileName } },
                    new object[] { 2, new[] {DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName, RecurseOpt } },
                    new object[] { 2, new[] {DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName, RecurseOpt } },
                    new object[] { 4, new[] {DirOpt, FileFixture.BaseTestDir, RecurseOpt } }
                });
            }
        }

        [Theory, PropertyData("Hashes")]
        public void SearchFile(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, NotEmptyFile, h.InitialString.Length), results[0]);
        }
        
        [Theory, PropertyData("Hashes")]
        public void SearchFileTimed(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString, TimeOpt);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileSearchTimeTpl);
        }

        [Theory, PropertyData("Hashes")]
        public void SearchFileRecursively(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString, RecurseOpt);
            Assert.Equal(2, results.Count);
        }
    }
}