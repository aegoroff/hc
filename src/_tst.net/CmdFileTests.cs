/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2015 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace _tst.net
{
    [Trait("Mode", "cmd")]
    public abstract class CmdFileTests<T> : FileTests<T>
        where T : Architecture, new()
    {
        private const string EmptyFileName = "empty";
        private const string NotEmptyFileName = "notempty";
        
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec$";
        private static string notEmptyFile = FileFixture.BaseTestDir + FileFixture.Slash + NotEmptyFileName;
        private static string emptyFile = FileFixture.BaseTestDir + FileFixture.Slash + EmptyFileName;
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

        protected override IList<string> RunFileHashCalculation(Hash h, string file)
        {
            return this.Runner.Run(h.Algorithm, FileOpt, file);
        }
        
        protected override IList<string> RunDirWithSpecialOption(Hash h, string option)
        {
            return this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, option);
        }

        [Theory, MemberData("HashesForCalcFile")]
        public void CalcFileWithLimitThatBiggerThenFileSize(Hash h, string limitOptions)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, limitOptions);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        public static IEnumerable<object[]> HashesForCalcFile
        {
            get { return CreateProperty(new object[] { LimitOpt + " 10" }); }
        }

        [Theory, MemberData("Hashes")]
        public void ValidateFileSuccess(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, HashOpt, h.HashString);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, "File is valid", h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void ValidateFileFailure(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, HashOpt, h.TrailPartStringHash);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, "File is invalid", h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileTime(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, TimeOpt);
            Asserts.StringMatching(results[0], FileResultTimeTpl);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileLimit(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, LimitOpt, "2");
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.StartPartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileOffset(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, OffsetOpt, "1");
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.TrailPartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcFileLimitAndOffset(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, LimitOpt, "1", OffsetOpt, "1");
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.MiddlePartStringHash, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("HashesForFileNumbericOptionsNegativeTest")]
        public void CalcFileNumbericOptionsNegativeTest(Hash h, string limit, string expectation, string option, string optionName)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, option, limit);
            Assert.Equal("Invalid " + optionName + " option must be positive but was " + expectation, results[4]);
            Assert.Equal(5, results.Count);
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

        [Theory, MemberData("HashesForFileLimitAndOffsetIncorrectNumbers")]
        public void CalcFileLimitAndOffsetIncorrectNumbers(Hash h, string limit, string offset)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, LimitOpt, limit, OffsetOpt, offset);
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

        [Theory, MemberData("Hashes")]
        public void CalcFileOffsetGreaterThenFileSize(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, notEmptyFile, OffsetOpt, "4");
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
                IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, file, OffsetOpt, "1024");
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
                IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, file, OffsetOpt, "1024", LimitOpt, "1048500");
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
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, unexist);
            var success = string.Format("{0} \\| .+ bytes \\| .+", unexist);
            Asserts.StringNotMatching(results[0], success);
            Asserts.StringMatching(results[0], string.Format("{0} \\| .+?", unexist));
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcEmptyFile(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, FileOpt, emptyFile);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcDir(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[1]);
            Assert.Equal(2, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void CalcDirOutputToFile(Hash h)
        {
            const string save = "result";
            var dir = Path.GetDirectoryName(this.Runner.TestExePath);
            var result = Path.Combine(dir, save);

            try
            {
                IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, "-o", save);
                
                Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
                Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[1]);
                Assert.Equal(2, results.Count);

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

        [Theory, MemberData("HashesWithoutCrc32")]
        public void CalcDirSfv(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, "--sfv");
            Assert.Equal("", results[0]);
            Assert.Equal(string.Format(" --sfv option doesn't support {0} algorithm. Only crc32 supported", h.Algorithm), results[1]);
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
                IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, RecurseOpt);
                Assert.Equal(6, results.Count);
            }
            finally
            {
                Directory.Delete(FileFixture.SubDir + sub2Suffix, true);
            }
        }

        [Theory, MemberData("Hashes")]
        public void CalcDirIncludeFilter(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName);
            Assert.Equal(string.Format(FileResultTpl, emptyFile, h.EmptyStringHash, 0), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcDirExcludeFilter(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName);
            Assert.Equal(string.Format(FileResultTpl, notEmptyFile, h.HashString, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("HashesForCalcDirTheory")]
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

        [Theory, MemberData("Hashes")]
        public void SearchFile(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString);
            Assert.Equal(string.Format(FileSearchTpl, notEmptyFile, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }
        
        [Theory, MemberData("Hashes")]
        public void SearchFileTimed(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString, TimeOpt);
            Asserts.StringMatching(results[0], FileSearchTimeTpl);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void SearchFileRecursively(Hash h)
        {
            IList<string> results = this.Runner.Run(h.Algorithm, DirOpt, FileFixture.BaseTestDir, SearchOpt, h.HashString, RecurseOpt);
            Assert.Equal(2, results.Count);
        }
    }
}