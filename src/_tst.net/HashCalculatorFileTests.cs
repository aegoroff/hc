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
    public abstract class HashCalculatorFileTests<T, THash> : FileTests<T, THash>
        where T : Architecture, new()
        where THash : Hash, new()
    {
        private const string EmptyFileName = "empty";
        private const string NotEmptyFileName = "notempty";
        private const string FileResultTpl = @"{0} | {2} bytes | {1}";
        private const string FileResultSfvTpl = @"{0}    {1}";
        private const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";
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

        protected override string Executable
        {
            get { return base.Executable + " " + this.Hash.Algorithm; }
        }

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

        [Theory]
        [InlineData( "" )]
        [InlineData( LimitOpt + " 10" )]
        public void CalcFile( string limitOptions )
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, limitOptions);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void ValidateFileSuccess()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, HashOpt, HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is valid", InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void ValidateFileFailure()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, HashOpt, TrailPartStringHash);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "File is invalid", InitialString.Length), results[0]);
        }

        [Fact]
        public void CalcFileTime()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, TimeOpt);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileResultTimeTpl);
        }

        [Fact] // TODO: Make theory
        public void CalcFileLimit()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, "2");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, StartPartStringHash, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileOffset()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, OffsetOpt, "1");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, TrailPartStringHash, InitialString.Length), results[0]);
        }

        [Fact] // TODO: Make theory
        public void CalcFileLimitAndOffset()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, "1", OffsetOpt, "1");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, MiddlePartStringHash, InitialString.Length), results[0]);
        }

        [Theory]
        [InlineData("9223372036854775808", "-9223372036854775808", LimitOpt, "limit")]
        [InlineData("-10", "-10", LimitOpt, "limit")]
        [InlineData("9223372036854775808", "-9223372036854775808", OffsetOpt, "offset")]
        [InlineData("-10", "-10", OffsetOpt, "offset")]
        public void CalcFileNumbericOptionsNegativeTest(string limit, string expectation, string option, string optionName)
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, option, limit);
            Assert.Equal(5, results.Count);
            Assert.Equal("Invalid " + optionName + " option must be positive but was " + expectation, results[4]);
        }

        [Theory]
        [InlineData("a", "1")]
        [InlineData("a", "0")]
        [InlineData("a", "a")]
        public void CalcFileLimitAndOffsetIncorrectNumbers(string limit, string offset)
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, limit, OffsetOpt, offset);
            Asserts.StringMatching(results[0], InvalidNumberTpl);
        }

        [Fact]
        public void CalcFileOffsetGreaterThenFileSIze()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, OffsetOpt, "4");
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, "Offset is greater then file size",
                                                 InitialString.Length), results[0]);
        }

        [Fact]
        public void CalcBigFile()
        {
            const string file = NotEmptyFile + "_big";
            CreateNotEmptyFile(file, 2 * 1024 * 1024);
            try
            {
                IList<string> results = this.Runner.Run(FileOpt, file);
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
                IList<string> results = this.Runner.Run(FileOpt, file, OffsetOpt, "1024");
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
                IList<string> results = this.Runner.Run(FileOpt, file, OffsetOpt, "1024", LimitOpt, "1048500");
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
            IList<string> results = this.Runner.Run(FileOpt, unexist);
            Assert.Equal(1, results.Count);
            string en = string.Format("{0} | The system cannot find the file specified.  ", unexist);
            string ru = string.Format("{0} | Не удается найти указанный файл.  ", unexist);
            Assert.Contains(results[0], new[] {en, ru});
        }

        [Fact]
        public void CalcEmptyFile()
        {
            IList<string> results = this.Runner.Run(FileOpt, EmptyFile);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
        }

        [Fact]
        public void CalcDir()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir);
            Assert.Equal(2, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[1]);
        }
        
        [Fact]
        public void CalcDirOutputToFile()
        {
            const string save = "result";
            var dir = Path.GetDirectoryName(this.Runner.TestExePath);
            var result = Path.Combine(dir, save);

            try
            {
                IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, "-o", save);
                Assert.Equal(2, results.Count);
                Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
                Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[1]);

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
        
        [Fact]
        public void CalcDirSfv()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, "--sfv");
            Assert.Equal(2, results.Count);
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(EmptyFile), EmptyStringHash), results[0]);
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(NotEmptyFile), HashString), results[1]);
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
                IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, RecurseOpt);
                Assert.Equal(6, results.Count);
            }
            finally
            {
                Directory.Delete(FileFixture.SubDir + sub2Suffix, true);
            }
        }

        [Fact]
        public void CalcDirIncludeFilter()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0), results[0]);
        }

        [Fact]
        public void CalcDirExcludeFilter()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length), results[0]);
        }

        [Theory]
        [InlineData(0, new[] { DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName, ExcludeOpt, EmptyFileName })]
        [InlineData(0, new[] { DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName + ";" + NotEmptyFileName })]
        [InlineData( 2, new[] {DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName + ";" + NotEmptyFileName })]
        [InlineData( 2, new[] {DirOpt, FileFixture.BaseTestDir, IncludeOpt, EmptyFileName, RecurseOpt })]
        [InlineData( 2, new[] {DirOpt, FileFixture.BaseTestDir, ExcludeOpt, EmptyFileName, RecurseOpt })]
        [InlineData( 4, new[] {DirOpt, FileFixture.BaseTestDir, RecurseOpt })]
        public void CalcDirTheory( int countResults, params string[] commandLine )
        {
            IList<string> results = this.Runner.Run(commandLine);
            Assert.Equal(countResults, results.Count);
        }

        [Fact]
        public void SearchFile()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, SearchOpt, HashString);
            Assert.Equal(1, results.Count);
            Assert.Equal(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length), results[0]);
        }
        
        [Fact]
        public void SearchFileTimed()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, SearchOpt, HashString, TimeOpt);
            Assert.Equal(1, results.Count);
            Asserts.StringMatching(results[0], FileSearchTimeTpl);
        }

        [Fact]
        public void SearchFileRecursively()
        {
            IList<string> results = this.Runner.Run(DirOpt, FileFixture.BaseTestDir, SearchOpt, HashString, RecurseOpt);
            Assert.Equal(2, results.Count);
        }
    }
}