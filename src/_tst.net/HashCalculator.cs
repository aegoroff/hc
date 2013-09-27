/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2013 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;

namespace _tst.net
{
    public abstract class HashCalculator<THash> : HashBase<THash> where THash : Hash, new()
    {
        private const string EmptyStr = "\"\"";
        private const string RestoredStringTemplate = "Initial string is: {0}";
        private const string NothingFound = "Nothing found";
        private const string BaseTestDir = @"C:\_tst.net";
        private const string SubDir = BaseTestDir + Slash + "sub";
        private const string EmptyFileName = "empty";
        private const string NotEmptyFileName = "notempty";
        private const string Slash = @"\";
        private const string FileResultTpl = @"{0} | {2} bytes | {1}";
        private const string FileResultSfvTpl = @"{0}    {1}";
        private const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";
        private const string FileSearchTpl = @"{0} | {1} bytes";
        private const string FileSearchTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec$";
        private const string NotEmptyFile = BaseTestDir + Slash + NotEmptyFileName;
        private const string EmptyFile = BaseTestDir + Slash + EmptyFileName;
        private const string LowCaseOpt = "-l";
        private const string IncludeOpt = "-i";
        private const string ExcludeOpt = "-e";
        private const string RecurseOpt = "-r";
        private const string DictOpt = "-a";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string CrackOpt = "-c";
        private const string PerfOpt = "-p";
        private const string HashOpt = "-m";
        private const string StringOpt = "-s";
        private const string DirOpt = "-d";
        private const string FileOpt = "-f";
        private const string SearchOpt = "-H";
        private const string LimitOpt = "-z";
        private const string OffsetOpt = "-q";
        private const string TimeOpt = "-t";
        private const string NoProbeOpt = "--noprobe";
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

        [Test]
        public void CalcString()
        {
            IList<string> results = this.Runner.Run(StringOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString));
        }

        [Test]
        public void CalcStringLowCaseOutput()
        {
            IList<string> results = this.Runner.Run(StringOpt, InitialString, LowCaseOpt);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
        }

        [Test]
        public void CalcEmptyString()
        {
            IList<string> results = this.Runner.Run(StringOpt, EmptyStr);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(EmptyStringHash));
        }

        [Test]
        public void CrackString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }
        
        [Test]
        public void CrackEmptyString()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashEmptyString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "Empty string")));
        }

        [Test]
        public void CrackStringUsingLowCaseHash()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString.ToLowerInvariant(), MaxOpt, "3", DictOpt, InitialString);
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
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, DictOpt, dict, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
        }

        [TestCase("a-zA-Z")]
        [TestCase("a-z")]
        [TestCase("A-Z")]
        [TestCase("abc")]
        public void CrackStringFailureUsingNonDefaultDictionary(string dict)
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, DictOpt, dict, MaxOpt, "3");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }


        [Test]
        public void CrackStringTooShortLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MaxOpt,
                                                (InitialString.Length - 1).ToString(), DictOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [Test]
        public void CrackStringTooLongMinLength()
        {
            IList<string> results = this.Runner.Run(CrackOpt, NoProbeOpt, HashOpt, HashString, MinOpt,
                                                ( InitialString.Length + 1 ).ToString(), MaxOpt,
                                                (InitialString.Length + 1).ToString(), DictOpt, InitialString);
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(NothingFound));
        }

        [TestCase( "" )]
        [TestCase( LimitOpt + " 10" )]
        public void CalcFile( string limitOptions )
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, limitOptions);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [Test]
        public void ValidateFileSuccess()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, HashOpt, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, "File is valid", InitialString.Length)));
        }

        [Test]
        public void ValidateFileFailure()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, HashOpt, TrailPartStringHash);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, "File is invalid", InitialString.Length)));
        }

        [Test]
        public void CalcFileTime()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, TimeOpt);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.StringMatching(FileResultTimeTpl));
        }

        [Test]
        public void CalcFileLimit()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, "2");
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, StartPartStringHash, InitialString.Length)));
        }


        [Test]
        public void CalcFileOffset()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, OffsetOpt, "1");
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, TrailPartStringHash, InitialString.Length)));
        }

        [TestCase("9223372036854775808", "-9223372036854775808", LimitOpt, "limit")]
        [TestCase("-10", "-10", LimitOpt, "limit")]
        [TestCase("9223372036854775808", "-9223372036854775808", OffsetOpt, "offset")]
        [TestCase("-10", "-10", OffsetOpt, "offset")]
        public void CalcFileNumbericOptionsNegativeTest(string limit, string expectation, string option, string optionName)
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, option, limit);
            Assert.That(results.Count, Is.EqualTo(5));
            Assert.That(results[4], Is.EqualTo("Invalid " + optionName + " option must be positive but was " + expectation));
        }

        [Test]
        public void CalcFileLimitAndOffset()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, "1", OffsetOpt, "1");
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, MiddlePartStringHash, InitialString.Length)));
        }
        
        [TestCase("a", "1")]
        [TestCase("a", "0")]
        [TestCase("a", "a")]
        public void CalcFileLimitAndOffsetIncorrectNumbers(string limit, string offset)
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, LimitOpt, limit, OffsetOpt, offset);
            Assert.That(results[0], Is.StringMatching(InvalidNumberTpl));
        }

        [Test]
        public void CalcFileOffsetGreaterThenFileSIze()
        {
            IList<string> results = this.Runner.Run(FileOpt, NotEmptyFile, OffsetOpt, "4");
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
                IList<string> results = this.Runner.Run(FileOpt, file);
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
                IList<string> results = this.Runner.Run(FileOpt, file, OffsetOpt, "1024");
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
                IList<string> results = this.Runner.Run(FileOpt, file, OffsetOpt, "1024", LimitOpt, "1048500");
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
            IList<string> results = this.Runner.Run(FileOpt, unexist);
            Assert.That(results.Count, Is.EqualTo(1));
            string en = string.Format("{0} | The system cannot find the file specified.  ", unexist);
            string ru = string.Format("{0} | Не удается найти указанный файл.  ", unexist);
            Assert.That(results[0], Is.InRange(en, ru));
        }

        [Test]
        public void CalcEmptyFile()
        {
            IList<string> results = this.Runner.Run(FileOpt, EmptyFile);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
        }

        [Test]
        public void CalcDir()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir);
            Assert.That(results.Count, Is.EqualTo(2));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
            Assert.That(results[1],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }
        
        [Test]
        public void CalcDirOutputToFile()
        {
            const string save = "result";
            var dir = Path.GetDirectoryName(this.Runner.TestExePath);
            var result = Path.Combine(dir, save);

            try
            {
                IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, "-o", save);
                Assert.That(results.Count, Is.EqualTo(2));
                Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
                Assert.That(results[1],
                    Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));

                Assert.That(File.Exists(result));
                var content = File.ReadAllText(result);
                var fromConsole = string.Join(Environment.NewLine, results) + Environment.NewLine;
                Assert.AreEqual(fromConsole, content);
            }
            finally
            {
                if (File.Exists(result))
                {
                    File.Delete(result);
                }
            }
        }
        
        [Test]
        public void CalcDirSfv()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, "--sfv");
            Assert.That(results.Count, Is.EqualTo(2));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultSfvTpl, Path.GetFileName(EmptyFile), EmptyStringHash)));
            Assert.That(results[1],
                        Is.EqualTo(string.Format(FileResultSfvTpl, Path.GetFileName(NotEmptyFile), HashString)));
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
                IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, RecurseOpt);
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
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, IncludeOpt, EmptyFileName);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
        }

        [Test]
        public void CalcDirExcludeFilter()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, InitialString.Length)));
        }

        [TestCase( 0, DirOpt, BaseTestDir, IncludeOpt, EmptyFileName, ExcludeOpt, EmptyFileName )]
        [TestCase( 0, DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName + ";" + NotEmptyFileName )]
        [TestCase( 2, DirOpt, BaseTestDir, IncludeOpt, EmptyFileName + ";" + NotEmptyFileName )]
        [TestCase( 2, DirOpt, BaseTestDir, IncludeOpt, EmptyFileName, RecurseOpt )]
        [TestCase( 2, DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName, RecurseOpt )]
        [TestCase( 4, DirOpt, BaseTestDir, RecurseOpt )]
        public void CalcDir( int countResults, params string[] commandLine )
        {
            IList<string> results = this.Runner.Run(commandLine);
            Assert.That(results.Count, Is.EqualTo(countResults));
        }

        [Test]
        public void SearchFile()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, SearchOpt, HashString);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0],
                        Is.EqualTo(string.Format(FileSearchTpl, NotEmptyFile, InitialString.Length)));
        }
        
        [Test]
        public void SearchFileTimed()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, SearchOpt, HashString, TimeOpt);
            Assert.That(results.Count, Is.EqualTo(1));
            Assert.That(results[0], Is.StringMatching(FileSearchTimeTpl));
        }

        [Test]
        public void SearchFileRecursively()
        {
            IList<string> results = this.Runner.Run(DirOpt, BaseTestDir, SearchOpt, HashString, RecurseOpt);
            Assert.That(results.Count, Is.EqualTo(2));
        }

        [Test]
        public void TestPerformance()
        {
            IList<string> results = this.Runner.Run(PerfOpt, DictOpt, "12345", MaxOpt, "5", MinOpt, "5");
            Assert.That(results.Count, Is.EqualTo(3));
            Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, "12345")));
        }
    }
}