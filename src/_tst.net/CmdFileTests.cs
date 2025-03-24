/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2025 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using FluentAssertions;
using Xunit;

namespace _tst.net;

[Trait("Mode", "cmd")]
public abstract class CmdFileTests<T>(FileFixture fixture) : FileTests<T>(fixture)
        where T : Architecture, new()
{
    private readonly FileFixture fixture = fixture;

    private const string EmptyFileName = "empty";
    private const string NotEmptyFileName = "notempty";

    private const string DirCmd = "dir";
    private const string FileCmd = "file";
    private const string StringCmd = "string";

    private const string FileSearchTpl = @"{0} | {1} bytes";
    private const string FileSearchTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec$";
    private readonly string notEmptyFile = Path.Combine(fixture.BaseTestDir, NotEmptyFileName);
    private readonly string emptyFile = Path.Combine(fixture.BaseTestDir, EmptyFileName);
    private const string IncludeOpt = "-i";
    private const string ExcludeOpt = "-e";
    private const string RecurseOpt = "-r";
    private const string HashOpt = "-m";
    private const string SearchOpt = "-H";
    private const string LimitOpt = "-z";
    private const string OffsetOpt = "-q";
    private const string TimeOpt = "-t";
    private const string Base64Opt = "-b";
    private const string SourceOpt = "-s";
    private const string InvalidNumberTpl = @"Invalid parameter --\w{3,6} (\w+)\. Must be number";
    private const string EmptyStr = "\"\"";

    protected override string EmptyFileNameProp => EmptyFileName;

    protected override string EmptyFileProp => this.emptyFile;

    protected override string NotEmptyFileNameProp => NotEmptyFileName;

    protected override string NotEmptyFileProp => this.notEmptyFile;

    protected override IList<string> RunFileHashCalculation(Hash h, string file) =>
            this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, file);

    protected override IList<string> RunDirWithSpecialOption(Hash h, string option) =>
            this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, option);

    [Theory, MemberData(nameof(HashesForCalcFile))]
    public void CalcFile_LimitBiggerThenFileSize_AllFileHashExpected(Hash h, string limitOptions)
    {
        // Arrange
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, limitOptions);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.HashString, h.InitialString.Length));
    }

    public static IEnumerable<object[]> HashesForCalcFile => CreateProperty([LimitOpt + " 10"]);

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_ValidateFile_Success(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, HashOpt, h.HashString);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, "File is valid", h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_ValidateFile_Failure(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, HashOpt, h.TrailPartStringHash);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, "File is invalid", h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_WihTimeOption_CalculationTimeInTheResult(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, TimeOpt);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().MatchRegex(FileResultTimeTpl);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Limit_StartPartHashExpected(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, LimitOpt, "2");

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.StartPartStringHash, h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Offset_TrailPartHashExpected(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, OffsetOpt, "1");

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.TrailPartStringHash, h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_LimitAndOffset_MiddlePartHashExpected(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, LimitOpt, "1", OffsetOpt, "1");

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.MiddlePartStringHash, h.InitialString.Length));
    }

    [Theory, MemberData(nameof(HashesForFileNumbericOptionsNegativeTest))]
    public void CalcFile_InvalidNumbericOptions_Failure(Hash h, string value, string expectation, string option, string optionName)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, option, value);

        // Assert
        results.Should().HaveCount(3);
        results[2].Should().Be($"Invalid {optionName} option must be positive but was {expectation}");
    }

    public static IEnumerable<object[]> HashesForFileNumbericOptionsNegativeTest => CreateProperty([
            new object[] { "-10223372036854775808", "-9223372036854775808", LimitOpt, "limit" }, 
                new object[] { "-10", "-10", LimitOpt, "limit" },
                new object[] { "-10223372036854775808", "-9223372036854775808", OffsetOpt, "offset" },
                new object[] { "-10", "-10", OffsetOpt, "offset" }
    ]);

    [Theory, MemberData(nameof(HashesForFileNumbericOptionsExtremeTest))]
    public void CalcFile_ExtremeNumbericOptions_Success(Hash h, string value, string option)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, option, value);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().MatchRegex(FileResultTimeTpl);
    }

    public static IEnumerable<object[]> HashesForFileNumbericOptionsExtremeTest => CreateProperty([
            new object[] { "18446744073709551615", LimitOpt }, 
                new object[] { "18446744073709551615", OffsetOpt }
    ]);

    [Theory, MemberData(nameof(HashesForFileLimitAndOffsetIncorrectNumbers))]
    public void CalcFile_LimitAndOffsetIncorrectNumbers_Failure(Hash h, string limit, string offset)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, LimitOpt, limit, OffsetOpt, offset);

        // Assert
        results[0].Should().MatchRegex(InvalidNumberTpl);
    }

    public static IEnumerable<object[]> HashesForFileLimitAndOffsetIncorrectNumbers => CreateProperty([
            new object[] { "a", "1" }, 
                new object[] { "a", "0" },
                new object[] { "a", "a" }
    ]);

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_OffsetGreaterThenFileSize_Failure(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.notEmptyFile, OffsetOpt, "4");

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileErrorTpl, this.notEmptyFile, "Offset is greater then file size"));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_BigWithOffset_Success(Hash h)
    {
        // Arrange
        var file = this.notEmptyFile + "_big";
        this.CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
        try
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, file, OffsetOpt, "1024");

            // Assert
            results.Should().HaveCount(1);
            results[0].Should().Contain(" Mb (2");
        }
        finally
        {
            File.Delete(file);
        }
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_BigWithLimitAndOffset_Success(Hash h)
    {
        // Arrange
        var file = this.notEmptyFile + "_big";
        this.CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
        try
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, file, OffsetOpt, "1024", LimitOpt, "1048500");

            // Assert
            results.Should().HaveCount(1);
            results[0].Should().Contain(" Mb (2");
        }
        finally
        {
            File.Delete(file);
        }
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Unexist_Failure(Hash h)
    {
        // Arrange
        const string unexist = "u";

        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, unexist);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().NotMatchRegex($"{unexist} \\| .+ bytes \\| .+");
        results[0].Should().MatchRegex($"{unexist} \\| .+?");
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Empty_Success(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, FileCmd, SourceOpt, this.emptyFile);
            
        // Assert
        results[0].Should().Be(string.Format(FileResultTpl, this.emptyFile, h.EmptyStringHash, 0));
        results.Should().HaveCount(1);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_SingleNoAdditionalOptions_AllDirFilesCalculated(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir);

        // Assert
        results.Should().HaveCount(2);
        results[0].Should().Be(string.Format(FileResultTpl, this.emptyFile, h.EmptyStringHash, 0));
        results[1].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.HashString, h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_SingleNoAdditionalOptionsOutputAsBase64_AllDirFilesCalculated(Hash h)
    {
        // Arrange
        var stringResults = this.Runner.Run(h.Algorithm, StringCmd, SourceOpt, h.InitialString, Base64Opt);
        var emptyStringResults = this.Runner.Run(h.Algorithm, StringCmd, SourceOpt, EmptyStr, Base64Opt);

        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, Base64Opt);

        // Assert
        results.Should().HaveCount(2);
        results[0].Should().Be(string.Format(FileResultTpl, this.emptyFile, emptyStringResults[0], 0));
        results[1].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, stringResults[0], h.InitialString.Length));
    }
        
    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_OutputToFile_FileResultEqualConsoleResult(Hash h)
    {
        // Arrange
        const string save = "result";
        var dir = Path.GetDirectoryName(this.Runner.TestExePath);
        var result = Path.Combine(dir, save);

        try
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, "-o", save);

            // Assert
            results[0].Should().Be(string.Format(FileResultTpl, this.emptyFile, h.EmptyStringHash, 0));
            results[1].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.HashString, h.InitialString.Length));
            results.Should().HaveCount(2);

            File.Exists(result).Should().BeTrue();
            var content = File.ReadAllText(result);
            var fromConsole = string.Join(Environment.NewLine, results) + Environment.NewLine;
            fromConsole.Should().Be(content);
        }
        finally
        {
            if (File.Exists(result))
            {
                File.Delete(result);
            }
        }
    }

    [Theory, MemberData(nameof(HashesWithoutCrc32))]
    [Trait("Category", "hanging")]
    public void CalcDir_SfvHashesThatNotSupportIt_Failure(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, "--sfv");

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be($" --sfv option doesn't support {h.Algorithm} algorithm. Only crc32 or crc32c supported");
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_RecursivelyManySubs_Success(Hash h)
    {
        // Arrange
        const string sub2Suffix = "2";
        Directory.CreateDirectory(this.fixture.SubDir + sub2Suffix);

        this.CreateEmptyFile(Path.Combine(this.fixture.SubDir + sub2Suffix, EmptyFileName));
        this.CreateNotEmptyFile(Path.Combine(this.fixture.SubDir + sub2Suffix, NotEmptyFileName), h.InitialString);

        try
        {
            // Act
            var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, RecurseOpt);
                
            // Assert
            results.Should().HaveCount(6);
        }
        finally
        {
            Directory.Delete(this.fixture.SubDir + sub2Suffix, true);
        }
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_IncludeFilter_OnlyFilesMatchFilterCalculated(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, IncludeOpt, EmptyFileName);

        // Assert
        results[0].Should().Be(string.Format(FileResultTpl, this.emptyFile, h.EmptyStringHash, 0));
        results.Should().HaveCount(1);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcDir_ExcludeFilter_OnlyFilesNotMatchFilterCalculated(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, ExcludeOpt, EmptyFileName);

        // Assert
        results[0].Should().Be(string.Format(FileResultTpl, this.notEmptyFile, h.HashString, h.InitialString.Length));
        results.Should().HaveCount(1);
    }

    [Theory, MemberData(nameof(HashesForCalcDirTheory))]
    public void CalcDir_DifferentOption_ResultAsExpected(Hash h, int countResults, params string[] commandLine)
    {
        // Arrange
        var cmd = new List<string> { h.Algorithm };
        cmd.AddRange([DirCmd, SourceOpt, this.fixture.BaseTestDir]);
        cmd.AddRange(commandLine);
    
        // Act
        var results = this.Runner.Run(cmd.ToArray());
    
        // Assert
        results.Count.Should().Be(countResults);
    }

    public static IEnumerable<object[]> HashesForCalcDirTheory => CreateProperty([
            new object[] { 0, new[] { IncludeOpt, EmptyFileName, ExcludeOpt, EmptyFileName } },
            new object[] { 0, new[] { ExcludeOpt, EmptyFileName + ";" + NotEmptyFileName } },
            new object[] { 2, new[] { IncludeOpt, EmptyFileName + ";" + NotEmptyFileName } },
            new object[] { 2, new[] { IncludeOpt, EmptyFileName, RecurseOpt } },
            new object[] { 2, new[] { ExcludeOpt, EmptyFileName, RecurseOpt } },
            new object[] { 4, new[] { RecurseOpt } }
    ]);

    [Theory, MemberData(nameof(Hashes))]
    public void SearchFile_NotRecursively_OnlyFileThatMatchesPassedHashFound(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, SearchOpt, h.HashString);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileSearchTpl, this.notEmptyFile, h.InitialString.Length));
    }
        
    [Theory, MemberData(nameof(Hashes))]
    public void SearchFile_NotRecursivelyTimed_OnlyFileThatMatchesPassedHashFound(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, SearchOpt, h.HashString, TimeOpt);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().MatchRegex(FileSearchTimeTpl);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void SearchFile_Recursively_OnlyFilesThatMatchesPassedHashFound(Hash h)
    {
        // Act
        var results = this.Runner.Run(h.Algorithm, DirCmd, SourceOpt, this.fixture.BaseTestDir, SearchOpt, h.HashString, RecurseOpt);

        // Assert
        results.Should().HaveCount(2);
    }
}
