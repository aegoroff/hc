/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2025 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using FluentAssertions;
using Xunit;

namespace _tst.net;

public class FileFixture : IDisposable
{
    internal readonly string BaseTestDir;

    private static readonly string DefaultBaseTestDir =
            OperatingSystem.IsWindows() ? @"C:\_tst.net" : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".local", "share", "_tst.net");

    internal readonly string SubDir;

    public FileFixture()
    {
        this.Dispose();
        this.BaseTestDir = (Environment.GetEnvironmentVariable("HC_TEST_DIR") ?? DefaultBaseTestDir).Trim();
        this.SubDir = Path.Combine(this.BaseTestDir, "sub");
        Directory.CreateDirectory(this.BaseTestDir);
        Directory.CreateDirectory(this.SubDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(this.BaseTestDir))
        {
            Directory.Delete(this.BaseTestDir, true);
        }
    }
}

[Trait("Group", "file")]
[Trait("Category", "file")]
public abstract class FileTests<T> : ExeWrapper<T>, IClassFixture<FileFixture>
        where T : Architecture, new()
{
    protected abstract string EmptyFileNameProp { get; }

    protected abstract string EmptyFileProp { get; }

    protected abstract string NotEmptyFileNameProp { get; }

    protected abstract string NotEmptyFileProp { get; }

    protected FileTests(FileFixture fixture) : base(new T()) => this.Initialize(fixture);

    private void Initialize(FileFixture fixture)
    {
        Hash h = new Md5();
        this.CreateEmptyFile(this.EmptyFileProp);
        this.CreateNotEmptyFile(this.NotEmptyFileProp, h.InitialString);

            
        this.CreateEmptyFile(Path.Combine(fixture.SubDir, this.EmptyFileNameProp));
        this.CreateNotEmptyFile(Path.Combine(fixture.SubDir, this.NotEmptyFileNameProp), h.InitialString);
    }

    protected void CreateNotEmptyFile(string path, string s, int minSize = 0)
    {
        using var fs = File.Create(path);
        var unicode = Encoding.Unicode.GetBytes(s);
        var buffer = Encoding.Convert(Encoding.Unicode, Encoding.ASCII, unicode);

        var written = 0;
        do
        {
            written += buffer.Length;
            fs.Write(buffer, 0, buffer.Length);
        } while (written <= minSize);
    }

    protected void CreateEmptyFile(string path)
    {
        using (File.Create(path))
        {
        }
    }

    protected static IEnumerable<object[]> CreateProperty(object[] data)
    {
        foreach (var h in Hashes)
        {
            foreach (var item in data)
            {
                if (item is not object[] items)
                {
                    yield return [h[0], item];
                }
                else
                {
                    var result = new List<object> { h[0] };
                    result.AddRange(items);
                    yield return result.ToArray();
                }
            }
        }
    }

    protected const string FileResultTpl = @"{0} | {2} bytes | {1}";

    protected const string FileErrorTpl = @"{0} | {1}";

    protected const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";

    private const string FileResultSfvTpl = @"{0}    {1}";

    protected abstract IList<string> RunFileHashCalculation(Hash h, string file);

    protected abstract IList<string> RunDirWithSpecialOption(Hash h, string option);

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Small_HashAndOutputAsExpected(Hash h)
    {
        // Act
        var results = this.RunFileHashCalculation(h, this.NotEmptyFileProp);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(string.Format(FileResultTpl, this.NotEmptyFileProp, h.HashString, h.InitialString.Length));
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcFile_Big_CalcSuccess(Hash h)
    {
        // Arrange
        var file = this.NotEmptyFileProp + "_big";
        this.CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
        try
        {
            // Act
            var results = this.RunFileHashCalculation(h, file);

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
    public void CalcDir_Checksumfile_CalculateSuccess(Hash h)
    {
        // Act
        var results = this.RunDirWithSpecialOption(h, "--checksumfile");

        // Assert
        results.Should().HaveCount(2);
        results[0].Should().Be(string.Format(FileResultSfvTpl, h.EmptyStringHash, this.EmptyFileProp));
        results[1].Should().Be(string.Format(FileResultSfvTpl, h.HashString, this.NotEmptyFileProp));
    }

    [Fact]
    public void CalcDir_SfvCrc32_CalculateSuccess()
    {
        // Arrange
        Hash h = new Crc32();

        // Act
        var results = this.RunDirWithSpecialOption(h, "--sfv");

        // Assert
        results.Should().HaveCount(2);
        results[0].Should().Be(string.Format(FileResultSfvTpl, Path.GetFileName(this.EmptyFileProp), h.EmptyStringHash));
        results[1].Should().Be(string.Format(FileResultSfvTpl, Path.GetFileName(this.NotEmptyFileProp), h.HashString));
    }

    public static IEnumerable<object[]> HashesWithoutCrc32 => from h in Hashes
                                                              where ((Hash)h[0]).Algorithm != "crc32" && ((Hash)h[0]).Algorithm != "crc32c"
                                                              select new[] { h[0] };

    public static IEnumerable<object[]> Hashes =>
    [
            [new Md4()],
            [new Md5()],
            [new Md2()],
            [new Sha1()],
            [new Sha224()],
            [new Sha256()],
            [new Sha384()],
            [new Sha512()],
            [new Whirlpool()],
            [new Crc32()],
            [new Crc32c()],
            [new Tiger()],
            [new Tiger2()],
            [new Ripemd128()],
            [new Ripemd160()],
            [new Ripemd256()],
            [new Ripemd320()],
            [new Gost()],
            [new Snefru128()],
            [new Snefru256()],
            [new Tth()],
            [new Haval_128_3()],
            [new Haval_128_4()],
            [new Haval_128_5()],
            [new Haval_160_3()],
            [new Haval_160_4()],
            [new Haval_160_5()],
            [new Haval_192_3()],
            [new Haval_192_4()],
            [new Haval_192_5()],
            [new Haval_224_3()],
            [new Haval_224_4()],
            [new Haval_224_5()],
            [new Haval_256_3()],
            [new Haval_256_4()],
            [new Haval_256_5()],
            [new Edonr256()],
            [new Edonr512()],
            [new Sha_3_224()],
            [new Sha_3_256()],
            [new Sha_3_384()],
            [new Sha_3_512()],
            [new Sha_3k_224()],
            [new Sha_3k_256()],
            [new Sha_3k_384()],
            [new Sha_3k_512()],
            [new Blake2b()],
            [new Blake2s()],
            [new Blake3()]
    ];
}
