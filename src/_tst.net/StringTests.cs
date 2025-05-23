/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2025 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Linq;
using FluentAssertions;
using Xunit;

namespace _tst.net;

[Trait("Group", "string")]
[Trait("Category", "string")]
public abstract class StringTests<T>(T data) : ExeWrapper<T>(data)
        where T : Architecture, new()
{
    protected const string RestoredStringTemplate = "Initial string is: {0}";

    private const string NothingFound = "Nothing found";

    protected static IEnumerable<object[]> CreateProperty(object[] data) =>
            from h in Hashes from item in data select new[] { h[0], item };

    public static IEnumerable<object[]> HashesAndNonDefaultDict => CreateProperty(["123", "0-9", "0-9a-z", "0-9A-Z"]);

    public static IEnumerable<object[]> HashesAndNonDefaultDictFailure =>
            CreateProperty(["a-zA-Z", "a-z", "A-Z", "abc"]);

    protected abstract IList<string> RunEmptyStringCrack(Hash h);

    protected abstract IList<string> RunStringCrack(Hash h);

    protected abstract IList<string> RunStringCrackTooShort(Hash h);

    protected abstract IList<string> RunStringCrackTooMinLength(Hash h);

    protected abstract IList<string> RunStringHash(Hash h);

    protected abstract IList<string> RunStringHashLowCase(Hash h);

    protected abstract IList<string> RunEmptyStringHash(Hash h);

    protected abstract IList<string> RunStringCrackLowCaseHash(Hash h);

    protected abstract IList<string> RunCrackStringUsingNonDefaultDictionary(Hash h, string dict);

    protected abstract IList<string> RunStringHashAsBase64(Hash h);

    [Theory, MemberData(nameof(Hashes))]
    public void CalcString_FullString_ResultAsExpected(Hash h)
    {
        // Act
        var results = this.RunStringHash(h);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(h.HashString);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcString_AsBase64_ResultAsExpected(Hash h)
    {
        // Arrange
        var bytes = Convert.FromHexString(h.HashString);
        var base64 = Convert.ToBase64String(bytes);

        // Act
        var results = this.RunStringHashAsBase64(h);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(base64);
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcString_FullStringLowCaseOutput_ResultAsExpected(Hash h)
    {
        // Act
        var results = this.RunStringHashLowCase(h);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(h.HashString.ToLowerInvariant());
    }

    [Theory, MemberData(nameof(Hashes))]
    public void CalcString_EmptyString_ExpectedEmtyStringHash(Hash h)
    {
        // Act
        var results = this.RunEmptyStringHash(h);

        // Assert
        results.Should().HaveCount(1);
        results[0].Should().Be(h.EmptyStringHash);
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(Hashes))]
    public void CrackString_DefaultOptions_Success(Hash h)
    {
        // Act
        var results = this.RunStringCrack(h);

        // Assert
        results.Should().HaveCount(2);
        results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString));
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(Hashes))]
    public void CrackString_Empty_Success(Hash h)
    {
        // Act
        var results = this.RunEmptyStringCrack(h);

        // Assert
        results[0].Should().Be("Attempts: 0 Time 00:00:0.000 Speed: 0 attempts/second");
        results[1].Should().Be(string.Format(RestoredStringTemplate, "Empty string"));
        results.Should().HaveCount(2);
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(Hashes))]
    public void CrackString_UsingLowCaseHash_Success(Hash h)
    {
        // Act
        var results = this.RunStringCrackLowCaseHash(h);

        // Assert
        results.Should().HaveCount(2);
        results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString));
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(HashesAndNonDefaultDict))]
    public void CrackString_UsingNonDefaultDictionary_Success(Hash h, string dict)
    {
        // Act
        var results = this.RunCrackStringUsingNonDefaultDictionary(h, dict);

        // Assert
        results[1].Should().Be(string.Format(RestoredStringTemplate, h.InitialString.Substring(0, 2)));
        results.Should().HaveCount(2);
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(HashesAndNonDefaultDictFailure))]
    public void CrackString_UsingNonDefaultDictionary_Failure(Hash h, string dict)
    {
        // Act
        var results = this.RunCrackStringUsingNonDefaultDictionary(h, dict);

        // Assert
        results[1].Should().Be(NothingFound);
        results.Should().HaveCount(2);
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(Hashes))]
    public void CrackString_TooShortLength_Failure(Hash h)
    {
        // Act
        var results = this.RunStringCrackTooShort(h);

        // Assert
        results.Should().HaveCount(2);
        results[1].Should().Be(NothingFound);
    }

    [Trait("Type", "crack")]
    [Theory, MemberData(nameof(Hashes))]
    public void CrackString_TooLongMinLength_Failure(Hash h)
    {
        // Act
        var results = this.RunStringCrackTooMinLength(h);

        // Assert
        results.Should().HaveCount(2);
        results[1].Should().Be(NothingFound);
    }

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
            [new Ntlm()],
            [new Blake2b()],
            [new Blake2s()],
            [new Blake3()]
    ];
}
