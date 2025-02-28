/*
 * Created by: egr
 * Created at: 05.05.2015
 * © 2009-2025 Alexander Egorov
 */
#if WINDOWS
using System.Collections.Generic;
using System.Runtime.Versioning;
using _tst.net;
using FluentAssertions;
using Xunit;
using Xunit.Abstractions;

namespace _tst.pgo;

[Trait("Arch", "x64")]
[SupportedOSPlatform("windows")]
public class PgoTestsWindows(ITestOutputHelper output) : PgoTests<ArchWindows>(output);

[Collection("SerializableTests")]
public abstract class PgoTests<T> : ExeWrapper<T>
        where T : Architecture, new()
{
    private const string CrackOpt = "hash";

    private const string SourceOpt = "-s";

    private const string MaxOpt = "-x";

    private const string MinOpt = "-n";

    private const string DictOpt = "-a";

    private const string NoProbeOpt = "--noprobe";

    private const string IncludeOpt = "-i";

    protected PgoTests(ITestOutputHelper output) : base(new T()) => this.Runner.Output = output;

    public static IEnumerable<object[]> Hashes => FileTests<ArchWindows>.Hashes;

    [Theory]
    [MemberData(nameof(Hashes))]
    public void Cases(Hash h)
    {
        // Act
        var r1 = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, SourceOpt, h.HashString, MaxOpt, "6", MinOpt, "1", DictOpt, "ASCII");
        var r2 = this.Runner.Run(h.Algorithm, "dir", SourceOpt, ".", IncludeOpt, "*.exe");

        // Assert
        r1.Should().HaveCount(2);
        r2.Should().HaveCount(3);
    }
}
#endif
