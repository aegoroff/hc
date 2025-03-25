#if !WINDOWS
using System;
using System.Runtime.Versioning;

namespace _tst.net;

[SupportedOSPlatform("linux")]
public class ArchLinux : Architecture
{
    protected override string RelativePath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") == null
                                                      ? "build"
                                                      : $"build-x86_64-linux-gnu-{Configuration}";

    public override string Arch => "x64";

    public override string Executable => "hc";
}
#endif
