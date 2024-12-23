#if WINDOWS
using System.IO;
using System.Runtime.Versioning;

namespace _tst.net;

[SupportedOSPlatform("windows")]
public class ArchWindows : Architecture
{
    protected override string RelativePath => Path.Combine("x64", Configuration);

    public override string Arch => "x64";

    public override string Executable => "hc.exe";
}
#endif
