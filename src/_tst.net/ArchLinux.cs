#if !WINDOWS
using System.Runtime.Versioning;

namespace _tst.net;

[SupportedOSPlatform("linux")]
public class ArchLinux : Architecture
{
    protected override string RelativePath => "build";

    public override string Arch => "x64";
    
    public override string Executable => "hc";
}
#endif
