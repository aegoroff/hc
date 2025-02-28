/*
 * Created by: egr
 * Created at: 28.10.2007
 * © 2009-2025 Alexander Egorov
 */
#if !WINDOWS
using System.Runtime.Versioning;
using Xunit;

namespace _tst.net;

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[Collection("SerializableTests")]
[SupportedOSPlatform("linux")]
public class CmdFileTestsLinux : CmdFileTests<ArchLinux>
{
}

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[SupportedOSPlatform("linux")]
public class CmdStringTestsLinux : CmdStringTests<ArchLinux>
{
}

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[Collection("SerializableTests")]
[SupportedOSPlatform("linux")]
public class GostTestsLinux : GostTests<ArchLinux>
{
}
#endif
