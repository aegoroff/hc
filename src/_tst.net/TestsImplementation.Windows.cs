/*
 * Created by: egr
 * Created at: 28.10.2007
 * © 2009-2025 Alexander Egorov
 */
#if WINDOWS
using System.Runtime.Versioning;
using Xunit;

namespace _tst.net;

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[Collection("SerializableTests")]
[SupportedOSPlatform("windows")]
public class CmdFileTestsWindows(FileFixture fixture) : CmdFileTests<ArchWindows>(fixture)

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[SupportedOSPlatform("windows")]
public class CmdStringTestsWindows : CmdStringTests<ArchWindows>

[Trait("Arch", "x64")]
[Trait("Category", "x64")]
[Collection("SerializableTests")]
[SupportedOSPlatform("windows")]
public class GostTestsWindows : GostTests<ArchWindows>
#endif
