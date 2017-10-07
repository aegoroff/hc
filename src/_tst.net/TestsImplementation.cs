/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2017 Alexander Egorov
*/

using Xunit;

namespace _tst.net
{
    [Trait("Arch", "x64")]
    [Trait("Category", "x64")]
    [Collection("SerializableTests")]
    public class CmdFileTestsWin64 : CmdFileTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    [Trait("Category", "x86")]
    [Collection("SerializableTests")]
    public class CmdFileTestsWin32 : CmdFileTests<ArchWin32>
    {
    }


    [Trait("Arch", "x64")]
    [Trait("Category", "x64")]
    public class CmdStringTestsWin64 : CmdStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    [Trait("Category", "x86")]
    public class CmdStringTestsWin32 : CmdStringTests<ArchWin32>
    {
    }

    [Trait("Arch", "x86")]
    [Trait("Category", "x86")]
    [Collection("SerializableTests")]
    public class GostTests32 : GostTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    [Trait("Category", "x64")]
    [Collection("SerializableTests")]
    public class GostTests64 : GostTests<ArchWin64>
    {
    }
}
