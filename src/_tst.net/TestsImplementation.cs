/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2022 Alexander Egorov
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

    [Trait("Arch", "x64")]
    [Trait("Category", "x64")]
    public class CmdStringTestsWin64 : CmdStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x64")]
    [Trait("Category", "x64")]
    [Collection("SerializableTests")]
    public class GostTests64 : GostTests<ArchWin64>
    {
    }
}
