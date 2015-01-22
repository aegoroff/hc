/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2015 Alexander Egorov
*/

using Xunit;

namespace _tst.net
{
    [Trait("Arch", "x64")]
    public class CmdFileTestsWin64 : CmdFileTests<ArchWin64>
    {
    }

    [Trait("Arch", "x64")]
    public class QueryFileTestsWin64 : QueryFileTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class CmdFileTestsWin32 : CmdFileTests<ArchWin32>
    {
    }

    [Trait("Arch", "x86")]
    public class QueryFileTestsWin32 : QueryFileTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class CmdStringTestsWin64 : CmdStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class CmdStringTestsWin32 : CmdStringTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class QueryStringTestsWin64: QueryStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class QueryStringTestsWin32: QueryStringTests<ArchWin32>
    {
    }

    [Trait("Arch", "x86")]
    public class GostTests32 : GostTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class GostTests64 : GostTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class Apc32 : Apc<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class Apc64 : Apc<ArchWin64>
    {
    }
}