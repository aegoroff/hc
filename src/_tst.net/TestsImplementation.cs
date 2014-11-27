/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2013 Alexander Egorov
*/

using Xunit;

namespace _tst.net
{
    [Trait("Arch", "x64")]
    public class HashCalculatorFileTestsWin64 : HashCalculatorFileTests<ArchWin64>
    {
    }

    [Trait("Arch", "x64")]
    public class HashQueryFileTestsWin64 : HashQueryFileTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class HashCalculatorFileTestsWin32 : HashCalculatorFileTests<ArchWin32>
    {
    }

    [Trait("Arch", "x86")]
    public class HashQueryFileTestsWin32 : HashQueryFileTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class HashCalculatorStringTestsWin64 : HashCalculatorStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class HashCalculatorStringTestsWin32 : HashCalculatorStringTests<ArchWin32>
    {
    }

    [Trait("Arch", "x64")]
    public class HashQueryStringTestsWin64: HashQueryStringTests<ArchWin64>
    {
    }

    [Trait("Arch", "x86")]
    public class HashQueryStringTestsWin32: HashQueryStringTests<ArchWin32>
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