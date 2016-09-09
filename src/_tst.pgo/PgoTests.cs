/*
* Created by: egr
* Created at: 05.05.2015
* © 2009-2016 Alexander Egorov
*/

using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using _tst.net;

namespace _tst.pgo
{
    public class PgoTests64 : PgoTests<ArchWin64>
    {
        public PgoTests64(ITestOutputHelper output) : base(output)
        {
        }
    }

    public class PgoTests32 : PgoTests<ArchWin32>
    {
        public PgoTests32(ITestOutputHelper output) : base(output)
        {
        }
    }

    [Collection("SerializableTests")]
    public abstract class PgoTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        private const string CrackOpt = "-c";
        private const string HashOpt = "-m";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string NoProbeOpt = "--noprobe";
        private const string DirOpt = "-d";
        private const string IncludeOpt = "-i";
        
        protected PgoTests(ITestOutputHelper output) : base(new T())
        {
            this.Runner.Output = output;
        }

        [Theory, MemberData(nameof(Hashes))]
        public void Cases(Hash h)
        {
            this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "4", MinOpt, "1");
            this.Runner.Run(h.Algorithm, DirOpt, ".", IncludeOpt, "*.exe");
        }

        protected override string Executable => "hc.exe";

        public static IEnumerable<object[]> Hashes => FileTests<ArchWin64>.Hashes;
    }
}