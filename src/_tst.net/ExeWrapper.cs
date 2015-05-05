/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2015 Alexander Egorov
*/

using System;
using Xunit;

namespace _tst.net
{
    public abstract class ExeWrapper<T> : IClassFixture<T> where T : Architecture, new()
    {
        protected string Arch { get; private set; }

        public ProcessRunner Runner { get; set; }

        protected abstract string Executable { get; }

        protected ExeWrapper(T data)
        {
            this.Arch = data.Arch;
            this.Runner = new ProcessRunner(string.Format(data.PathTemplate, this.Executable));
        }
    }

    public abstract class Architecture
    {
        public string PathTemplate
        {
            get { return Environment.CurrentDirectory + this.RelativePath; }
        }

        protected abstract string RelativePath { get; }

        public abstract string Arch { get; }

#if DEBUG
        internal const string Configuration = "Debug";
#else
        internal const string Configuration = "Release";
#endif


    }

    public class ArchWin32 : Architecture
    {
        protected override string RelativePath
        {
            get { return @"\..\..\..\" + Configuration + @"\{0}"; }
        }

        public override string Arch
        {
            get { return "x86"; }
        }
    }

    public class ArchWin64 : Architecture
    {
        protected override string RelativePath
        {
            get { return @"\..\..\..\x64\" + Configuration + @"\{0}"; }
        }

        public override string Arch
        {
            get { return "x64"; }
        }
    }
}