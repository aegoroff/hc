/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2014 Alexander Egorov
*/

using System;
using Xunit;

namespace _tst.net
{
    public abstract class ExeWrapper<T> : IUseFixture<T> where T : Architecture, new()
    {
        protected string Arch { get; private set; }

        public ProcessRunner Runner { get; set; }

        protected abstract string Executable { get; }

        public void SetFixture(T data)
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
    }

    public class ArchWin32 : Architecture
    {
        protected override string RelativePath
        {
            get { return @"\..\..\..\Release\{0}"; }
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
            get { return @"\..\..\..\x64\Release\{0}"; }
        }

        public override string Arch
        {
            get { return "x64"; }
        }
    }
}