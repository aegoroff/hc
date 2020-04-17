/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2019 Alexander Egorov
*/

using System;
using Xunit;

namespace _tst.net
{
    public abstract class ExeWrapper<T> : IClassFixture<T>
        where T : Architecture, new()
    {
        protected string Arch { get; private set; }

        protected ProcessRunner Runner { get; private set; }

        protected abstract string Executable { get; }

        protected ExeWrapper(T data)
        {
            this.Arch = data.Arch;
            this.Runner = new ProcessRunner(string.Format(data.PathTemplate, this.Executable));
        }
    }

    public abstract class Architecture
    {
        public string PathTemplate => BasePath.Trim().TrimEnd('\\') + RelativeCommonPath + this.RelativePath;

        private static string BasePath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") ?? Environment.CurrentDirectory;

        protected abstract string RelativePath { get; }

        private static string RelativeCommonPath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") == null ? @"\..\..\..\" : @"\";

        public abstract string Arch { get; }

#if DEBUG

        internal const string Configuration = "Debug";

#else
        internal const string Configuration = "Release";
#endif
    }

    public class ArchWin64 : Architecture
    {
        protected override string RelativePath => @"x64\" + Configuration + @"\{0}";

        public override string Arch => "x64";
    }
}
