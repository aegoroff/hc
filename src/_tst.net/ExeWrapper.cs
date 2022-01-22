/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2022 Alexander Egorov
*/

using System;
using System.IO;
using Xunit;

namespace _tst.net
{
    public abstract class ExeWrapper<T> : IClassFixture<T>
        where T : Architecture, new()
    {
        protected string Arch { get; }

        protected ProcessRunner Runner { get; }

        protected abstract string Executable { get; }

        protected ExeWrapper(T data)
        {
            this.Arch = data.Arch;
            this.Runner = new ProcessRunner(Path.Combine(data.ExecutableBasePath, this.Executable));
        }
    }

    public abstract class Architecture
    {
        public string ExecutableBasePath => Path.Combine(BasePath, RelativeCommonPath, this.RelativePath);

        private static string BasePath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") ?? Environment.CurrentDirectory;

        protected abstract string RelativePath { get; }

        private static string RelativeCommonPath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") == null ? Path.Combine("..", "..", "..") : string.Empty;

        public abstract string Arch { get; }

#if DEBUG

        internal const string Configuration = "Debug";

#else
        internal const string Configuration = "Release";
#endif
    }

    public class ArchWin64 : Architecture
    {
        protected override string RelativePath => Path.Combine("x64", Configuration);

        public override string Arch => "x64";
    }
}
