/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2026 Alexander Egorov
*/

using System;
using System.IO;
using Xunit;

namespace _tst.net;

public abstract class ExeWrapper<T>(T data) : IClassFixture<T>
        where T : Architecture, new()
{
    protected string Arch { get; } = data.Arch;

    protected ProcessRunner Runner { get; } = new(Path.Combine(data.ExecutableBasePath, data.Executable));
}

public abstract class Architecture
{
    public string ExecutableBasePath => Path.Combine(BasePath, RelativeCommonPath, this.RelativePath);

    private static string BasePath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") ?? Environment.CurrentDirectory;

    protected abstract string RelativePath { get; }

    private static string RelativeCommonPath => Environment.GetEnvironmentVariable("PROJECT_BASE_PATH") == null ? DefaultRelativeCommonPath : string.Empty;

    private static string DefaultRelativeCommonPath => OperatingSystem.IsWindows()
                                                               ? Path.Combine("..", "..", "..")
                                                               : Path.Combine("..", "..", "..", "..");

    public abstract string Arch { get; }
    
    public abstract string Executable { get; }

#if DEBUG

    internal const string Configuration = "Debug";

#else
        internal const string Configuration = "Release";
#endif
}
