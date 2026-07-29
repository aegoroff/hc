/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2026 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Xunit.Abstractions;

namespace _tst.net;

///<summary>
/// Represents an executable file run wrapper
///</summary>
public sealed class ProcessRunner
{
    ///<summary>
    /// Initializes a new instance of the <see cref="ProcessRunner"/> class
    ///</summary>
    ///<param name="testExePath">Path to executable file</param>
    public ProcessRunner(string testExePath) => this.TestExePath = testExePath;

    public string TestExePath
    {
        get;
    }

    public ITestOutputHelper Output { get; set; }

    [Conditional("PROFILE_TESTS")]
    private void OutputParameters(StringBuilder sb) => this.WriteLine(sb.ToString());

    private void WriteLine(string format, params object[] args)
    {
        if (this.Output == null)
        {
            Console.WriteLine(format, args);
        }
        else
        {
            this.Output.WriteLine(format, args);
        }
    }

    /// <summary>
    /// Runs executable
    /// </summary>
    /// <returns>Standard output strings</returns>
    public IList<string> Run(params string[] commandLine)
    {
        var sb = new StringBuilder();

        foreach (var parameter in commandLine)
        {
            sb.AddParameter(parameter);
        }

        this.OutputParameters(sb);

        var args = sb.ToString();

        var app = new Process
                  {
                          StartInfo =
                          {
                                  FileName = this.TestExePath,
                                  Arguments = args,
                                  UseShellExecute = false,
                                  RedirectStandardOutput = true,
                                  WorkingDirectory = this.TestExePath.GetDirectoryName(),
                                  CreateNoWindow = true
                          }
                  };

        IList<string> result = new List<string>();
#if PROFILE_TESTS
            var sw = new Stopwatch();
#endif
        using (app)
        {
            app.OutputDataReceived += delegate(object sender, DataReceivedEventArgs eventArgs)
            {
                if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    result.Add(eventArgs.Data);
                }
            };
#if PROFILE_TESTS
                sw.Start();
#endif
            app.Start();
            app.BeginOutputReadLine();
#if PROFILE_TESTS
                sw.Stop();
                this.WriteLine("Run: {0} time: {1}", Path.GetFileName(executable), sw.Elapsed);
#endif

            app.WaitForExit();
        }
        // Drop GPU fallback/diagnostic noise so crack assertions stay stable
        // with or without a working NVIDIA driver/toolkit.
        return result.Where(s => !IsGpuDiagnosticLine(s)).ToList();
    }

    private static bool IsGpuDiagnosticLine(string line) =>
            line.Contains("GPU present but driver's CUDA version", StringComparison.Ordinal) ||
            line.Contains("GPU unavailable (driver/toolkit); using CPU only", StringComparison.Ordinal);
}
