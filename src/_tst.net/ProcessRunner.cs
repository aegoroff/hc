/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2019 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Xunit.Abstractions;

namespace _tst.net
{
    ///<summary>
    /// Represents an executable file run wrapper
    ///</summary>
    public sealed class ProcessRunner
    {
        private readonly string testExePath;

        ///<summary>
        /// Initializes a new instance of the <see cref="ProcessRunner"/> class
        ///</summary>
        ///<param name="testExePath">Path to executable file</param>
        public ProcessRunner(string testExePath)
        {
            this.testExePath = testExePath;
        }

        public string TestExePath => this.testExePath;

        public ITestOutputHelper Output { get; set; }

        [Conditional("PROFILE_TESTS")]
        private void OutputParameters(StringBuilder sb)
        {
            this.WriteLine(sb.ToString());
        }

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
        /// <returns>Standart ouput strings</returns>
        public IList<string> Run(params string[] commandLine)
        {
            var sb = new StringBuilder();

            foreach (var parameter in commandLine)
            {
                sb.AddParameter(parameter);
            }

            this.OutputParameters(sb);

            var parts = this.TestExePath.Split('\\');
            var exe = parts[parts.Length - 1].Split(' ');
            var executable = this.TestExePath;
            var args = sb.ToString();
            if (exe.Length > 1)
            {
                executable = this.TestExePath.Substring(0, this.TestExePath.Length - exe[exe.Length - 1].Length);
                args = exe[exe.Length - 1] + " " + sb;
            }

            var app = new Process
                      {
                          StartInfo =
                          {
                              FileName = executable,
                              Arguments = args,
                              UseShellExecute = false,
                              RedirectStandardOutput = true,
                              WorkingDirectory = executable.GetDirectoryName(),
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
            return result;
        }
    }
}
