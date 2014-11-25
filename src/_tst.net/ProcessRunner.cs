/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2009-2013 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

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
        public ProcessRunner( string testExePath )
        {
            this.testExePath = testExePath;
        }

        public string TestExePath
        {
            get { return this.testExePath; }
        }
        
        [Conditional("DEBUG")]
        static void OutputParameters(StringBuilder sb)
        {
            Console.WriteLine(sb.ToString());
        }

        /// <summary>
        /// Runs executable
        /// </summary>
        /// <returns>Standart ouput strings</returns>
        public IList<string> Run( params string[] commandLine )
        {
            var sb = new StringBuilder();

            foreach ( var parameter in commandLine )
            {
                sb.AddParameter(parameter);
            }
            
            OutputParameters(sb);
            
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

            IList<string> result;
            using ( app )
            {
                app.Start();

                result = app.StandardOutput.ReadLines().Result;

                app.WaitForExit();
            }
            return result;
        }
    }
}