/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2011 Alexander Egorov
 */

using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace trid
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

        /// <summary>
        /// Runs executable
        /// </summary>
        /// <returns>Standart ouput strings</returns>
        public IList<string> Run( params string[] commandLine )
        {
            StringBuilder sb = new StringBuilder();

            foreach ( string parameter in commandLine )
            {
                sb.AddParameter(parameter);
            }

            Process app = new Process
                              {
                                  StartInfo =
                                      {
                                          FileName = testExePath,
                                          Arguments = sb.ToString(),
                                          UseShellExecute = false,
                                          RedirectStandardOutput = true,
                                          WorkingDirectory = testExePath.GetDirectoryName(),
                                          CreateNoWindow = true
                                      }
                              };

            IList<string> result;
            using ( app )
            {
                app.Start();

                result = app.StandardOutput.ReadLines();

                app.WaitForExit();
            }
            return result;
        }
    }

    public static class Extensions
    {
        private const string EscapeSymbol = "\"";

        public static void AddParameter( this StringBuilder builder, string parameter )
        {
            if ( parameter.Contains(" ") )
            {
                builder.Append(EscapeSymbol);
                builder.Append(parameter);
                builder.Append(EscapeSymbol);
            }
            else
            {
                builder.Append(parameter);
            }
            builder.Append(" ");
        }

        public static IList<string> ReadLines( this StreamReader reader )
        {
            List<string> result = new List<string>();

            while ( !reader.EndOfStream )
            {
                result.Add(reader.ReadLine());
            }

            return result;
        }

        internal static string GetDirectoryName(this string path)
        {
            return Path.GetDirectoryName(Path.GetFullPath(path));
        }
    }
}