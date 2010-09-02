/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace _tst.net
{
	///<summary>
	/// Represents Google test executable run wrapper
	///</summary>
	internal sealed class ProcessRunner
	{
		private readonly string _testExePath;

		///<summary>
		/// Initializes a new instance of the <see cref="ProcessRunner"/> class
		///</summary>
		///<param name="testExePath">Path to Google test executable</param>
		public ProcessRunner( string testExePath )
		{
			_testExePath = testExePath;
		}

		/// <summary>
		/// Runs exe
		/// </summary>
		/// <returns>Standart ouput strings</returns>
		public IList<string> Run( string commandLine )
		{
			string dir = Path.GetDirectoryName(Path.GetFullPath(_testExePath));

			List<string> result = new List<string>();

			Process app = new Process
			              	{
			              		StartInfo =
			              			{
			              				FileName = _testExePath,
			              				Arguments = commandLine,
			              				UseShellExecute = false,
			              				RedirectStandardOutput = true,
			              				WorkingDirectory = dir,
			              				CreateNoWindow = true
			              			}
			              	};

			using ( app )
			{
				app.Start();

				while ( !app.StandardOutput.EndOfStream )
				{
					result.Add(app.StandardOutput.ReadLine());
				}

				app.WaitForExit();
			}
			return result;
		}
	}
}