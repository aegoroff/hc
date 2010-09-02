/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NUnit.Framework;

namespace _tst.net
{
	public abstract class THashCalculator
	{
		private static readonly string PathTemplate = Environment.CurrentDirectory + @"\..\..\..\Release\{0}";
		private const string CrackTemplate = "-c -m {0}";
		private const string CalcStringTemplate = "-s {0}";
		private const string CalcFileTemplate = "-f {0}";
		private const string CalcDirTemplate = "-d {0}";
		private const string EmptyTemplate = "\"{0}\"";
		private const string RestoredStringTemplate = "Initial string is: {0}";
		private const string LowCaseFlag = "-l";
		const string NothingFound = "Nothing found";
		const string BaseTestDir = @"C:\_tst.net";
		const string EmptyFileName = "empty";
		const string NotEmptyFileName = "notempty";
		const string Slash = @"\";
		const string FileResultTpl = @"{0} | {2} bytes | {1}";
		const string NotEmptyFile = BaseTestDir + Slash + NotEmptyFileName;
		const string EmptyFile = BaseTestDir + Slash + EmptyFileName;

		protected abstract string Executable { get; }
		
		protected abstract string HashString { get; }
		
		protected abstract string EmptyStringHash { get; }

		protected virtual string InitialString
		{
			get { return "123"; }
		}

		private ProcessRunner _runner;

		[SetUp]
		public void Setup()
		{
			_runner = new ProcessRunner(string.Format(PathTemplate, Executable));
		}

		[TestFixtureSetUp]
		public void TestFixtureSetup()
		{
			if (!Directory.Exists(BaseTestDir))
			{
				Directory.CreateDirectory(BaseTestDir);
			}
			else
			{
				Directory.Delete(BaseTestDir, true);
				Directory.CreateDirectory(BaseTestDir);
			}
			using (File.Create(EmptyFile))
			{
			}

			FileStream fs = File.Create(NotEmptyFile);
			using ( fs )
			{
				byte[] unicode = Encoding.Unicode.GetBytes(InitialString);
				byte[] buffer = Encoding.Convert(Encoding.Unicode, Encoding.ASCII, unicode);
				fs.Write(buffer, 0, buffer.Length);
			}
		}

		[TestFixtureTearDown]
		public void TestFixtureTearDown()
		{
			if (Directory.Exists(BaseTestDir))
			{
				Directory.Delete(BaseTestDir, true);
			}
		}

		[Test]
		public void CalcString()
		{
			
			IList<string> results = _runner.Run(string.Format(CalcStringTemplate, InitialString));
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(HashString));
		}
		
		[Test]
		public void CalcStringLowCaseOutput()
		{
			IList<string> results = _runner.Run(string.Format(CalcStringTemplate, InitialString) + " " + LowCaseFlag);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
		}

		[Test]
		public void CalcEmptyString()
		{
			IList<string> results = _runner.Run(string.Format(CalcStringTemplate, string.Format(EmptyTemplate, string.Empty)));
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(EmptyStringHash));
		}
		
		[Test]
		public void CrackString()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString));
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}
		
		[Test]
		public void CrackStringUsingLowCaseHash()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString.ToLowerInvariant()));
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}
		
		[Test]
		public void CrackStringUsingNonDefaultDictionary()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString) + " -a 12345");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}

		[Test]
		public void CrackStringBadDictionary()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString) + " -a abcd");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}
		
		[Test]
		public void CrackStringTooShortLength()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString) + " -x 2");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}
		
		[Test]
		public void CrackStringTooLongMinLength()
		{
			IList<string> results = _runner.Run(string.Format(CrackTemplate, HashString) + " -n 4 -x 5 -a 12345");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}

		[Test]
		public void CalcFile()
		{

			IList<string> results = _runner.Run(string.Format(CalcFileTemplate, NotEmptyFile));
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
		
		[Test]
		public void CalcUnexistFile()
		{
			const string unexist = "u";
			IList<string> results = _runner.Run(string.Format(CalcFileTemplate, unexist));
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format("{0} | The system cannot find the file specified.  ", unexist)));
		}
		
		[Test]
		public void CalcEmptyFile()
		{

			IList<string> results = _runner.Run(string.Format(CalcFileTemplate, EmptyFile));
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
		}

		[Test]
		public void CalcDir()
		{
			IList<string> results = _runner.Run(string.Format(CalcDirTemplate, BaseTestDir));
			Assert.That(results.Count, Is.EqualTo(2));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
			Assert.That(results[1], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
		
		[Test]
		public void CalcDirIncludeFilter()
		{
			IList<string> results = _runner.Run(string.Format(CalcDirTemplate, BaseTestDir) + " -i " + EmptyFileName);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
		}
		
		[Test]
		public void CalcDirExcludeFilter()
		{
			IList<string> results = _runner.Run(string.Format(CalcDirTemplate, BaseTestDir) + " -e " + EmptyFileName);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
	}
}
