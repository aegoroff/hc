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
		private const string EmptyStr = "\"\"";
		private const string RestoredStringTemplate = "Initial string is: {0}";
		const string NothingFound = "Nothing found";
		const string BaseTestDir = @"C:\_tst.net";
		const string SubDir = BaseTestDir + Slash + "sub";
		const string EmptyFileName = "empty";
		const string NotEmptyFileName = "notempty";
		const string Slash = @"\";
		const string FileResultTpl = @"{0} | {2} bytes | {1}";
		const string NotEmptyFile = BaseTestDir + Slash + NotEmptyFileName;
		const string EmptyFile = BaseTestDir + Slash + EmptyFileName;
		private const string LowCaseOpt = "-l";
		private const string IncludeOpt = "-i";
		private const string ExcludeOpt = "-e";
		private const string RecurseOpt = "-r";
		private const string DictOpt = "-a";
		private const string MaxOpt = "-x";
		private const string MinOpt = "-n";
		private const string CrackOpt = "-c";
		private const string HashOpt = "-m";
		private const string StringOpt = "-s";
		private const string DirOpt = "-d";
		private const string FileOpt = "-f";
		private const string SearchOpt = "-h";

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

			Directory.CreateDirectory(SubDir);

			CreateEmptyFile(EmptyFile);
			CreateNotEmptyFile(NotEmptyFile);

			CreateEmptyFile(SubDir + Slash + EmptyFileName);
			CreateNotEmptyFile(SubDir + Slash + NotEmptyFileName);
		}

		private void CreateNotEmptyFile(string path, int minSize = 0)
		{
			FileStream fs = File.Create(path);
			using ( fs )
			{
				byte[] unicode = Encoding.Unicode.GetBytes(InitialString);
				byte[] buffer = Encoding.Convert(Encoding.Unicode, Encoding.ASCII, unicode);

				int written = 0;
				do
				{
					written += buffer.Length;
					fs.Write(buffer, 0, buffer.Length);
				} while ( written <= minSize );
			}
		}

		private static void CreateEmptyFile( string path )
		{
			using (File.Create(path))
			{
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
			
			IList<string> results = _runner.Run(StringOpt, InitialString);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(HashString));
		}
		
		[Test]
		public void CalcStringLowCaseOutput()
		{
			IList<string> results = _runner.Run(StringOpt, InitialString, LowCaseOpt);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(HashString.ToLowerInvariant()));
		}

		[Test]
		public void CalcEmptyString()
		{
			IList<string> results = _runner.Run(StringOpt, EmptyStr);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(EmptyStringHash));
		}
		
		[Test]
		public void CrackString()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString);
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}
		
		[Test]
		public void CrackStringUsingLowCaseHash()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString.ToLowerInvariant());
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}
		
		[Test]
		public void CrackStringUsingNonDefaultDictionary()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString, DictOpt, "12345");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(string.Format(RestoredStringTemplate, InitialString)));
		}

		[Test]
		public void CrackStringBadDictionary()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString, DictOpt, "abcd");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}
		
		[Test]
		public void CrackStringTooShortLength()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString, MaxOpt, "2");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}
		
		[Test]
		public void CrackStringTooLongMinLength()
		{
			IList<string> results = _runner.Run(CrackOpt, HashOpt, HashString, MinOpt, "4", MaxOpt, "5", DictOpt, "12345");
			Assert.That(results.Count, Is.EqualTo(3));
			Assert.That(results[2], Is.EqualTo(NothingFound));
		}

		[Test]
		public void CalcFile()
		{

			IList<string> results = _runner.Run(FileOpt, NotEmptyFile);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
		
		[Test]
		public void CalcBigFile()
		{
			const string file = NotEmptyFile + "_big";
			CreateNotEmptyFile(file, 2 * 1024 * 1024);
			try
			{
				IList<string> results = _runner.Run(FileOpt, file);
				Assert.That(results.Count, Is.EqualTo(1));
			}
			finally
			{
				File.Delete(file);
			}
		}
		
		[Test]
		public void CalcUnexistFile()
		{
			const string unexist = "u";
			IList<string> results = _runner.Run(FileOpt, unexist);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format("{0} | The system cannot find the file specified.  ", unexist)));
		}
		
		[Test]
		public void CalcEmptyFile()
		{

			IList<string> results = _runner.Run(FileOpt, EmptyFile);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
		}

		[Test]
		public void CalcDir()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir);
			Assert.That(results.Count, Is.EqualTo(2));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
			Assert.That(results[1], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
		
		[Test]
		public void CalcDirRecursively()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, RecurseOpt);
			Assert.That(results.Count, Is.EqualTo(4));
		}
		
		[Test]
		public void CalcDirRecursivelyManySubs()
		{
			const string sub2Suffix = "2";
			Directory.CreateDirectory(SubDir + sub2Suffix);

			CreateEmptyFile(SubDir + sub2Suffix + Slash + EmptyFileName);
			CreateNotEmptyFile(SubDir + sub2Suffix + Slash + NotEmptyFileName);

			try
			{
				IList<string> results = _runner.Run(DirOpt, BaseTestDir, RecurseOpt);
				Assert.That(results.Count, Is.EqualTo(6));
			}
			finally
			{
				Directory.Delete(SubDir + sub2Suffix, true);
			}
		}

		[Test]
		public void CalcDirIncludeExcludeFilterSamePattern()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, IncludeOpt, EmptyFileName, ExcludeOpt, EmptyFileName);
			Assert.That(results.Count, Is.EqualTo(0));
		}
		
		[Test]
		public void CalcDirIncludeFilter()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, IncludeOpt, EmptyFileName);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, EmptyFile, EmptyStringHash, 0)));
		}
		
		[Test]
		public void CalcDirIncludeFilterAll()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, IncludeOpt, EmptyFileName + ";" + NotEmptyFileName);
			Assert.That(results.Count, Is.EqualTo(2));
		}
		
		[Test]
		public void CalcDirExcludeFilter()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName);
			Assert.That(results.Count, Is.EqualTo(1));
			Assert.That(results[0], Is.EqualTo(string.Format(FileResultTpl, NotEmptyFile, HashString, 3)));
		}
		
		[Test]
		public void CalcDirExcludeFilterAll()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName + ";" + NotEmptyFileName);
			Assert.That(results.Count, Is.EqualTo(0));
		}
		
		[Test]
		public void CalcDirIncludeFilterRecursively()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, IncludeOpt, EmptyFileName, RecurseOpt);
			Assert.That(results.Count, Is.EqualTo(2));
		}
		
		[Test]
		public void CalcDirExcludeFilterRecursively()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, ExcludeOpt, EmptyFileName, RecurseOpt);
			Assert.That(results.Count, Is.EqualTo(2));
		}

		[Test]
		public void SearshFile()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, SearchOpt, HashString);
			Assert.That(results.Count, Is.EqualTo(1));
		}
		
		[Test]
		public void SearshFileRecursively()
		{
			IList<string> results = _runner.Run(DirOpt, BaseTestDir, SearchOpt, HashString, RecurseOpt);
			Assert.That(results.Count, Is.EqualTo(2));
		}
	}
}
