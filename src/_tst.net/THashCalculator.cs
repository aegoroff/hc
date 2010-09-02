/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using NUnit.Framework;

namespace _tst.net
{
	public abstract class THashCalculator
	{
		private static readonly string PathTemplate = Environment.CurrentDirectory + @"\..\..\..\Release\{0}";
		private const string CrackTemplate = "-c -m {0}";
		private const string CalcStringTemplate = "-s {0}";
		private const string EmptyTemplate = "\"{0}\"";
		private const string RestoredStringTemplate = "Initial string is: {0}";
		private const string LowCaseFlag = "-l";
		const string NothingFound = "Nothing found";

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
	}
}
