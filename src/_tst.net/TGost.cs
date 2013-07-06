using System;
using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;

namespace _tst.net
{
    [TestFixture]
    public class TGost32 : TGost
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }

        protected override string Arch
        {
            get { return "x86"; }
        }
    }

    [TestFixture]
    public class TGost64 : TGost
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\x64\Release\{0}"; }
        }

        protected override string Arch
        {
            get { return "x64"; }
        }
    }
    
    public abstract class TGost
    {
        protected ProcessRunner Runner { get; set; }

        protected abstract string PathTemplate { get; }
        protected abstract string Arch { get; }
        private const string HashStringQueryTpl = "for string '{0}' do {1};";

        private string Executable
        {
            get { return "hq.exe"; }
        }

        string ProjectPath
        {
            get { return Environment.CurrentDirectory + @"\..\.."; }
        }

        [SetUp]
        public void Setup()
        {
            this.Runner = new ProcessRunner(string.Format(PathTemplate, Executable));
        }

        [Test]
        public void Test()
        {
            var testVectorsPath = Path.Combine(ProjectPath, "gost_tv_cryptopro.txt");
            var vectors = File.ReadAllLines(testVectorsPath);
            foreach (var vector in vectors)
            {
                var parts = vector.Split(new[] {'='}, StringSplitOptions.RemoveEmptyEntries);
                var expected = parts[1].Trim();
                var testData = parts[0].Trim();
                var str = new Regex(@"^GOST\(""(.*?)""\)$");
                var match = str.Match(testData);
                if (!match.Success)
                {
                    continue;
                }
                var testString = match.Groups[1].Value.Trim('"');
                var results = Runner.Run("-C", string.Format(HashStringQueryTpl, testString, "gost"));
                Assert.That(results.Count, Is.EqualTo(1));
                Assert.That(results[0].ToLowerInvariant(), Is.EqualTo(expected.ToLowerInvariant()));
            }
        }
    }
}