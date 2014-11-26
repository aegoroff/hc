using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace _tst.net
{
    public abstract class GostTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        private const string HashStringQueryTpl = "for string '{0}' do {1};";

        protected override string Executable
        {
            get { return "hc.exe"; }
        }

        string ProjectPath
        {
            get { return Environment.CurrentDirectory + @"\..\.."; }
        }

        [Fact]
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
                var results = this.Runner.Run("-C", string.Format(HashStringQueryTpl, testString, "gost"));
                Assert.Equal(1, results.Count);
                Assert.Equal(expected.ToLowerInvariant(), results[0].ToLowerInvariant());
            }
        }
    }
}