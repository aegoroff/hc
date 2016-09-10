/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2016 Alexander Egorov
 */

using System;
using System.IO;
using System.Text.RegularExpressions;
using Xunit;

namespace _tst.net
{
    public abstract class GostTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        protected GostTests() : base(new T())
        {
        }

        protected override string Executable => "hc.exe";

        string ProjectPath => Environment.CurrentDirectory + @"\..\..";

        [Fact]
        public void Test()
        {
            var testVectorsPath = Path.Combine(this.ProjectPath, "gost_tv_cryptopro.txt");
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
                if (string.IsNullOrWhiteSpace(testString))
                {
                    testString = "\"\"";
                }
                var results = this.Runner.Run("gost", "-s", testString);
                
                Assert.Equal(expected.ToLowerInvariant(), results[0].ToLowerInvariant());
                Assert.Equal(1, results.Count);
            }
        }
    }
}