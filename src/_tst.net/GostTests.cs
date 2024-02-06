/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2024 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using FluentAssertions;
using Xunit;

namespace _tst.net
{
    public abstract class GostTests<T> : ExeWrapper<T>
            where T : Architecture, new()
    {
        protected GostTests() : base(new T())
        {
        }

        protected override string Executable => "hc.exe";

        private static string ProjectPath => Path.Combine(Environment.CurrentDirectory, "..", "..");

        [Theory, MemberData(nameof(GostData))]
        public void CalcString_GostTestHashes_Success(string testString, string expected)
        {
            // Arrange
            var expectation = expected.ToLowerInvariant();

            // Act
            var results = this.Runner.Run("gost", "string", "-s", testString);

            // Assert
            results[0].ToLowerInvariant().Should().Be(expectation);
            results.Should().HaveCount(1);
        }

        public static IEnumerable<object[]> GostData
        {
            get
            {
                var testVectorsPath = Path.Combine(ProjectPath, "gost_tv_cryptopro.txt");
                var vectors = File.ReadAllLines(testVectorsPath);
                foreach (var vector in vectors)
                {
                    var parts = vector.Split(new[] { '=' }, StringSplitOptions.RemoveEmptyEntries);
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

                    yield return new object[] { testString, expected };
                }
            }
        }
    }
}
