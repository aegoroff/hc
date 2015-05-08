/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2015 Alexander Egorov
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace _tst.net
{
    public static class Asserts
    {
        public static void StringMatching(string actual, string expected)
        {
            Assert.True(Regex.IsMatch(actual, expected), string.Format("String:\n\n {0} \n\ndoesn's match pattern:\n\n {1}\n\n", actual, expected));
        }
        
        public static void StringNotMatching(string actual, string expected)
        {
            Assert.False(Regex.IsMatch(actual, expected), string.Format("String:\n\n {0} \n\nis match pattern:\n\n {1} \n\nbut it shouldn't\n\n", actual, expected));
        }

        public static string Normalize(this string expectation)
        {
            var parts = expectation.Split('\n');
            return string.Join(Environment.NewLine, parts.Select(s => s.Trim()));
        }
        
        public static string Normalize(this IList<string> actual)
        {
            return string.Join(Environment.NewLine, actual.Select(s => s.Trim()));
        }
    }
}