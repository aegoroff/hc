/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2015 Alexander Egorov
*/

using System.Text.RegularExpressions;
using Xunit;

namespace _tst.net
{
    public static class Asserts
    {
        public static void StringMatching(string actual, string expected)
        {
            Assert.True(Regex.IsMatch(actual, expected), string.Format("String:\n {0} \ndoesn's match pattern:\n {1}", actual, expected));
        }
        
        public static void StringNotMatching(string actual, string expected)
        {
            Assert.False(Regex.IsMatch(actual, expected), string.Format("String:\n {0} \nis match pattern:\n {1} \nbut it shouldn't", actual, expected));
        }
    }
}