/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2014 Alexander Egorov
*/

using System.Text.RegularExpressions;
using Xunit;

namespace _tst.net
{
    public static class Asserts
    {
        public static void StringMatching(string actual, string expected)
        {
            Assert.True(Regex.IsMatch(actual, expected), string.Format("String {0} doesn's match pattern {1}", actual, expected));
        }
        
        public static void StringNotMatching(string actual, string expected)
        {
            Assert.False(Regex.IsMatch(actual, expected), string.Format("String {0} is match pattern {1} but it shouldn't", actual, expected));
        }
    }
}