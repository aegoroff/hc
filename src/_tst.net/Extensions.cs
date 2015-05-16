/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2015 Alexander Egorov
*/

using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace _tst.net
{
    public static class Extensions
    {
        private const string EscapeSymbol = "\"";

        public static void AddParameter(this StringBuilder builder, string parameter)
        {
            if (parameter.Contains(" "))
            {
                builder.Append(EscapeSymbol);
                builder.Append(parameter);
                builder.Append(EscapeSymbol);
            }
            else
            {
                builder.Append(parameter);
            }
            builder.Append(" ");
        }

        public static IList<string> ReadLines(this StreamReader reader)
        {
            var result = new List<string>();

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (!string.IsNullOrWhiteSpace(line))
                {
                    result.Add(line);
                }
            }
            return result;
        }

        internal static string GetDirectoryName(this string path)
        {
            return Path.GetDirectoryName(Path.GetFullPath(path));
        }
    }
}