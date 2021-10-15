/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2021 Alexander Egorov
*/

using System.IO;
using System.Text;

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

        internal static string GetDirectoryName(this string path)
        {
            return Path.GetDirectoryName(Path.GetFullPath(path));
        }
    }
}
