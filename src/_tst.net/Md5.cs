/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Md5 : THashCalculator
	{
		protected override string Executable
		{
			get { return "md5.exe"; }
		}

		protected override string HashString
		{
			get { return "202CB962AC59075B964B07152D234B70"; }
		}

		protected override string EmptyStringHash
		{
			get { return "D41D8CD98F00B204E9800998ECF8427E"; }
		}
	}
}