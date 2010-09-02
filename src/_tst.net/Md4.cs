/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Md4 : THashCalculator
	{
		protected override string Executable
		{
			get { return "md4.exe"; }
		}

		protected override string HashString
		{
			get { return "C58CDA49F00748A3BC0FCFA511D516CB"; }
		}

		protected override string EmptyStringHash
		{
			get { return "31D6CFE0D16AE931B73C59D7E0C089C0"; }
		}
	}
}