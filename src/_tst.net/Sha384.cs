/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Sha384 : THashCalculator
	{
		protected override string Executable
		{
			get { return "sha384.exe"; }
		}

		protected override string HashString
		{
			get { return "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F"; }
		}

		protected override string EmptyStringHash
		{
			get { return "38B060A751AC96384CD9327EB1B1E36A21FDB71114BE07434C0CC7BF63F6E1DA274EDEBFE76F65FBD51AD2F14898B95B"; }
		}
	}
}