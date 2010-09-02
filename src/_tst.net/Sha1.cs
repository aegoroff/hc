/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Sha1 : THashCalculator
	{
		protected override string Executable
		{
			get { return "sha1.exe"; }
		}

		protected override string HashString
		{
			get { return "40BD001563085FC35165329EA1FF5C5ECBDBBEEF"; }
		}

		protected override string EmptyStringHash
		{
			get { return "DA39A3EE5E6B4B0D3255BFEF95601890AFD80709"; }
		}
	}
}