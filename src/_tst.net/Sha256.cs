/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Sha256 : THashCalculator
	{
		protected override string Executable
		{
			get { return "sha256.exe"; }
		}

		protected override string HashString
		{
			get { return "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3"; }
		}

		protected override string EmptyStringHash
		{
			get { return "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"; }
		}
	}
}