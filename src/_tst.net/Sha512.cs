/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Sha512 : THashCalculator
	{
		protected override string Executable
		{
			get { return "sha512.exe"; }
		}

		protected override string HashString
		{
			get { return "3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2"; }
		}

		protected override string EmptyStringHash
		{
			get { return "CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE47D0D13C5D85F2B0FF8318D2877EEC2F63B931BD47417A81A538327AF927DA3E"; }
		}
	}
}