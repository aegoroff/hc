/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2007-2010 Alexander Egorov
 */

namespace _tst.net
{
	public abstract class Hash
	{
		public abstract string Executable { get; }

		public abstract string HashString { get; }

		public abstract string EmptyStringHash { get; }

		public virtual string InitialString
		{
			get { return "123"; }
		}
	}

	public class Whirlpool : Hash
	{
		public override string Executable
		{
			get { return "whirlpool.exe"; }
		}

		public override string HashString
		{
			get
			{
				return
					"344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F";
			}
		}

		public override string EmptyStringHash
		{
			get
			{
				return
					"19FA61D75522A4669B44E39C1D2E1726C530232130D407F89AFEE0964997F7A73E83BE698B288FEBCF88E3E03C4F0757EA8964E59B63D93708B138CC42A66EB3";
			}
		}
	}

	public class Sha512 : Hash
	{
		public override string Executable
		{
			get { return "sha512.exe"; }
		}

		public override string HashString
		{
			get
			{
				return
					"3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2";
			}
		}

		public override string EmptyStringHash
		{
			get
			{
				return
					"CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE47D0D13C5D85F2B0FF8318D2877EEC2F63B931BD47417A81A538327AF927DA3E";
			}
		}
	}

	public class Sha384 : Hash
	{
		public override string Executable
		{
			get { return "sha384.exe"; }
		}

		public override string HashString
		{
			get { return "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F"; }
		}

		public override string EmptyStringHash
		{
			get { return "38B060A751AC96384CD9327EB1B1E36A21FDB71114BE07434C0CC7BF63F6E1DA274EDEBFE76F65FBD51AD2F14898B95B"; }
		}
	}

	public class Sha256 : Hash
	{
		public override string Executable
		{
			get { return "sha256.exe"; }
		}

		public override string HashString
		{
			get { return "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3"; }
		}

		public override string EmptyStringHash
		{
			get { return "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"; }
		}
	}

	public class Sha1 : Hash
	{
		public override string Executable
		{
			get { return "sha1.exe"; }
		}

		public override string HashString
		{
			get { return "40BD001563085FC35165329EA1FF5C5ECBDBBEEF"; }
		}

		public override string EmptyStringHash
		{
			get { return "DA39A3EE5E6B4B0D3255BFEF95601890AFD80709"; }
		}
	}

	public class Md5 : Hash
	{
		public override string Executable
		{
			get { return "md5.exe"; }
		}

		public override string HashString
		{
			get { return "202CB962AC59075B964B07152D234B70"; }
		}

		public override string EmptyStringHash
		{
			get { return "D41D8CD98F00B204E9800998ECF8427E"; }
		}
	}

	public class Md4 : Hash
	{
		public override string Executable
		{
			get { return "md4.exe"; }
		}

		public override string HashString
		{
			get { return "C58CDA49F00748A3BC0FCFA511D516CB"; }
		}

		public override string EmptyStringHash
		{
			get { return "31D6CFE0D16AE931B73C59D7E0C089C0"; }
		}
	}
}