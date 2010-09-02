/*
 * Created by: egr
 * Created at: 02.09.2010
 * © 2007-2010 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
	[TestFixture]
	public class Whirlpool : THashCalculator
	{
		protected override string Executable
		{
			get { return "whirlpool.exe"; }
		}

		protected override string HashString
		{
			get { return "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F"; }
		}

		protected override string EmptyStringHash
		{
			get { return "19FA61D75522A4669B44E39C1D2E1726C530232130D407F89AFEE0964997F7A73E83BE698B288FEBCF88E3E03C4F0757EA8964E59B63D93708B138CC42A66EB3"; }
		}
	}
}