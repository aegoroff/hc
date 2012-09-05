/*
 * Created by: Egorov Alexander
 * Created at: 05.03.2010
 * Copyright: (c) InfoWatch 2004-2010
 */

namespace TridInformer
{
	internal struct Signature
	{
		internal string Separator;
		internal string Type;
		internal string File;
		internal string Extension;
		internal string Description;
		internal string Url;
		internal int Weight;
		internal int ExamplesCount;

		public override string ToString()
		{
			return string.Format("{0}{1}{2}{1}{5}{1}{4}{1}{3}{1}{6}{1}{7}", File, Separator, Type, Description, ExamplesCount, Extension, Weight, Url);
		}
	}
}