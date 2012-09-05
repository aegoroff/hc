/*
 * Created by: egr
 * Created at: 05.09.2012
 * © 2012 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Xml;
using TridInformer;

namespace trid
{
	internal class Program
	{
		private static void Main(string[] args)
		{
			if (args.Length < 1)
			{
				return;
			}
			string separator = "\t";

		    if (args.Length == 2)
			{
				switch (args[1])
				{
					case "-t":
						separator = "\t";
						break;
					case "-c":
						separator = ",";
						break;
					default:
						separator = "\t";
						break;
				}
			}

		    string[] paths = Directory.GetFiles(args[0], "*.trid.xml");
            string hqlPath = Path.Combine(args[0], "hql");
			int maxLength = 0;
			string maxType = string.Empty;
			var uniques = new Dictionary<string, string>(paths.Length);
			var duplicates = new Dictionary<string, string>();
            CreateFilteredDirectory(hqlPath);
			foreach (string path in paths)
			{
				string file = Path.GetFileName(path);
				var doc = new XmlDocument();
				doc.Load(path);

				if (doc.DocumentElement == null)
				{
					return;
				}

				var signature = new Signature {Separator = separator, File = file};

			    foreach (XmlNode node in doc.DocumentElement.ChildNodes)
				{
					if (node.Name == "Info")
					{
						// FileType
						signature.Type = node.ChildNodes[0].ChildNodes[0].Value;
						// Ext
						if (node.ChildNodes[1].ChildNodes[0] != null)
						{
							signature.Extension = node.ChildNodes[1].ChildNodes[0].Value;
						}

						maxLength = Math.Max(signature.Type.Length, maxLength);
						if (signature.Type.Length == maxLength)
						{
							maxType = signature.Type;
						}
						if (!uniques.ContainsKey(signature.Type))
						{
							uniques.Add(signature.Type, file);
						}
						else
						{
                            if (!duplicates.ContainsKey(uniques[signature.Type]))
						    {
						        duplicates.Add(uniques[signature.Type], signature.Type);
						    }
							duplicates.Add(file, signature.Type);
						}
						if (node.ChildNodes[2].Name == "ExtraInfo")
						{
							if (node.ChildNodes[2].ChildNodes[0].ChildNodes[0] != null)
							{
								signature.Description = node.ChildNodes[2].ChildNodes[0].ChildNodes[0].Value;
							}
							if (node.ChildNodes[2].ChildNodes[1].ChildNodes[0] != null)
							{
								signature.Url = node.ChildNodes[2].ChildNodes[1].ChildNodes[0].Value;
							}
						}
					}
					if (node.Name == "General")
					{
						// FileNum
						string filesNumber = node.ChildNodes[0].ChildNodes[0].Value;
						
						Int32.TryParse(filesNumber, NumberStyles.Integer, CultureInfo.InvariantCulture, out signature.ExamplesCount);
					}

                    if (node.Name == "FrontBlock")
                    {
                        var query = new StringBuilder();
                        query.AppendFormat("let fileName = 'hq.{0}';", signature.Extension);
                        query.AppendLine();
                        
                        foreach (XmlNode pattern in node.ChildNodes)
                        {
                            int limit = -1;
                            int offset = -1;
                            foreach (XmlNode childNode in pattern.ChildNodes)
                            {
                                if (childNode.Name == "Bytes")
                                {
                                    limit = childNode.ChildNodes[0].Value.Length / 2;
                                }
                                if (childNode.Name == "Pos")
                                {
                                    offset = int.Parse(childNode.ChildNodes[0].Value);
                                }
                            }
                            if (limit >= 0 && offset >= 0)
                            {
                                query.AppendFormat("for file f from fileName let f.offset = {0}, f.limit = {1} do md5;", offset, limit);
                                query.AppendLine();
                                var hqlFilePath = Path.Combine(hqlPath, signature.File + ".hql");
                                File.WriteAllText(hqlFilePath, query.ToString());
                            }
                        }
                    }
				}

				Console.WriteLine(signature.ToString());
			}
			Console.WriteLine("\n\nMax type name: {0}\nLength: {1}", maxType, maxLength);

			if (duplicates.Count > 0)
			{
				Console.WriteLine("\n\nDuplicates:\n");
			}
			foreach (KeyValuePair<string, string> pair in duplicates)
			{
				Console.WriteLine("{0}{1}{2}", pair.Key, separator, pair.Value);
			}
		}

		private static void CreateFilteredDirectory(string filteredDirPath)
		{
			if (!Directory.Exists(filteredDirPath))
			{
				Directory.CreateDirectory(filteredDirPath);
			}
			else
			{
				Directory.Delete(filteredDirPath, true);
				Directory.CreateDirectory(filteredDirPath);
			}
		}
	}
}