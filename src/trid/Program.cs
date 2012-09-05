/*
 * Created by: egr
 * Created at: 05.09.2012
 * © 2012 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Xml;
using TridInformer;

namespace trid
{
	internal class Program
	{
		private const int ExamplesThreshold = 20;
		private const int WeightThreshold = 10;
		private const string FilteredExamplesDir = "filtered";
		private const string FilteredExamplesByCategoryDir = "filteredByCategory";

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
			var categoryMap = new Dictionary<string, string>();
			if (args.Length == 3)
			{
			    string[] categoryFiles = Directory.GetFiles(args[2], "*Category.txt");
			    foreach (string file in categoryFiles)
				{
					string[] lines = File.ReadAllLines(file);
					foreach (string line in lines)
					{
						string[] columns = line.Split('\t');
						if (columns.Length < 2)
						{
							continue;
						}
						string categoryFile = Path.GetFileNameWithoutExtension(file);
						categoryFile = categoryFile.Replace("Category", string.Empty);
						string t = columns[1].Trim().ToUpperInvariant();
						if (categoryMap.ContainsKey(t))
						{
							continue;
						}
						categoryMap.Add(t, categoryFile);
					}
				}
			}

		    string[] paths = Directory.GetFiles(args[0], "*.trid.xml");
			string filteredDirPath = Path.Combine(args[0], FilteredExamplesDir);
            string filteredByCategoryDirPath = Path.Combine(args[0], FilteredExamplesByCategoryDir);
			int maxLength = 0;
			string maxType = string.Empty;
			var uniques = new Dictionary<string, string>(paths.Length);
			var duplicates = new Dictionary<string, string>();
			CreateFilteredDirectory(filteredDirPath);
			CreateFilteredDirectory(filteredByCategoryDirPath);
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
							duplicates.Add(uniques[signature.Type], signature.Type);
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
						foreach (XmlNode pattern in node.ChildNodes)
						{
							if (pattern.ChildNodes[0].Name == "Bytes")
							{
								signature.Weight += pattern.ChildNodes[0].ChildNodes[0].Value.Length / 2;
							}
						}
					}
					if (node.Name == "GlobalStrings")
					{
						foreach (XmlNode pattern in node.ChildNodes)
						{
							if (pattern.Name == "String")
							{
								signature.Weight += pattern.ChildNodes[0].Value.Length * 500;
							}
						}
					}
				}
				if ((signature.ExamplesCount >= ExamplesThreshold && signature.Weight >= WeightThreshold) || signature.Weight >= WeightThreshold)
				{
					File.Copy(path, filteredDirPath + "\\" + file);
				}
				if (categoryMap.Count > 0)
				{
					string t = signature.Type.Trim().ToUpperInvariant();
					if (categoryMap.ContainsKey(t))
					{
						string categoryDir = filteredByCategoryDirPath + @"\" + categoryMap[t];
						if (!Directory.Exists(categoryDir))
						{
							Directory.CreateDirectory(categoryDir);
						}
						File.Copy(path, categoryDir + @"\" + file);
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