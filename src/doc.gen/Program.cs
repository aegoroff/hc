/*
 * Created by: egr
 * Created at: 19.10.2010
 * © 2007-2011 Alexander Egorov
 */

using System;
using System.Collections.ObjectModel;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using Antlr3.ST;

namespace doc.gen
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                return;
            }
            string docPath = args[0];

            Configuration configuration =
                (Configuration) ConfigurationManager.GetSection(Configuration.RootElement);

            foreach (Calculator calculator in configuration.Calculators)
            {
                CreateDocumentationTxt(docPath, @"Readme.ru.st", calculator, "ru");
                CreateDocumentationTxt(docPath, @"Readme.en.st", calculator, "en");
            }
            CreateApcDocumentationTxt(docPath, @"Readme.Htpwdc.ru.st", "ru");
            CreateApcDocumentationTxt(docPath, @"Readme.Htpwdc.en.st", "en");
        }

        private static void CreateDocumentationTxt(string docPath, string template, Calculator calculator, string lang)
        {
            StringTemplate stringTemplate = new StringTemplate(File.ReadAllText(Path.Combine(docPath, template)));
            stringTemplate.SetAttribute("hashName", calculator.HashName);
            stringTemplate.SetAttribute("mOpt", calculator.MOpt);
            stringTemplate.SetAttribute("appName", calculator.AppName);
            stringTemplate.SetAttribute("hashOf123", calculator.HashOf123);
            stringTemplate.SetAttribute("hashOfFile", calculator.HashOfFile);
            stringTemplate.SetAttribute("spaceCountMOpt", new string(' ', calculator.SpaceCountMOpt));
            stringTemplate.SetAttribute("spaceCountSearchOpt", new string(' ', calculator.SpaceCountSearchOpt));
            File.WriteAllText(Path.Combine(docPath, @"Readme." + calculator.AppName + "." + lang + ".txt"),
                              stringTemplate.ToString(), Encoding.UTF8);
        }
        
        private static void CreateApcDocumentationTxt(string docPath, string template, string lang)
        {
            StringTemplate stringTemplate = new StringTemplate(File.ReadAllText(Path.Combine(docPath, template)));
            stringTemplate.SetAttribute("exeName", "apc.exe");
            stringTemplate.SetAttribute("appName", "Apache passwords cracker");
            File.WriteAllText(Path.Combine(docPath, @"Readme.apc." + lang + ".txt"),
                              stringTemplate.ToString(), Encoding.UTF8);
        }
    }

    [Serializable]
    public struct Calculator
    {
        [XmlElement(ElementName = "HashName")] public string HashName;

        [XmlElement(ElementName = "MOpt")] public string MOpt;

        [XmlElement(ElementName = "AppName")] public string AppName;

        [XmlElement(ElementName = "HashOf123")] public string HashOf123;

        [XmlElement(ElementName = "HashOfFile")] public string HashOfFile;

        [XmlElement(ElementName = "SpaceCountMOpt")] public int SpaceCountMOpt;

        [XmlElement(ElementName = "SpaceCountSearchOpt")] public int SpaceCountSearchOpt;
    }

    [Serializable]
    [XmlRoot(RootElement)]
    public class Configuration : IConfigurationSectionHandler
    {
        private readonly Collection<Calculator> calculators = new Collection<Calculator>();
        public const string RootElement = "Configuration";

        /// <summary>
        /// Attributes set
        /// </summary>
        [XmlElement(ElementName = "Calculator")]
        public Collection<Calculator> Calculators
        {
            get { return calculators; }
        }

        public object Create(object parent, object configContext, XmlNode section)
        {
            return Create(section);
        }


        private static object Create(XmlNode section)
        {
            XmlSerializer serializer = new XmlSerializer(typeof (Configuration));
            XmlNodeList confList = section.SelectNodes("//" + RootElement);
            if (confList == null || confList.Count < 1)
            {
                throw new Exception("Configuration error");
            }
            XmlNodeReader rdr = new XmlNodeReader(confList[0]);
            return serializer.Deserialize(rdr);
        }

        /// <summary>
        /// Creates new configuration object using XML configuration file
        /// </summary>
        /// <param name="filePath">Path to configuration file</param>
        /// <returns><see cref="Configuration"/> object created using XML file</returns>
        public static Configuration Create(string filePath)
        {
            XmlDocument d;
            try
            {
                d = new XmlDocument();
                d.Load(filePath);
            }
            catch (Exception e)
            {
                Trace.WriteLine(e.Message);
                return null;
            }
            return (Configuration) Create(d);
        }
    }
}