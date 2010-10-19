/*
 * Created by: egr
 * Created at: 19.10.2010
 * © 2007-2010 Alexander Egorov
 */

using System.IO;
using System.Text;
using Antlr3.ST;

namespace doc.gen
{
    class Program
    {
        static void Main(string[] args)
        {
            string docPath = @"C:\hg\hc\docs\";
            string templateRu = @"Readme.ru.st";

            

            Calculator whirlpool = new Calculator
                                       {
                                           HashName = "WHIRLPOOL",
                                           MOpt = "whirlpool",
                                           AppName = "whirlpool",
                                           HashOf123 =
                                               "BD801451D24470DF899173BFF3C04E875BE46C97D1529F84269C70C26C0F7D31D1AD21CBD985E7CFD7E1496B3BA5905789BC0790817DA26DC36D7ECA14B689D7",
                                           HashOfFile =
                                               "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F"
                                       };
            Calculator sha512 = new Calculator
                                       {
                                           HashName = "SHA512",
                                           MOpt = "sha512",
                                           AppName = "sha512",
                                           HashOf123 =
                                               "6F6C7ED600C5E27023D63AF4F3943DDEF0309FE4CF2F6C4630985F06639FCDE93AB55EE9821D576C625A99AD62A0E3E9CC2396622B271BA8D94BC29866F46923",
                                           HashOfFile =
                                               "3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2"
                                       };

            CreateDocumentationTxt(docPath, templateRu, whirlpool, "whirlpool");
            CreateDocumentationTxt(docPath, templateRu, sha512, "sha512");
        }

        private static void CreateDocumentationTxt(string docPath, string template, Calculator calculator, string output)
        {
            StringTemplate stringTemplate = new StringTemplate(File.ReadAllText(docPath + template));
            stringTemplate.SetAttribute("hashName", calculator.HashName);
            stringTemplate.SetAttribute("mOpt", calculator.MOpt);
            stringTemplate.SetAttribute("appName", calculator.AppName);
            stringTemplate.SetAttribute("hashOf123", calculator.HashOf123);
            stringTemplate.SetAttribute("hashOfFile", calculator.HashOfFile);
            File.WriteAllText(docPath + @"Readme." + output + ".ru.txt", stringTemplate.ToString(), Encoding.UTF8);
        }
    }

    public struct Calculator
    {
        internal string HashName;
        internal string MOpt;
        internal string AppName;
        internal string HashOf123;
        internal string HashOfFile;
    }
}
