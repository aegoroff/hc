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
            if ( args.Length == 0 )
            {
                return;
            }
            string docPath = args[0];
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
            Calculator sha384 = new Calculator
                                       {
                                           HashName = "SHA384",
                                           MOpt = "sha384",
                                           AppName = "sha384",
                                           HashOf123 =
                                               "AFE0F32AFCA5A9A8422A82FAFB369C14342791EC780D8825465D3B8960A6EA6575EFF9DC5A7C8C563EC39E043E76CCC5",
                                           HashOfFile =
                                               "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F"
                                       };
            Calculator sha256 = new Calculator
                                       {
                                           HashName = "SHA256",
                                           MOpt = "sha256",
                                           AppName = "sha256",
                                           HashOf123 =
                                               "0A3B10B4A34A250A87B47D538333F4B06589171C7DFEEE26FF84CC82BAC874FB",
                                           HashOfFile =
                                               "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3"
                                       };
            Calculator sha1 = new Calculator
                                       {
                                           HashName = "SHA1",
                                           MOpt = "sha1",
                                           AppName = "sha1",
                                           HashOf123 =
                                               "274F856438363F4032C8B87CF6BF49CEB9B5AC3C",
                                           HashOfFile =
                                               "40BD001563085FC35165329EA1FF5C5ECBDBBEEF"
                                       };
            Calculator md5 = new Calculator
                                       {
                                           HashName = "MD5",
                                           MOpt = "md5",
                                           AppName = "md5",
                                           HashOf123 =
                                               "E0C110627FA4B42189C8DFD717957537",
                                           HashOfFile =
                                               "202CB962AC59075B964B07152D234B70"
                                       };
            Calculator md4 = new Calculator
                                       {
                                           HashName = "MD4",
                                           MOpt = "md4",
                                           AppName = "md4",
                                           HashOf123 =
                                               "3689CA24BF71B39B6612549D87DCEA68",
                                           HashOfFile =
                                               "C58CDA49F00748A3BC0FCFA511D516CB"
                                       };

            CreateDocumentationTxt(docPath, templateRu, whirlpool, "ru");
            CreateDocumentationTxt(docPath, templateRu, sha512, "ru");
            CreateDocumentationTxt(docPath, templateRu, sha384, "ru");
            CreateDocumentationTxt(docPath, templateRu, sha256, "ru");
            CreateDocumentationTxt(docPath, templateRu, sha1, "ru");
            CreateDocumentationTxt(docPath, templateRu, md5, "ru");
            CreateDocumentationTxt(docPath, templateRu, md4, "ru");
        }

        private static void CreateDocumentationTxt(string docPath, string template, Calculator calculator, string lang)
        {
            StringTemplate stringTemplate = new StringTemplate(File.ReadAllText(Path.Combine(docPath, template)));
            stringTemplate.SetAttribute("hashName", calculator.HashName);
            stringTemplate.SetAttribute("mOpt", calculator.MOpt);
            stringTemplate.SetAttribute("appName", calculator.AppName);
            stringTemplate.SetAttribute("hashOf123", calculator.HashOf123);
            stringTemplate.SetAttribute("hashOfFile", calculator.HashOfFile);
            File.WriteAllText(docPath + @"Readme." + calculator.AppName + "." + lang +".txt", stringTemplate.ToString(), Encoding.UTF8);
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
