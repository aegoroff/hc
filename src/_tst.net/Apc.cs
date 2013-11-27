/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2013 Alexander Egorov
*/

using System;
using System.IO;
using NUnit.Framework;

namespace _tst.net
{
    [TestFixture]
    public class Apc32 : Apc
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\Release\{0}"; }
        }

        protected override string Arch
        {
            get { return "x86"; }
        }
    }

    [TestFixture]
    public class Apc64 : Apc
    {
        protected override string PathTemplate
        {
            get { return Environment.CurrentDirectory + @"\..\..\..\x64\Release\{0}"; }
        }

        protected override string Arch
        {
            get { return "x64"; }
        }
    }

    public abstract class Apc
    {
        protected ProcessRunner Runner { get; set; }
        

        private readonly string htpasswdPath = Path.Combine(Path.GetTempPath(), "htpasswd.txt");

        protected abstract string PathTemplate { get; }
        protected abstract string Arch { get; }

        private const string HtpasswdContent =
            @"egr1:Protected by AskApache:2eed68ccbf8405b0d6cc5a62df1edc54
egr2:{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=
egr3:$5NylHzFCY.No
egr4:$apr1$uths1zqo$4i/Rducjac63A.ExW4K6N1";

        private string Executable
        {
            get { return "apc.exe"; }
        }

        [TestFixtureSetUp]
        public void CreateFile()
        {
            File.WriteAllText(this.htpasswdPath, HtpasswdContent);
        }

        [TestFixtureTearDown]
        public void RemoveFile()
        {
            if (File.Exists(this.htpasswdPath))
            {
                File.Delete(this.htpasswdPath);
            }
        }

        [SetUp]
        public void Setup()
        {
            this.Runner = new ProcessRunner(string.Format(PathTemplate, Executable));
        }

        [Test]
        public void CrackAll()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"Login: egr1 Hash: 2eed68ccbf8405b0d6cc5a62df1edc54

Attempts: \d+ Time 00:00:0\.\d+
Nothing found

-------------------------------------------------

Login: egr2 Hash: \{SHA\}QL0AFWMIX8NRZTKeof9cXsvbvu8=

Attempts: \d+ Time 00:00:0\.\d+
Password is: 123

-------------------------------------------------

Login: egr3 Hash: \$5NylHzFCY\.No

Attempts: \d+ Time 00:00:0\.\d+
Nothing found

-------------------------------------------------

Login: egr4 Hash: \$apr1\$uths1zqo\$4i/Rducjac63A\.ExW4K6N1

Attempts: \d+ Time 00:00:0\.\d+
Password is: 123"));
        }
        
        [Test]
        public void CrackOne()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3", "-l", "egr2");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"Login: egr2 Hash: \{SHA\}QL0AFWMIX8NRZTKeof9cXsvbvu8=

Attempts: \d+ Time 00:00:0\.\d+
Password is: 123"));
        }
        
        [Test]
        public void IncompatibleOptions()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(string.Format(@"
Apache passwords cracker \d+?\.\d+?\.\d+?\.\d+? {0}
Copyright \(C\) 2009-\d+ Alexander Egorov\. All rights reserved\.

Incompatible options: impossible to crack file and hash simultaneously", Arch)));
        }
        
        [Test]
        public void CrackHashSuccess()
        {
            var results = Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"
Attempts: \d+ Time 00:00:0\.\d+
Password is: 123"));
        }
        
        [TestCase("1")]
        [TestCase("4")]
        [TestCase("1000")]
        public void CrackHashThreads(string threads)
        {
            var results = Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=", "-t", threads);
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"
Attempts: \d+ Time 00:00:0\.\d+
Password is: 123"));
        }
        
        [Test]
        public void CrackHashFailure()
        {
            var results = Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=", "-x", "2");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"
Attempts: \d+ Time 00:00:0\.\d+
Nothing found"));
        }
        
        [Test]
        public void CrackOneNoMatch()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-a", "a-z", "-x", "3", "-l", "egr2");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"Login: egr2 Hash: \{SHA\}QL0AFWMIX8NRZTKeof9cXsvbvu8=

Attempts: \d+ Time 00:00:0\.\d+
Nothing found"));
        }
        
        [Test]
        public void CrackUnexisting()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3", "-l", "egr5");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@""));
        }
        
        [Test]
        public void Listing()
        {
            var results = Runner.Run("-f", this.htpasswdPath, "-i");
            Assert.That(string.Join(Environment.NewLine, results), Is.StringMatching(@"file: .*?
 accounts:
   egr1
   egr2
   egr3
   egr4"));
        }
    }
}