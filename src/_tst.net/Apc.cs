/*
* Created by: egr
* Created at: 28.10.2007
* © 2009-2015 Alexander Egorov
*/

using System;
using System.IO;
using Xunit;

namespace _tst.net
{
    public class ApcFixture : IDisposable
    {
        private readonly string htpasswdPath = Path.Combine(Path.GetTempPath(), "htpasswd.txt");

        private const string HtpasswdContent =
    @"egr1:Protected by AskApache:2eed68ccbf8405b0d6cc5a62df1edc54
egr2:{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=
egr3:$5NylHzFCY.No
egr4:$apr1$uths1zqo$4i/Rducjac63A.ExW4K6N1";
        
        public ApcFixture()
        {
            File.WriteAllText(this.HtpasswdPath, HtpasswdContent);
        }

        public string HtpasswdPath
        {
            get { return this.htpasswdPath; }
        }

        public void Dispose()
        {
            if (File.Exists(this.HtpasswdPath))
            {
                File.Delete(this.HtpasswdPath);
            }
        }
    }

    public abstract class Apc<T> : ExeWrapper<T>, IClassFixture<ApcFixture> where T : Architecture, new()
    {
        private readonly string htpasswdPath;

        protected Apc() : base(new T())
        {
            this.htpasswdPath = new ApcFixture().HtpasswdPath;
        }

        protected override string Executable
        {
            get { return "apc.exe"; }
        }

        [Fact]
        public void CrackAll()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"Login: egr1 Hash: 2eed68ccbf8405b0d6cc5a62df1edc54

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
Password is: 123");
        }
        
        [Fact]
        public void CrackOne()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3", "-l", "egr2");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"Login: egr2 Hash: \{SHA\}QL0AFWMIX8NRZTKeof9cXsvbvu8=

Attempts: \d+ Time 00:00:0\.\d+
Password is: 123");
        }
        
        [Fact]
        public void IncompatibleOptions()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), string.Format(@"
Apache passwords cracker \d+?\.\d+?\.\d+?\.\d+? {0}
Copyright \(C\) 2009-\d+ Alexander Egorov\. All rights reserved\.

Incompatible options: impossible to crack file and hash simultaneously", Arch));
        }
        
        [Fact]
        public void CrackHashSuccess()
        {
            var results = this.Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"
Attempts: \d+ Time 00:00:0\.\d+
Password is: 123");
        }
        
        [Theory]
        [InlineData("1")]
        [InlineData("4")]
        [InlineData("1000")]
        public void CrackHashThreads(string threads)
        {
            var results = this.Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=", "-t", threads);
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"
Attempts: \d+ Time 00:00:0\.\d+
Password is: 123");
        }
        
        [Fact]
        public void CrackHashFailure()
        {
            var results = this.Runner.Run("-h", "{SHA}QL0AFWMIX8NRZTKeof9cXsvbvu8=", "-x", "2");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"
Attempts: \d+ Time 00:00:0\.\d+
Nothing found");
        }
        
        [Fact]
        public void CrackOneNoMatch()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-a", "a-z", "-x", "3", "-l", "egr2");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"Login: egr2 Hash: \{SHA\}QL0AFWMIX8NRZTKeof9cXsvbvu8=

Attempts: \d+ Time 00:00:0\.\d+
Nothing found");
        }
        
        [Fact]
        public void CrackUnexisting()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-a", "0-9", "-x", "3", "-l", "egr5");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"");
        }
        
        [Fact]
        public void Listing()
        {
            var results = this.Runner.Run("-f", this.htpasswdPath, "-i");
            Asserts.StringMatching(string.Join(Environment.NewLine, results), @"file: .*?
 accounts:
   egr1
   egr2
   egr3
   egr4");
        }
    }
}