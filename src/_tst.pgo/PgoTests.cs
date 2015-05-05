using System.Collections.Generic;
using System.IO;
using Xunit;
using Xunit.Abstractions;
using _tst.net;

namespace _tst.pgo
{
    public class PgoTests64 : PgoTests<ArchWin64>
    {
        public PgoTests64(ITestOutputHelper output) : base(output)
        {
        }
    }

    public class PgoTests32 : PgoTests<ArchWin32>
    {
        public PgoTests32(ITestOutputHelper output) : base(output)
        {
        }
    }

    [Collection("SerializableTests")]
    public abstract class PgoTests<T> : ExeWrapper<T> where T : Architecture, new()
    {
        private const string CrackOpt = "-c";
        private const string HashOpt = "-m";
        private const string MaxOpt = "-x";
        private const string MinOpt = "-n";
        private const string NoProbeOpt = "--noprobe";
        private const string DirOpt = "-d";
        private const string IncludeOpt = "-i";
        private const string QueryOpt = "-C";
        private const string FileContent = @"# Comment
# Comment
for file f from dir 'c:' where f.size == 0 do find;
# Comment
for file f from dir 'c:' where f.size == 0 do find;
for file f from dir 'c:' where f.name == 'test' do find;
for string '123' do md5;
for string s from hash '202CB962AC59075B964B07152D234B70' let s.max = 5 do crack md5;
for string s from hash '202CB962AC59075B964B07152D234B70' let s.min = 3 do crack md5;
for string s from hash '202CB962AC59075B964B07152D234B70' let s.dict = '0-9' do crack md5;
for string s from hash '202CB962AC59075B964B07152D234B70' let s.max = 5, s.dict = '0-9', s.min = 3 do crack md5;
for string s from hash 'D41D8CD98F00B204E9800998ECF8427E' let s.min = 4 do crack md5;
for file f from dir 'c:' where f.size == 0 do find;
for file f from dir 'c:' where f.size == 0 and f.name ~ filemask do find;
for file f from dir 'c:' where f.size == 0 or f.name ~ '*.exe' do find;
for file f from dir 'c:' where f.size == 0 and (f.name !~ '*.exe' or f.path ~ 'c:\temp\*') do find;
for file f from '1' do md5;
for file f from '1' let f.limit = 10 do md5;
for file f from '1' let f.md5 = 'D41D8CD98F00B204E9800998ECF8427E' do validate;
for file f from parameter where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E' do validate;
for file f from dir 'c:' where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E' do find;
for file f from dir '.' where f.md5 == 'D41D8CD98F00B204E9800998ECF84271' and f.limit == 100 and f.offset == 10 do find;
for file f from dir '.' where f.size < 0 and f.md5 == 'D41D8CD98F00B204E9800998ECF84271' do find;
";

        private readonly ITestOutputHelper output;
        
        protected PgoTests(ITestOutputHelper output) : base(new T())
        {
            this.output = output;
        }

        [Theory, MemberData("Hashes")]
        public void Cases(Hash h)
        {
            var results = this.Runner.Run(h.Algorithm, CrackOpt, NoProbeOpt, HashOpt, h.HashString, MaxOpt, "3", MinOpt, "2");
            this.WriteResults(results);
            results = this.Runner.Run(h.Algorithm, NoProbeOpt, DirOpt, ".", IncludeOpt, "*.exe");
            this.WriteResults(results);
            results = this.Runner.Run(
                QueryOpt,
                string.Format("let filemask = '.*exe$'; for file f from dir '.'  where f.{0} == '{1}' and f.size > 20 and f.name ~ filemask do find;", h.Algorithm, h.HashString), 
                NoProbeOpt);
            this.WriteResults(results);
        }

        [Fact]
        public void ParseBigFile()
        {
            const string temp = "t.hlq";
            try
            {
                for (var i = 0; i < 1000; i++)
                {
                    File.AppendAllText(temp, FileContent);
                }
                for (var i = 0; i < 5; i++)
                {
                    var results = this.Runner.Run("-S", NoProbeOpt, "-F", Path.GetFullPath(temp));
                    this.WriteResults(results);
                }
            }
            finally
            {
                if (File.Exists(temp))
                {
                    File.Delete(temp);
                }
            }
        }

        private void WriteResults(IEnumerable<string> results)
        {
            foreach (var result in results)
            {
                this.output.WriteLine(result);
            }
        }

        protected override string Executable
        {
            get { return "hc.exe"; }
        }

        public static IEnumerable<object[]> Hashes
        {
            get
            {
                return new[]
                {
                    new object[] {new Md4()},
                    new object[] {new Md5()},
                    new object[] {new Md2()},
                    new object[] {new Sha1()},
                    new object[] {new Sha224()},
                    new object[] {new Sha256()},
                    new object[] {new Sha384()},
                    new object[] {new Sha512()},
                    new object[] {new Whirlpool()},
                    new object[] {new Crc32()},
                    new object[] {new Tiger()},
                    new object[] {new Tiger2()},
                    new object[] {new Rmd128()},
                    new object[] {new Rmd160()},
                    new object[] {new Rmd256()},
                    new object[] {new Rmd320()},
                    new object[] {new Gost()},
                    new object[] {new Snefru128()},
                    new object[] {new Snefru256()},
                    new object[] {new Tth()},
                    new object[] {new Haval_128_3()},
                    new object[] {new Haval_128_4()},
                    new object[] {new Haval_128_5()},
                    new object[] {new Haval_160_3()},
                    new object[] {new Haval_160_4()},
                    new object[] {new Haval_160_5()},
                    new object[] {new Haval_192_3()},
                    new object[] {new Haval_192_4()},
                    new object[] {new Haval_192_5()},
                    new object[] {new Haval_224_3()},
                    new object[] {new Haval_224_4()},
                    new object[] {new Haval_224_5()},
                    new object[] {new Haval_256_3()},
                    new object[] {new Haval_256_4()},
                    new object[] {new Haval_256_5()},
                    new object[] {new Edonr256()},
                    new object[] {new Edonr512()},
                    new object[] {new Sha_3_224()},
                    new object[] {new Sha_3_256()},
                    new object[] {new Sha_3_384()},
                    new object[] {new Sha_3_512()},
                    new object[] {new Sha_3K_224()},
                    new object[] {new Sha_3K_256()},
                    new object[] {new Sha_3K_384()},
                    new object[] {new Sha_3K_512()},
                    new object[] {new Ntlm()}
                };
            }
        }
    }
}