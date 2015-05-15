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
let filemask = '123';
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
                for (var i = 0; i < 700; i++)
                {
                    File.AppendAllText(temp, FileContent);
                }
                for (var i = 0; i < 4; i++)
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
            get { return FileTests<ArchWin64>.Hashes; }
        }
    }
}