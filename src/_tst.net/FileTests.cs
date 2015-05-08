/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2015 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;

namespace _tst.net
{
    public class FileFixture : IDisposable
    {
        internal const string Slash = @"\";
        internal static string BaseTestDir = Environment.GetEnvironmentVariable("HC_TEST_DIR") ?? @"C:\_tst.net";
        internal static string SubDir = BaseTestDir + Slash + "sub";

        public FileFixture()
        {
            Environment.GetEnvironmentVariable("HC_TEST_DIR");
            this.Dispose();
            Directory.CreateDirectory(BaseTestDir);
            Directory.CreateDirectory(SubDir);
        }
        
        public void Dispose()
        {
            if (Directory.Exists(BaseTestDir))
            {
                Directory.Delete(BaseTestDir, true);
            }
        }
    }

    [Trait("Group", "file")]
    [Trait("Category", "file")]
    public abstract class FileTests<T> : ExeWrapper<T>, IClassFixture<FileFixture>
        where T : Architecture, new()
    {
        protected abstract string EmptyFileNameProp { get; }
        protected abstract string EmptyFileProp { get; }
        protected abstract string NotEmptyFileNameProp { get; }
        protected abstract string NotEmptyFileProp { get; }

        protected override string Executable
        {
            get { return "hc.exe"; }
        }

        protected FileTests() : base(new T())
        {
            Initialize();
        }

        private void Initialize()
        {
            Hash h = new Md5();
            this.CreateEmptyFile(this.EmptyFileProp);
            this.CreateNotEmptyFile(this.NotEmptyFileProp, h.InitialString);

            this.CreateEmptyFile(FileFixture.SubDir + FileFixture.Slash + this.EmptyFileNameProp);
            this.CreateNotEmptyFile(FileFixture.SubDir + FileFixture.Slash + this.NotEmptyFileNameProp, h.InitialString);
        }

        protected void CreateNotEmptyFile(string path, string s, int minSize = 0)
        {
            FileStream fs = File.Create(path);
            using (fs)
            {
                byte[] unicode = Encoding.Unicode.GetBytes(s);
                byte[] buffer = Encoding.Convert(Encoding.Unicode, Encoding.ASCII, unicode);

                int written = 0;
                do
                {
                    written += buffer.Length;
                    fs.Write(buffer, 0, buffer.Length);
                } while (written <= minSize);
            }
        }

        protected void CreateEmptyFile(string path)
        {
            using (File.Create(path))
            {
            }
        }

        protected static IEnumerable<object[]> CreateProperty(object[] data)
        {
            foreach (var h in Hashes)
            {
                foreach (var item in data)
                {
                    var items = item as object[];
                    if (items == null)
                    {
                        yield return new[] { h[0], item };
                    }
                    else
                    {
                        var result = new List<object> { h[0] };
                        result.AddRange(items);
                        yield return result.ToArray();
                    }
                }
            }
        }

        protected const string FileResultTpl = @"{0} | {2} bytes | {1}";
        protected const string FileErrorTpl = @"{0} | {1}";
        protected const string FileResultTimeTpl = @"^(.*?) | \d bytes | \d\.\d{3} sec | ([0-9a-zA-Z]{32,128}?)$";
        protected const string FileResultSfvTpl = @"{0}    {1}";

        protected abstract IList<string> RunFileHashCalculation(Hash h, string file);
        
        protected abstract IList<string> RunDirWithSpecialOption(Hash h, string option);

        [Theory, MemberData("Hashes")]
        public void CalcFile(Hash h)
        {
            IList<string> results = RunFileHashCalculation(h, NotEmptyFileProp);
            Assert.Equal(string.Format(FileResultTpl, NotEmptyFileProp, h.HashString, h.InitialString.Length), results[0]);
            Assert.Equal(1, results.Count);
        }

        [Theory, MemberData("Hashes")]
        public void CalcBigFile(Hash h)
        {
            string file = NotEmptyFileProp + "_big";
            CreateNotEmptyFile(file, h.InitialString, 2 * 1024 * 1024);
            try
            {
                IList<string> results = RunFileHashCalculation(h, file);
                Assert.Contains(" Mb (2", results[0]);
                Assert.Equal(1, results.Count);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Theory, MemberData("Hashes")]
        public void CalcDirChecksumfile(Hash h)
        {
            IList<string> results = this.RunDirWithSpecialOption(h, "--checksumfile");
            Assert.Equal(string.Format(FileResultSfvTpl, h.EmptyStringHash, EmptyFileProp), results[0]);
            Assert.Equal(string.Format(FileResultSfvTpl, h.HashString, NotEmptyFileProp), results[1]);
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void CalcDirSfvCrc32()
        {
            Hash h = new Crc32();
            IList<string> results = this.RunDirWithSpecialOption(h, "--sfv");
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(EmptyFileProp), h.EmptyStringHash), results[0]);
            Assert.Equal(string.Format(FileResultSfvTpl, Path.GetFileName(NotEmptyFileProp), h.HashString), results[1]);
            Assert.Equal(2, results.Count);
        }

        public static IEnumerable<object[]> HashesWithoutCrc32
        {
            get { return from h in Hashes where ((Hash) h[0]).Algorithm != "crc32" select new[] { h[0] }; }
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
                    new object[] {new Sha_3K_512()}
                };
            }
        }
    }
}