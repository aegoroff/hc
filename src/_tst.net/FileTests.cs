/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2013 Alexander Egorov
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Xunit;

namespace _tst.net
{
    public class FileFixture : IDisposable
    {
        internal const string Slash = @"\";
        internal const string BaseTestDir = @"C:\_tst.net";
        internal const string SubDir = BaseTestDir + Slash + "sub";

        public FileFixture()
        {
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

    [Trait("Category", "file")]
    public abstract class FileTests<T> : ExeWrapper<T>, IUseFixture<FileFixture>
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

        protected FileTests()
        {
            Initialize();
        }

        public void SetFixture(FileFixture data)
        {
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