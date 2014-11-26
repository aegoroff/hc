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

    public abstract class FileTests<T, THash> : ExeWrapper<T>, IUseFixture<FileFixture>
        where T : Architecture, new()
        where THash : Hash, new()
    {
        protected abstract string EmptyFileNameProp { get; }
        protected abstract string EmptyFileProp { get; }
        protected abstract string NotEmptyFileNameProp { get; }
        protected abstract string NotEmptyFileProp { get; }
        protected Hash Hash { get; private set; }

        protected override string Executable
        {
            get { return "hc.exe"; }
        }

        protected string InitialString
        {
            get { return this.Hash.InitialString; }
        }

        protected string HashString
        {
            get { return this.Hash.HashString; }
        }

        protected string StartPartStringHash
        {
            get { return this.Hash.StartPartStringHash; }
        }

        protected string MiddlePartStringHash
        {
            get { return this.Hash.MiddlePartStringHash; }
        }

        protected string TrailPartStringHash
        {
            get { return this.Hash.TrailPartStringHash; }
        }

        protected string EmptyStringHash
        {
            get { return this.Hash.EmptyStringHash; }
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
            this.Hash = new THash();
            this.CreateEmptyFile(this.EmptyFileProp);
            this.CreateNotEmptyFile(this.NotEmptyFileProp);

            this.CreateEmptyFile(FileFixture.SubDir + FileFixture.Slash + this.EmptyFileNameProp);
            this.CreateNotEmptyFile(FileFixture.SubDir + FileFixture.Slash + this.NotEmptyFileNameProp);
        }

        protected void CreateNotEmptyFile(string path, int minSize = 0)
        {
            FileStream fs = File.Create(path);
            using (fs)
            {
                byte[] unicode = Encoding.Unicode.GetBytes(this.InitialString);
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