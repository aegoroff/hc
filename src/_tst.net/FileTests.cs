/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2013 Alexander Egorov
 */

using System;
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
            if (!Directory.Exists(BaseTestDir))
            {
                Directory.CreateDirectory(BaseTestDir);
            }
            else
            {
                Directory.Delete(BaseTestDir, true);
                Directory.CreateDirectory(BaseTestDir);
            }

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
    }
}