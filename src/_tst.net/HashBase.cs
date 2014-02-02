/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2009-2013 Alexander Egorov
 */

using System.IO;
using System.Text;
using NUnit.Framework;

namespace _tst.net
{
    public abstract class HashBase<THash> where THash : Hash, new()
    {
        protected abstract string EmptyFileNameProp { get; }
        protected abstract string EmptyFileProp { get; }
        protected abstract string NotEmptyFileNameProp { get; }
        protected abstract string NotEmptyFileProp { get; }
        protected abstract string BaseTestDirProp { get; }
        protected abstract string SubDirProp { get; }
        protected abstract string SlashProp { get; }
        protected abstract string PathTemplate { get; }

        protected string InitialString
        {
            get { return this.Hash.InitialString; }
        }
        
        protected Hash Hash { get; set; }

        protected ProcessRunner Runner { get; set; }

        protected virtual string Executable
        {
            get { return this.Hash.Executable; }
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

        [SetUp]
        public void Setup()
        {
            this.Runner = new ProcessRunner(string.Format(PathTemplate, Executable));
        }

        [TestFixtureSetUp]
        public virtual void TestFixtureSetup()
        {
            this.Hash = new THash();
            if (!Directory.Exists(BaseTestDirProp))
            {
                Directory.CreateDirectory(BaseTestDirProp);
            }
            else
            {
                Directory.Delete(BaseTestDirProp, true);
                Directory.CreateDirectory(BaseTestDirProp);
            }

            Directory.CreateDirectory(SubDirProp);

            CreateEmptyFile(EmptyFileProp);
            CreateNotEmptyFile(NotEmptyFileProp);

            CreateEmptyFile(SubDirProp + SlashProp + this.EmptyFileNameProp);
            CreateNotEmptyFile(SubDirProp + SlashProp + this.NotEmptyFileNameProp);
        }

        protected void CreateNotEmptyFile( string path, int minSize = 0 )
        {
            FileStream fs = File.Create(path);
            using ( fs )
            {
                byte[] unicode = Encoding.Unicode.GetBytes(InitialString);
                byte[] buffer = Encoding.Convert(Encoding.Unicode, Encoding.ASCII, unicode);

                int written = 0;
                do
                {
                    written += buffer.Length;
                    fs.Write(buffer, 0, buffer.Length);
                } while ( written <= minSize );
            }
        }

        protected void CreateEmptyFile( string path )
        {
            using ( File.Create(path) )
            {
            }
        }

        [TestFixtureTearDown]
        public virtual void TestFixtureTearDown()
        {
            if (Directory.Exists(BaseTestDirProp))
            {
                Directory.Delete(BaseTestDirProp, true);
            }
        }
    }
}