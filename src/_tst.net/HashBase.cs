/*
 * Created by: egr
 * Created at: 05.12.2011
 * © 2007-2011 Alexander Egorov
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
        protected abstract string InitialString { get; }
        public Hash Hash { get; set; }

        [TestFixtureSetUp]
        public void TestFixtureSetup()
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
    }
}