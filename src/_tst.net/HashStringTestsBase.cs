/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

using NUnit.Framework;

namespace _tst.net
{
    public abstract class HashStringTestsBase<THash> where THash : Hash, new()
    {
        protected abstract string PathTemplate { get; }
        protected Hash Hash { get; set; }
        protected ProcessRunner Runner { get; set; }

        protected string InitialString
        {
            get { return this.Hash.InitialString; }
        }

        protected virtual string Executable
        {
            get { return this.Hash.Executable; }
        }

        protected string HashString
        {
            get { return this.Hash.HashString; }
        }

        protected string EmptyStringHash
        {
            get { return this.Hash.EmptyStringHash; }
        }

        [SetUp]
        public void Setup()
        {
            this.Runner = new ProcessRunner(string.Format(this.PathTemplate, this.Executable));
        }

        [TestFixtureSetUp]
        public void TestFixtureSetup()
        {
            this.Hash = new THash();
        }
    }
}