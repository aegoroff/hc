/*
 * Created by: egr
 * Created at: 02.02.2014
 * © 2009-2014 Alexander Egorov
 */

namespace _tst.net
{
    public abstract class StringTests<T, THash> : ExeWrapper<T> where T : Architecture, new() where THash: Hash, new ()
    {
        protected StringTests()
        {
            this.Hash = new THash();
        }
        
        protected Hash Hash { get; private set; }

        protected string InitialString
        {
            get { return this.Hash.InitialString; }
        }

        protected string HashString
        {
            get { return this.Hash.HashString; }
        }

        protected string EmptyStringHash
        {
            get { return this.Hash.EmptyStringHash; }
        }
        
        protected override string Executable
        {
            get { return "hc.exe"; }
        }
    }
}