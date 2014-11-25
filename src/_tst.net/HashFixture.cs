/*
* Created by: egr
* Created at: 25.11.2014
* © 2009-2014 Alexander Egorov
*/

namespace _tst.net
{
    public class HashFixture<THash> where THash : Hash, new()
    {
        public HashFixture()
        {
            this.Hash = new THash();
        }

        public Hash Hash { get; set; }
    }
}