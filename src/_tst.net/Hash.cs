/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2007-2011 Alexander Egorov
 */

namespace _tst.net
{
    public abstract class Hash
    {
        public abstract string Executable { get; }

        public abstract string HashString { get; }

        public abstract string EmptyStringHash { get; }

        public abstract string StartPartStringHash { get; }

        public abstract string MiddlePartStringHash { get; }

        public abstract string TrailPartStringHash { get; }

        public virtual string InitialString
        {
            get { return "123"; }
        }
    }

    public class Whirlpool : Hash
    {
        public override string Executable
        {
            get { return "whirlpool.exe"; }
        }

        public override string HashString
        {
            get
            {
                return
                    "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "19FA61D75522A4669B44E39C1D2E1726C530232130D407F89AFEE0964997F7A73E83BE698B288FEBCF88E3E03C4F0757EA8964E59B63D93708B138CC42A66EB3";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "24E3253CEEB4E32B854C86DAFD7DDD6D747D8C9DE574A003E9D5A590CC20E1254B853A85AE845AB1266874BB70DA8DAC00CA3991C2F3E46E008AD19340E06DBF";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "6034BC99BF63372B3BFA27E1759AE8F337E35C113CC004FB1E7987D463CE301032B98C582BC1163F76176AF6A6CC75841C370C202A0844D23D47BC13373A459B";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "18417525E4D773854FDF954B1C44810628A2C67EA3B3F64229858721A614683A4C125AA5E7BA1FD7504C4A8E654239666EAB6A7D2E67C4F837B1E12459CA2680";
            }
        }
    }

    public class Sha512 : Hash
    {
        public override string Executable
        {
            get { return "sha512.exe"; }
        }

        public override string HashString
        {
            get
            {
                return
                    "3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE47D0D13C5D85F2B0FF8318D2877EEC2F63B931BD47417A81A538327AF927DA3E";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "5AADB45520DCD8726B2822A7A78BB53D794F557199D5D4ABDEDD2C55A4BD6CA73607605C558DE3DB80C8E86C3196484566163ED1327E82E8B6757D1932113CB8";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "40B244112641DD78DD4F93B6C9190DD46E0099194D5A44257B7EFAD6EF9FF4683DA1EDA0244448CB343AA688F5D3EFD7314DAFE580AC0BCBF115AECA9E8DC114";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "6FF334E1051A09E90127BA4E309E026BB830163A2CE3A355AF2CE2310FF6E7E9830D20196A3472BFC8632FD3B60CB56102A84FAE70AB1A32942055EB40022225";
            }
        }
    }

    public class Sha384 : Hash
    {
        public override string Executable
        {
            get { return "sha384.exe"; }
        }

        public override string HashString
        {
            get { return "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F"; }
        }

        public override string EmptyStringHash
        {
            get { return "38B060A751AC96384CD9327EB1B1E36A21FDB71114BE07434C0CC7BF63F6E1DA274EDEBFE76F65FBD51AD2F14898B95B"; }
        }

        public override string StartPartStringHash
        {
            get { return "1E237288D39D815ABC653BEFCAB0EB70966558A5BBC10A24739C116ED2F615BE31E81670F02AF48FE3CF5112F0FA03E8"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "D063457705D66D6F016E4CDD747DB3AF8D70EBFD36BADD63DE6C8CA4A9D8BFB5D874E7FBD750AA804DCADDAE7EEEF51E"; }
        }

        public override string TrailPartStringHash
        {
            get { return "6FDA40FC935C39C3894CA91B3FAF4ACB16FE34D1FC2992C7019F2E35F98FDA0AA18B39727F9F0759E6F1CD737CA5C948"; }
        }
    }

    public class Sha256 : Hash
    {
        public override string Executable
        {
            get { return "sha256.exe"; }
        }

        public override string HashString
        {
            get { return "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3"; }
        }

        public override string EmptyStringHash
        {
            get { return "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"; }
        }

        public override string StartPartStringHash
        {
            get { return "6B51D431DF5D7F141CBECECCF79EDF3DD861C3B4069F0B11661A3EEFACBBA918"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "D4735E3A265E16EEE03F59718B9B5D03019C07D8B6C51F90DA3A666EEC13AB35"; }
        }

        public override string TrailPartStringHash
        {
            get { return "535FA30D7E25DD8A49F1536779734EC8286108D115DA5045D77F3B4185D8F790"; }
        }
    }

    public class Sha1 : Hash
    {
        public override string Executable
        {
            get { return "sha1.exe"; }
        }

        public override string HashString
        {
            get { return "40BD001563085FC35165329EA1FF5C5ECBDBBEEF"; }
        }

        public override string EmptyStringHash
        {
            get { return "DA39A3EE5E6B4B0D3255BFEF95601890AFD80709"; }
        }

        public override string StartPartStringHash
        {
            get { return "7B52009B64FD0A2A49E6D8A939753077792B0554"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "DA4B9237BACCCDF19C0760CAB7AEC4A8359010B0"; }
        }

        public override string TrailPartStringHash
        {
            get { return "D435A6CDD786300DFF204EE7C2EF942D3E9034E2"; }
        }
    }

    public class Md5 : Hash
    {
        public override string Executable
        {
            get { return "md5.exe"; }
        }

        public override string HashString
        {
            get { return "202CB962AC59075B964B07152D234B70"; }
        }

        public override string EmptyStringHash
        {
            get { return "D41D8CD98F00B204E9800998ECF8427E"; }
        }

        public override string StartPartStringHash
        {
            get { return "C20AD4D76FE97759AA27A0C99BFF6710"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "C81E728D9D4C2F636F067F89CC14862C"; }
        }

        public override string TrailPartStringHash
        {
            get { return "37693CFC748049E45D87B8C7D8B9AACD"; }
        }
    }

    public class Md4 : Hash
    {
        public override string Executable
        {
            get { return "md4.exe"; }
        }

        public override string HashString
        {
            get { return "C58CDA49F00748A3BC0FCFA511D516CB"; }
        }

        public override string EmptyStringHash
        {
            get { return "31D6CFE0D16AE931B73C59D7E0C089C0"; }
        }

        public override string StartPartStringHash
        {
            get { return "114C5A33B8D4127FBE492BD6583AEB4D"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "2687049D90DA05D5C9D9AEBED9CDE2A8"; }
        }

        public override string TrailPartStringHash
        {
            get { return "B5839E01E3BB8E57E3FD273A16684618"; }
        }
    }
    
    public class Crc32 : Hash
    {
        public override string Executable
        {
            get { return "crc32.exe"; }
        }

        public override string HashString
        {
            get { return "884863D2"; }
        }

        public override string EmptyStringHash
        {
            get { return "00000000"; }
        }

        public override string StartPartStringHash
        {
            get { return "4F5344CD"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "1AD5BE0D"; }
        }

        public override string TrailPartStringHash
        {
            get { return "13792798"; }
        }
    }
}