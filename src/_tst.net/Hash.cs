/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2009-2013 Alexander Egorov
 */

namespace _tst.net
{
    public abstract class Hash
    {
        public virtual string Executable
        {
            get { return "hq.exe " + Algorithm; }
        }

        /// <summary>
        /// Gets the hash of "123" string
        /// </summary>
        public abstract string HashString { get; }

        public abstract string EmptyStringHash { get; }

        /// <summary>
        /// Gets the hash of "12" string
        /// </summary>
        public abstract string StartPartStringHash { get; }

        /// <summary>
        /// Gets the hash of "2" string
        /// </summary>
        public abstract string MiddlePartStringHash { get; }

        /// <summary>
        /// Gets the hash of "23" string
        /// </summary>
        public abstract string TrailPartStringHash { get; }
        
        public abstract string Algorithm { get; }

        public virtual string InitialString
        {
            get { return "123"; }
        }
    }

    public class Tth : Hash
    {
        public override string HashString
        {
            get
            {
                return
                    "E091CFC8F2BC148030F99CBF276B45481ED525CA31EB2EB5";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "5D9ED00A030E638BDB753A6A24FB900E5A63B8E73E6C25B6";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "2765BAA085857604FDB2119B3467E0D2D62F33082931977D";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "466434F0406152138183A157995DF819E5B42FDAA5F98EB4";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "EA3F9A51C877F82EAD99680E1457E4137866A034474F5186";
            }
        }

        public override string Algorithm
        {
            get { return "tth"; }
        }
    }
    
    public class Snefru256 : Hash
    {
        public override string HashString
        {
            get
            {
                return
                    "9A26D1977B322678918E6C3EF1D8291A5A1DCF1AF2FC363DA1666D5422D0A1DE";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "8617F366566A011837F4FB4BA5BEDEA2B892F3ED8B894023D16AE344B2BE5881";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "8D9855DCD9AF52E0CB69ADACEE58159E28DE93DAFF3F801700FF857901E25583";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "70D4951B1B78F820A573BB7E1AC475137D423E7782A437C77F628F2B9A28CE6B";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "FE34437F38B165E8C9693FA22DD52A2DE0D8219F43608F85281E0282BF4D2CFB";
            }
        }

        public override string Algorithm
        {
            get { return "snefru256"; }
        }
    }
    
    public class Snefru128 : Hash
    {
        public override string HashString
        {
            get
            {
                return
                    "ED592424402DBDC9190D700A696EEB6A";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "8617F366566A011837F4FB4BA5BEDEA2";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "4F91363E0D4FC5F6ACB3F456D1CBECD8";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "3DE6FE287D24A9C0942082EEC49AE41D";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "AAA96D6A326F75847904084A12FAF26D";
            }
        }

        public override string Algorithm
        {
            get { return "snefru128"; }
        }
    }
    
    public class Gost : Hash
    {
        public override string HashString
        {
            get
            {
                return
                    "5EF18489617BA2D8D2D7E0DA389AAA4FF022AD01A39512A4FEA1A8C45E439148";
            }
        }

        public override string EmptyStringHash
        {
            get
            {
                return
                    "981E5F3CA30C841487830F84FB433E13AC1101569B9C13584AC483234CD656C0";
            }
        }

        public override string StartPartStringHash
        {
            get
            {
                return
                    "4292481B4AB59A961FF0F7A8E61CA179D0C582018E410C7A986A93EE61840A91";
            }
        }

        public override string MiddlePartStringHash
        {
            get
            {
                return
                    "5B2BEFFE097310AD85DB4B5D94A1D145C2C87AF4F354650484C06B1DD2DFF8DE";
            }
        }

        public override string TrailPartStringHash
        {
            get
            {
                return
                    "A03BF052504B300AA392D03A62145517B6A4C7FF3B1EE41F7D3322CB5B38ACEB";
            }
        }

        public override string Algorithm
        {
            get { return "gost"; }
        }
    }
    
    public class Whirlpool : Hash
    {
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

        public override string Algorithm
        {
            get { return "whirlpool"; }
        }
    }

    public class Sha512 : Hash
    {
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

        public override string Algorithm
        {
            get { return "sha512"; }
        }
    }

    public class Sha384 : Hash
    {
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

        public override string Algorithm
        {
            get { return "sha384"; }
        }
    }

    public class Sha256 : Hash
    {
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

        public override string Algorithm
        {
            get { return "sha256"; }
        }
    }

    public class Sha1 : Hash
    {
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

        public override string Algorithm
        {
            get { return "sha1"; }
        }
    }

    public class Md5 : Hash
    {
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

        public override string Algorithm
        {
            get { return "md5"; }
        }
    }

    public class Md4 : Hash
    {
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

        public override string Algorithm
        {
            get { return "md4"; }
        }
    }
    
    public class Md2 : Hash
    {
        public override string HashString
        {
            get { return "EF1FEDF5D32EAD6B7AAF687DE4ED1B71"; }
        }

        public override string EmptyStringHash
        {
            get { return "8350E5A3E24C153DF2275C9F80692773"; }
        }

        public override string StartPartStringHash
        {
            get { return "D818FDDA9B607DE69729F9E602ED56EF"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "EF39FBF69170B58787CE4E574DB9D842"; }
        }

        public override string TrailPartStringHash
        {
            get { return "F02FC6E199BEB84CF21CF46DDF3CC980"; }
        }

        public override string Algorithm
        {
            get { return "md2"; }
        }
    }
    
    public class Tiger : Hash
    {
        public override string HashString
        {
            get { return "A86807BB96A714FE9B22425893E698334CD71E36B0EEF2BE"; }
        }

        public override string EmptyStringHash
        {
            get { return "3293AC630C13F0245F92BBB1766E16167A4E58492DDE73F3"; }
        }

        public override string StartPartStringHash
        {
            get { return "DC5215E41490E4774986E9BD6220D4C30FB10634C9DC71C6"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "001EBB99B29DDEF56F2F587342BD11680A91CA5726DF8D25"; }
        }

        public override string TrailPartStringHash
        {
            get { return "A0C9D328DC8222F51549C9FE52EB0A9ED4744BF05CF1F671"; }
        }

        public override string Algorithm
        {
            get { return "tiger"; }
        }
    }
    
    public class Tiger2 : Hash
    {
        public override string HashString
        {
            get { return "598B54A953F0ABF9BA647793A3C7C0C4EB8A68698F3594F4"; }
        }

        public override string EmptyStringHash
        {
            get { return "4441BE75F6018773C206C22745374B924AA8313FEF919F41"; }
        }

        public override string StartPartStringHash
        {
            get { return "A5A7008BEE42D5693DC61033B851DE23355D20661C264BC2"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "4F7FEA3FDDAE271A8F1FCBF974425775F23DC21CE393A102"; }
        }

        public override string TrailPartStringHash
        {
            get { return "8A384C20D6F8B3BE611B42D2DCEBAD8FEDF896B08D8EA6C3"; }
        }

        public override string Algorithm
        {
            get { return "tiger2"; }
        }
    }
    
    public class Rmd128 : Hash
    {
        public override string HashString
        {
            get { return "781F357C35DF1FEF3138F6D29670365A"; }
        }

        public override string EmptyStringHash
        {
            get { return "CDF26213A150DC3ECB610F18F6B38B46"; }
        }

        public override string StartPartStringHash
        {
            get { return "798EF9FF954BD5C63530DDB56CB489E1"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "C29837877F697E2BB6BCE5011D64AB04"; }
        }

        public override string TrailPartStringHash
        {
            get { return "5B3F3A4213CCA5DAE9E4ECAA97F0D2C9"; }
        }

        public override string Algorithm
        {
            get { return "ripemd128"; }
        }
    }
    
    public class Rmd160 : Hash
    {
        public override string HashString
        {
            get { return "E3431A8E0ADBF96FD140103DC6F63A3F8FA343AB"; }
        }

        public override string EmptyStringHash
        {
            get { return "9C1185A5C5E9FC54612808977EE8F548B2258D31"; }
        }

        public override string StartPartStringHash
        {
            get { return "58BD2C615CE3FBFA69B1E0E309B610E40CB4C83F"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "412FC6097E62D5C494B8DF37E3805805467D1A2C"; }
        }

        public override string TrailPartStringHash
        {
            get { return "CD220E1B7BD30595052C5D85B1D5ABD091AC3DA8"; }
        }

        public override string Algorithm
        {
            get { return "ripemd160"; }
        }
    }
    
    public class Rmd256 : Hash
    {
        public override string HashString
        {
            get { return "8536753AD7BFACE2DBA89FB318C95B1B42890016057D4C3A2F351CEC3ACBB28B"; }
        }

        public override string EmptyStringHash
        {
            get { return "02BA4C4E5F8ECD1877FC52D64D30E37A2D9774FB1E5D026380AE0168E3C5522D"; }
        }

        public override string StartPartStringHash
        {
            get { return "382EC3F1718B7286F1FA763BDEFD4034F0C3C57173C4309DF45CBCDF63EA5CF7"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "EFECA918CE39EDF8C9B05801EDD0BBB40E1A3A420C4DFC6D2D4E3D04F2943DBE"; }
        }

        public override string TrailPartStringHash
        {
            get { return "0E070FBA7E86586FC4B1A151DE12F69F75F2608542198115013B66BA235809A9"; }
        }

        public override string Algorithm
        {
            get { return "ripemd256"; }
        }
    }
    
    public class Rmd320 : Hash
    {
        public override string HashString
        {
            get { return "BFA11B73AD4E6421A8BA5A1223D9C9F58A5AD456BE98BEE5BFCD19A3ECDC6140CE4C700BE860FDA9"; }
        }

        public override string EmptyStringHash
        {
            get { return "22D65D5661536CDC75C1FDF5C6DE7B41B9F27325EBC61E8557177D705A0EC880151C3A32A00899B8"; }
        }

        public override string StartPartStringHash
        {
            get { return "805A9DB9A0AB592A89A045444158AED4708971545F21D5617F7CC97FF1582D0E0761B2F612A99416"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "B2D2ECEC765A0D5179F9E60AC115D314534E3EA54374321E49397E30D415476037D9D75C6051F4BC"; }
        }

        public override string TrailPartStringHash
        {
            get { return "9D6D35D86FD7208B4EA47F5DF3E41C8373FCB3CE33B174E41F95D276ED0C4BC505E67374BE3D7586"; }
        }

        public override string Algorithm
        {
            get { return "ripemd320"; }
        }
    }
    
    public class Sha224 : Hash
    {
        public override string HashString
        {
            get { return "78D8045D684ABD2EECE923758F3CD781489DF3A48E1278982466017F"; }
        }

        public override string EmptyStringHash
        {
            get { return "D14A028C2A3A2BC9476102BB288234C415A2B01F828EA62AC5B3E42F"; }
        }

        public override string StartPartStringHash
        {
            get { return "3C794F0C67BD561CE841FC6A5999BF0DF298A0F0AE3487EFDA9D0EF4"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "58B2AAA0BFAE7ACC021B3260E941117B529B2E69DE878FD7D45C61A9"; }
        }

        public override string TrailPartStringHash
        {
            get { return "BD1A1BDF6EAE5EE14C3FEE371CCA975A5E052009BC67CE8F11CB7271"; }
        }

        public override string Algorithm
        {
            get { return "sha224"; }
        }
    }
    
    public class Crc32 : Hash
    {
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

        public override string Algorithm
        {
            get { return "crc32"; }
        }
    }
}