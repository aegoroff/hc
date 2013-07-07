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

    
    public class Haval_128_5 : Hash
    {
        public override string HashString
        {
            get { return "092356CE125C84828EA26E633328EF0B"; }
        }

        public override string EmptyStringHash
        {
            get { return "184B8482A0C050DCA54B59C7F05BF5DD"; }
        }

        public override string StartPartStringHash
        {
            get { return "6D417B9019FE6D0F4BCC23F1EAF7AAA3"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "F8C0ED63C4A9AB5BFD15E117D1AB260D"; }
        }

        public override string TrailPartStringHash
        {
            get { return "3F7E90ED422E644C57217DCE4FA340A1"; }
        }

        public override string Algorithm
        {
            get { return "haval-128-5"; }
        }
    }


    public class Haval_160_3 : Hash
    {
        public override string HashString
        {
            get { return "9AA8070C350A5B8E9EF84D50C501488DCD209D89"; }
        }

        public override string EmptyStringHash
        {
            get { return "D353C3AE22A25401D257643836D7231A9A95F953"; }
        }

        public override string StartPartStringHash
        {
            get { return "EBB548637C716F026F3018735BA6F6033526A8C2"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "C20E1B242B7A98ABCCACF16ED59274EB1B37E3DC"; }
        }

        public override string TrailPartStringHash
        {
            get { return "9324822D9B2C6901565196584088D3496D28B533"; }
        }

        public override string Algorithm
        {
            get { return "haval-160-3"; }
        }
    }


    public class Haval_160_4 : Hash
    {
        public override string HashString
        {
            get { return "7F21296963CC57E11A3DF4EC10BC79A4489125B8"; }
        }

        public override string EmptyStringHash
        {
            get { return "1D33AAE1BE4146DBAACA0B6E70D7A11F10801525"; }
        }

        public override string StartPartStringHash
        {
            get { return "2A9E46B62883F89ACCEA8D4B2E4FEE7D62E15A8F"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "AB0551EA9FA84E128D4C483A04C86D99479E9408"; }
        }

        public override string TrailPartStringHash
        {
            get { return "6BEA40B3117FC2741C6D1DAA6C661835FC0594F7"; }
        }

        public override string Algorithm
        {
            get { return "haval-160-4"; }
        }
    }


    public class Haval_160_5 : Hash
    {
        public override string HashString
        {
            get { return "8FF0C07890BE1CD2388DB65C85DA7B6C34E8A3D1"; }
        }

        public override string EmptyStringHash
        {
            get { return "255158CFC1EED1A7BE7C55DDD64D9790415B933B"; }
        }

        public override string StartPartStringHash
        {
            get { return "94302B716D76A079688C61AD457515026D803DFC"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "B93D49BAAD60D772484E97A127217410867F2182"; }
        }

        public override string TrailPartStringHash
        {
            get { return "E6B094A29324BD1D5BEFA5222C948A363F8B7DFA"; }
        }

        public override string Algorithm
        {
            get { return "haval-160-5"; }
        }
    }


    public class Haval_192_3 : Hash
    {
        public override string HashString
        {
            get { return "B00150CCD88C4404BBB4DE1D044D22CDE1D0AF78BFCFE911"; }
        }

        public override string EmptyStringHash
        {
            get { return "E9C48D7903EAF2A91C5B350151EFCB175C0FC82DE2289A4E"; }
        }

        public override string StartPartStringHash
        {
            get { return "70923D7E0C6B2E1E60921FF7A15C2FE979054CC2A9408FC0"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "2F36EAB57A6750A26E531EB29AF68E8AB99DD7B0FFA0ED90"; }
        }

        public override string TrailPartStringHash
        {
            get { return "B761D8B30682CFA436A42F616F21E0423BB965E7D72479C6"; }
        }

        public override string Algorithm
        {
            get { return "haval-192-3"; }
        }
    }


    public class Haval_192_4 : Hash
    {
        public override string HashString
        {
            get { return "47E4674075CB59C43DFF566B98B40F62F2652B5697B89C28"; }
        }

        public override string EmptyStringHash
        {
            get { return "4A8372945AFA55C7DEAD800311272523CA19D42EA47B72DA"; }
        }

        public override string StartPartStringHash
        {
            get { return "494BC12E35701EBEE02C1668312C157F1D718DDD15C68F79"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "5E1EA11E4CEAE5A3434C9E833E04CAC0254F211693E7D62B"; }
        }

        public override string TrailPartStringHash
        {
            get { return "C78E5A9590E28E364E51530450F51B182FF5A1244B6F682F"; }
        }

        public override string Algorithm
        {
            get { return "haval-192-4"; }
        }
    }


    public class Haval_192_5 : Hash
    {
        public override string HashString
        {
            get { return "575C8E28A5BCFBC10179020D70C6C367280B40FC7AD806C3"; }
        }

        public override string EmptyStringHash
        {
            get { return "4839D0626F95935E17EE2FC4509387BBE2CC46CB382FFE85"; }
        }

        public override string StartPartStringHash
        {
            get { return "4F2B554760AE4A7F36F4439C0C39BECBBAA198CDF936B7EC"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "D4BFF180C4598DF62B227F5A540837EBAA616EE6C61C5F05"; }
        }

        public override string TrailPartStringHash
        {
            get { return "F8573316B48D2C417EC3F79A234E5690066A0FA5947694BF"; }
        }

        public override string Algorithm
        {
            get { return "haval-192-5"; }
        }
    }


    public class Haval_224_3 : Hash
    {
        public override string HashString
        {
            get { return "A294D60D7351B4BC2E5962F5FF5A620B430B5069F27923E70D8AFBF0"; }
        }

        public override string EmptyStringHash
        {
            get { return "C5AAE9D47BFFCAAF84A8C6E7CCACD60A0DD1932BE7B1A192B9214B6D"; }
        }

        public override string StartPartStringHash
        {
            get { return "3371D568ED929816A63D9A5EF162FD8B3DB1AF983EB9513612D14D25"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "2E9B0C63E53755C70F926E3CE7C1BA57511D78E6AD83DF9751B36A52"; }
        }

        public override string TrailPartStringHash
        {
            get { return "92A067B4D7E1812BEC3087354943882BB2C3CEC34DD396B87948534F"; }
        }

        public override string Algorithm
        {
            get { return "haval-224-3"; }
        }
    }


    public class Haval_224_4 : Hash
    {
        public override string HashString
        {
            get { return "B9E3BCFBC5EA72626CACFBEB0E055CB89ADF2CE9B0E24A3C8A32CB34"; }
        }

        public override string EmptyStringHash
        {
            get { return "3E56243275B3B81561750550E36FCD676AD2F5DD9E15F2E89E6ED78E"; }
        }

        public override string StartPartStringHash
        {
            get { return "11F905EAE1EA61672970041C2074CF98703AF963999909C2A9DE84B3"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "88FA26CAE0ECDC529F81905E9A336D99AE39986692989B72E05905C9"; }
        }

        public override string TrailPartStringHash
        {
            get { return "29A3368506242C5B35BA859077BA8810147F3DDE200301270364C514"; }
        }

        public override string Algorithm
        {
            get { return "haval-224-4"; }
        }
    }


    public class Haval_224_5 : Hash
    {
        public override string HashString
        {
            get { return "FC2D1B6F27FB775D8E7030715AF85B646239C9D9D675CCFF309B49B7"; }
        }

        public override string EmptyStringHash
        {
            get { return "4A0513C032754F5582A758D35917AC9ADF3854219B39E3AC77D1837E"; }
        }

        public override string StartPartStringHash
        {
            get { return "41C1B8A6AC60949AA2A50313F19D100910881BBF0BC5761F88CDEBC6"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "DAB55FE059D3DBCACF9E8C5A55C21D850391582CB2E4831AAA4E75D1"; }
        }

        public override string TrailPartStringHash
        {
            get { return "458827F727F77C599B452FDBC657245CB8226B4487F9987BD755E1DA"; }
        }

        public override string Algorithm
        {
            get { return "haval-224-5"; }
        }
    }


    public class Haval_256_3 : Hash
    {
        public override string HashString
        {
            get { return "E3891CB6FD1A883A1AE723F13BA336F586FA8C10506C4799C209D10113675BC1"; }
        }

        public override string EmptyStringHash
        {
            get { return "4F6938531F0BC8991F62DA7BBD6F7DE3FAD44562B8C6F4EBF146D5B4E46F7C17"; }
        }

        public override string StartPartStringHash
        {
            get { return "F96418428C992DAB2139CFDB82D89725A192AB53F1F4563D59C0473A15B3418B"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "A1055E7620768718DC9635D0358F3E4AF845F596C0BAED6A1BF0132A33F0F59A"; }
        }

        public override string TrailPartStringHash
        {
            get { return "066DEC0561FD9E2E89A24BC2DE241B2CA099AD5B360C33876F84B262631A4DAC"; }
        }

        public override string Algorithm
        {
            get { return "haval-256-3"; }
        }
    }


    public class Haval_256_4 : Hash
    {
        public override string HashString
        {
            get { return "A16D7FCD48CED7B612FF2C35D78241EB89A752EFF2931647A32C2C3C22F8D747"; }
        }

        public override string EmptyStringHash
        {
            get { return "C92B2E23091E80E375DADCE26982482D197B1A2521BE82DA819F8CA2C579B99B"; }
        }

        public override string StartPartStringHash
        {
            get { return "50AA70038496D65EF6DA866025B31EF493FE33DC5289B615EA3FCA9442705146"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "72AEF38030403F9143002BF1FF8BFC393B0A51A60B27F3C331DCB844A37D1EFC"; }
        }

        public override string TrailPartStringHash
        {
            get { return "8DD972314909F89C8C41026E53288C126FCA5762BF8530028B47790C6224A86F"; }
        }

        public override string Algorithm
        {
            get { return "haval-256-4"; }
        }
    }


    public class Haval_256_5 : Hash
    {
        public override string HashString
        {
            get { return "386DBED5748A4B9E9409D8CE94ACFE8DF324A166EAC054E9817F85F7AEC8AED5"; }
        }

        public override string EmptyStringHash
        {
            get { return "BE417BB4DD5CFB76C7126F4F8EEB1553A449039307B1A3CD451DBFDC0FBBE330"; }
        }

        public override string StartPartStringHash
        {
            get { return "95CCCBF651A772BCE1270C14C262292E973362C06B871D2FA1DDA092DFC2908D"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "2D7584D413364CB958B63D74B4972B97FC3E1154A302D93782C19E49489B964F"; }
        }

        public override string TrailPartStringHash
        {
            get { return "F119285F0556724635892BA10F40400C0F7140905A4A65D28F51063B3518EFFD"; }
        }

        public override string Algorithm
        {
            get { return "haval-256-5"; }
        }
    }

    public class Haval_128_4 : Hash
    {
        public override string HashString
        {
            get { return "7FD91A17538880FB2007F59A49B1C5A5"; }
        }

        public override string EmptyStringHash
        {
            get { return "EE6BBF4D6A46A679B3A856C88538BB98"; }
        }

        public override string StartPartStringHash
        {
            get { return "68B13909A2FB3843E58C058616E99592"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "46FF1335106879C451A7ADFB41D7E937"; }
        }

        public override string TrailPartStringHash
        {
            get { return "09DA21F61301ED3C4F9CAC4583F99BAD"; }
        }

        public override string Algorithm
        {
            get { return "haval-128-4"; }
        }
    }
    
    public class Haval_128_3 : Hash
    {
        public override string HashString
        {
            get { return "BDC9FC6D0E82C40FA3DE3FD54803DBD1"; }
        }

        public override string EmptyStringHash
        {
            get { return "C68F39913F901F3DDF44C707357A7D70"; }
        }

        public override string StartPartStringHash
        {
            get { return "EAB14FB0CB7F5B15C1751B9ED601B2AE"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "68FE782E5651504AA6C017A8B40D7AF5"; }
        }

        public override string TrailPartStringHash
        {
            get { return "EB351A7781DBC1C0E7DAFF5915577AFC"; }
        }

        public override string Algorithm
        {
            get { return "haval-128-3"; }
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