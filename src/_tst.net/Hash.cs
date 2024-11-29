/*
 * Created by: egr
 * Created at: 11.09.2010
 * © 2009-2024 Alexander Egorov
 */

namespace _tst.net;

public abstract class Hash
{
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

    public string InitialString => "123";
}

public class Crc32 : Hash
{
    public override string HashString => "884863D2";

    public override string EmptyStringHash => "00000000";

    public override string StartPartStringHash => "4F5344CD";

    public override string MiddlePartStringHash => "1AD5BE0D";

    public override string TrailPartStringHash => "13792798";

    public override string Algorithm => "crc32";
}

public class Crc32c : Hash
{
    public override string HashString => "107B2FB2";

    public override string EmptyStringHash => "00000000";

    public override string StartPartStringHash => "7355C460";

    public override string MiddlePartStringHash => "83A56A17";

    public override string TrailPartStringHash => "B5D9EFFA";

    public override string Algorithm => "crc32c";
}

public class Md4 : Hash
{
    public override string HashString => "C58CDA49F00748A3BC0FCFA511D516CB";

    public override string EmptyStringHash => "31D6CFE0D16AE931B73C59D7E0C089C0";

    public override string StartPartStringHash => "114C5A33B8D4127FBE492BD6583AEB4D";

    public override string MiddlePartStringHash => "2687049D90DA05D5C9D9AEBED9CDE2A8";

    public override string TrailPartStringHash => "B5839E01E3BB8E57E3FD273A16684618";

    public override string Algorithm => "md4";
}

public class Md5 : Hash
{
    public override string HashString => "202CB962AC59075B964B07152D234B70";

    public override string EmptyStringHash => "D41D8CD98F00B204E9800998ECF8427E";

    public override string StartPartStringHash => "C20AD4D76FE97759AA27A0C99BFF6710";

    public override string MiddlePartStringHash => "C81E728D9D4C2F636F067F89CC14862C";

    public override string TrailPartStringHash => "37693CFC748049E45D87B8C7D8B9AACD";

    public override string Algorithm => "md5";
}

public class Sha1 : Hash
{
    public override string HashString => "40BD001563085FC35165329EA1FF5C5ECBDBBEEF";

    public override string EmptyStringHash => "DA39A3EE5E6B4B0D3255BFEF95601890AFD80709";

    public override string StartPartStringHash => "7B52009B64FD0A2A49E6D8A939753077792B0554";

    public override string MiddlePartStringHash => "DA4B9237BACCCDF19C0760CAB7AEC4A8359010B0";

    public override string TrailPartStringHash => "D435A6CDD786300DFF204EE7C2EF942D3E9034E2";

    public override string Algorithm => "sha1";
}

public class Sha256 : Hash
{
    public override string HashString => "A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3";

    public override string EmptyStringHash => "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

    public override string StartPartStringHash => "6B51D431DF5D7F141CBECECCF79EDF3DD861C3B4069F0B11661A3EEFACBBA918";

    public override string MiddlePartStringHash => "D4735E3A265E16EEE03F59718B9B5D03019C07D8B6C51F90DA3A666EEC13AB35";

    public override string TrailPartStringHash => "535FA30D7E25DD8A49F1536779734EC8286108D115DA5045D77F3B4185D8F790";

    public override string Algorithm => "sha256";
}

public class Sha384 : Hash
{
    public override string HashString => "9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F";

    public override string EmptyStringHash => "38B060A751AC96384CD9327EB1B1E36A21FDB71114BE07434C0CC7BF63F6E1DA274EDEBFE76F65FBD51AD2F14898B95B";

    public override string StartPartStringHash => "1E237288D39D815ABC653BEFCAB0EB70966558A5BBC10A24739C116ED2F615BE31E81670F02AF48FE3CF5112F0FA03E8";

    public override string MiddlePartStringHash => "D063457705D66D6F016E4CDD747DB3AF8D70EBFD36BADD63DE6C8CA4A9D8BFB5D874E7FBD750AA804DCADDAE7EEEF51E";

    public override string TrailPartStringHash => "6FDA40FC935C39C3894CA91B3FAF4ACB16FE34D1FC2992C7019F2E35F98FDA0AA18B39727F9F0759E6F1CD737CA5C948";

    public override string Algorithm => "sha384";
}

public class Sha512 : Hash
{
    public override string HashString => "3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2";

    public override string EmptyStringHash => "CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE47D0D13C5D85F2B0FF8318D2877EEC2F63B931BD47417A81A538327AF927DA3E";

    public override string StartPartStringHash => "5AADB45520DCD8726B2822A7A78BB53D794F557199D5D4ABDEDD2C55A4BD6CA73607605C558DE3DB80C8E86C3196484566163ED1327E82E8B6757D1932113CB8";

    public override string MiddlePartStringHash => "40B244112641DD78DD4F93B6C9190DD46E0099194D5A44257B7EFAD6EF9FF4683DA1EDA0244448CB343AA688F5D3EFD7314DAFE580AC0BCBF115AECA9E8DC114";

    public override string TrailPartStringHash => "6FF334E1051A09E90127BA4E309E026BB830163A2CE3A355AF2CE2310FF6E7E9830D20196A3472BFC8632FD3B60CB56102A84FAE70AB1A32942055EB40022225";

    public override string Algorithm => "sha512";
}

public class Whirlpool : Hash
{
    public override string HashString => "344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F";

    public override string EmptyStringHash => "19FA61D75522A4669B44E39C1D2E1726C530232130D407F89AFEE0964997F7A73E83BE698B288FEBCF88E3E03C4F0757EA8964E59B63D93708B138CC42A66EB3";

    public override string StartPartStringHash => "24E3253CEEB4E32B854C86DAFD7DDD6D747D8C9DE574A003E9D5A590CC20E1254B853A85AE845AB1266874BB70DA8DAC00CA3991C2F3E46E008AD19340E06DBF";

    public override string MiddlePartStringHash => "6034BC99BF63372B3BFA27E1759AE8F337E35C113CC004FB1E7987D463CE301032B98C582BC1163F76176AF6A6CC75841C370C202A0844D23D47BC13373A459B";

    public override string TrailPartStringHash => "18417525E4D773854FDF954B1C44810628A2C67EA3B3F64229858721A614683A4C125AA5E7BA1FD7504C4A8E654239666EAB6A7D2E67C4F837B1E12459CA2680";

    public override string Algorithm => "whirlpool";
}

public class Md2 : Hash
{
    public override string HashString => "EF1FEDF5D32EAD6B7AAF687DE4ED1B71";

    public override string EmptyStringHash => "8350E5A3E24C153DF2275C9F80692773";

    public override string StartPartStringHash => "D818FDDA9B607DE69729F9E602ED56EF";

    public override string MiddlePartStringHash => "EF39FBF69170B58787CE4E574DB9D842";

    public override string TrailPartStringHash => "F02FC6E199BEB84CF21CF46DDF3CC980";

    public override string Algorithm => "md2";
}

public class Sha224 : Hash
{
    public override string HashString => "78D8045D684ABD2EECE923758F3CD781489DF3A48E1278982466017F";

    public override string EmptyStringHash => "D14A028C2A3A2BC9476102BB288234C415A2B01F828EA62AC5B3E42F";

    public override string StartPartStringHash => "3C794F0C67BD561CE841FC6A5999BF0DF298A0F0AE3487EFDA9D0EF4";

    public override string MiddlePartStringHash => "58B2AAA0BFAE7ACC021B3260E941117B529B2E69DE878FD7D45C61A9";

    public override string TrailPartStringHash => "BD1A1BDF6EAE5EE14C3FEE371CCA975A5E052009BC67CE8F11CB7271";

    public override string Algorithm => "sha224";
}

public class Tiger : Hash
{
    public override string HashString => "A86807BB96A714FE9B22425893E698334CD71E36B0EEF2BE";

    public override string EmptyStringHash => "3293AC630C13F0245F92BBB1766E16167A4E58492DDE73F3";

    public override string StartPartStringHash => "DC5215E41490E4774986E9BD6220D4C30FB10634C9DC71C6";

    public override string MiddlePartStringHash => "001EBB99B29DDEF56F2F587342BD11680A91CA5726DF8D25";

    public override string TrailPartStringHash => "A0C9D328DC8222F51549C9FE52EB0A9ED4744BF05CF1F671";

    public override string Algorithm => "tiger";
}

public class Tiger2 : Hash
{
    public override string HashString => "598B54A953F0ABF9BA647793A3C7C0C4EB8A68698F3594F4";

    public override string EmptyStringHash => "4441BE75F6018773C206C22745374B924AA8313FEF919F41";

    public override string StartPartStringHash => "A5A7008BEE42D5693DC61033B851DE23355D20661C264BC2";

    public override string MiddlePartStringHash => "4F7FEA3FDDAE271A8F1FCBF974425775F23DC21CE393A102";

    public override string TrailPartStringHash => "8A384C20D6F8B3BE611B42D2DCEBAD8FEDF896B08D8EA6C3";

    public override string Algorithm => "tiger2";
}

public class Ripemd128 : Hash
{
    public override string HashString => "781F357C35DF1FEF3138F6D29670365A";

    public override string EmptyStringHash => "CDF26213A150DC3ECB610F18F6B38B46";

    public override string StartPartStringHash => "798EF9FF954BD5C63530DDB56CB489E1";

    public override string MiddlePartStringHash => "C29837877F697E2BB6BCE5011D64AB04";

    public override string TrailPartStringHash => "5B3F3A4213CCA5DAE9E4ECAA97F0D2C9";

    public override string Algorithm => "ripemd128";
}

public class Ripemd160 : Hash
{
    public override string HashString => "E3431A8E0ADBF96FD140103DC6F63A3F8FA343AB";

    public override string EmptyStringHash => "9C1185A5C5E9FC54612808977EE8F548B2258D31";

    public override string StartPartStringHash => "58BD2C615CE3FBFA69B1E0E309B610E40CB4C83F";

    public override string MiddlePartStringHash => "412FC6097E62D5C494B8DF37E3805805467D1A2C";

    public override string TrailPartStringHash => "CD220E1B7BD30595052C5D85B1D5ABD091AC3DA8";

    public override string Algorithm => "ripemd160";
}

public class Ripemd256 : Hash
{
    public override string HashString => "8536753AD7BFACE2DBA89FB318C95B1B42890016057D4C3A2F351CEC3ACBB28B";

    public override string EmptyStringHash => "02BA4C4E5F8ECD1877FC52D64D30E37A2D9774FB1E5D026380AE0168E3C5522D";

    public override string StartPartStringHash => "382EC3F1718B7286F1FA763BDEFD4034F0C3C57173C4309DF45CBCDF63EA5CF7";

    public override string MiddlePartStringHash => "EFECA918CE39EDF8C9B05801EDD0BBB40E1A3A420C4DFC6D2D4E3D04F2943DBE";

    public override string TrailPartStringHash => "0E070FBA7E86586FC4B1A151DE12F69F75F2608542198115013B66BA235809A9";

    public override string Algorithm => "ripemd256";
}

public class Ripemd320 : Hash
{
    public override string HashString => "BFA11B73AD4E6421A8BA5A1223D9C9F58A5AD456BE98BEE5BFCD19A3ECDC6140CE4C700BE860FDA9";

    public override string EmptyStringHash => "22D65D5661536CDC75C1FDF5C6DE7B41B9F27325EBC61E8557177D705A0EC880151C3A32A00899B8";

    public override string StartPartStringHash => "805A9DB9A0AB592A89A045444158AED4708971545F21D5617F7CC97FF1582D0E0761B2F612A99416";

    public override string MiddlePartStringHash => "B2D2ECEC765A0D5179F9E60AC115D314534E3EA54374321E49397E30D415476037D9D75C6051F4BC";

    public override string TrailPartStringHash => "9D6D35D86FD7208B4EA47F5DF3E41C8373FCB3CE33B174E41F95D276ED0C4BC505E67374BE3D7586";

    public override string Algorithm => "ripemd320";
}

public class Gost : Hash
{
    public override string HashString => "5EF18489617BA2D8D2D7E0DA389AAA4FF022AD01A39512A4FEA1A8C45E439148";

    public override string EmptyStringHash => "981E5F3CA30C841487830F84FB433E13AC1101569B9C13584AC483234CD656C0";

    public override string StartPartStringHash => "4292481B4AB59A961FF0F7A8E61CA179D0C582018E410C7A986A93EE61840A91";

    public override string MiddlePartStringHash => "5B2BEFFE097310AD85DB4B5D94A1D145C2C87AF4F354650484C06B1DD2DFF8DE";

    public override string TrailPartStringHash => "A03BF052504B300AA392D03A62145517B6A4C7FF3B1EE41F7D3322CB5B38ACEB";

    public override string Algorithm => "gost";
}

public class Snefru256 : Hash
{
    public override string HashString => "9A26D1977B322678918E6C3EF1D8291A5A1DCF1AF2FC363DA1666D5422D0A1DE";

    public override string EmptyStringHash => "8617F366566A011837F4FB4BA5BEDEA2B892F3ED8B894023D16AE344B2BE5881";

    public override string StartPartStringHash => "8D9855DCD9AF52E0CB69ADACEE58159E28DE93DAFF3F801700FF857901E25583";

    public override string MiddlePartStringHash => "70D4951B1B78F820A573BB7E1AC475137D423E7782A437C77F628F2B9A28CE6B";

    public override string TrailPartStringHash => "FE34437F38B165E8C9693FA22DD52A2DE0D8219F43608F85281E0282BF4D2CFB";

    public override string Algorithm => "snefru256";
}

public class Snefru128 : Hash
{
    public override string HashString => "ED592424402DBDC9190D700A696EEB6A";

    public override string EmptyStringHash => "8617F366566A011837F4FB4BA5BEDEA2";

    public override string StartPartStringHash => "4F91363E0D4FC5F6ACB3F456D1CBECD8";

    public override string MiddlePartStringHash => "3DE6FE287D24A9C0942082EEC49AE41D";

    public override string TrailPartStringHash => "AAA96D6A326F75847904084A12FAF26D";

    public override string Algorithm => "snefru128";
}

public class Tth : Hash
{
    public override string HashString => "E091CFC8F2BC148030F99CBF276B45481ED525CA31EB2EB5";

    public override string EmptyStringHash => "5D9ED00A030E638BDB753A6A24FB900E5A63B8E73E6C25B6";

    public override string StartPartStringHash => "2765BAA085857604FDB2119B3467E0D2D62F33082931977D";

    public override string MiddlePartStringHash => "466434F0406152138183A157995DF819E5B42FDAA5F98EB4";

    public override string TrailPartStringHash => "EA3F9A51C877F82EAD99680E1457E4137866A034474F5186";

    public override string Algorithm => "tth";
}

public class Haval_128_3 : Hash
{
    public override string HashString => "BDC9FC6D0E82C40FA3DE3FD54803DBD1";

    public override string EmptyStringHash => "C68F39913F901F3DDF44C707357A7D70";

    public override string StartPartStringHash => "EAB14FB0CB7F5B15C1751B9ED601B2AE";

    public override string MiddlePartStringHash => "68FE782E5651504AA6C017A8B40D7AF5";

    public override string TrailPartStringHash => "EB351A7781DBC1C0E7DAFF5915577AFC";

    public override string Algorithm => "haval-128-3";
}

public class Haval_128_4 : Hash
{
    public override string HashString => "7FD91A17538880FB2007F59A49B1C5A5";

    public override string EmptyStringHash => "EE6BBF4D6A46A679B3A856C88538BB98";

    public override string StartPartStringHash => "68B13909A2FB3843E58C058616E99592";

    public override string MiddlePartStringHash => "46FF1335106879C451A7ADFB41D7E937";

    public override string TrailPartStringHash => "09DA21F61301ED3C4F9CAC4583F99BAD";

    public override string Algorithm => "haval-128-4";
}

public class Haval_128_5 : Hash
{
    public override string HashString => "092356CE125C84828EA26E633328EF0B";

    public override string EmptyStringHash => "184B8482A0C050DCA54B59C7F05BF5DD";

    public override string StartPartStringHash => "6D417B9019FE6D0F4BCC23F1EAF7AAA3";

    public override string MiddlePartStringHash => "F8C0ED63C4A9AB5BFD15E117D1AB260D";

    public override string TrailPartStringHash => "3F7E90ED422E644C57217DCE4FA340A1";

    public override string Algorithm => "haval-128-5";
}

public class Haval_160_3 : Hash
{
    public override string HashString => "9AA8070C350A5B8E9EF84D50C501488DCD209D89";

    public override string EmptyStringHash => "D353C3AE22A25401D257643836D7231A9A95F953";

    public override string StartPartStringHash => "EBB548637C716F026F3018735BA6F6033526A8C2";

    public override string MiddlePartStringHash => "C20E1B242B7A98ABCCACF16ED59274EB1B37E3DC";

    public override string TrailPartStringHash => "9324822D9B2C6901565196584088D3496D28B533";

    public override string Algorithm => "haval-160-3";
}

public class Haval_160_4 : Hash
{
    public override string HashString => "7F21296963CC57E11A3DF4EC10BC79A4489125B8";

    public override string EmptyStringHash => "1D33AAE1BE4146DBAACA0B6E70D7A11F10801525";

    public override string StartPartStringHash => "2A9E46B62883F89ACCEA8D4B2E4FEE7D62E15A8F";

    public override string MiddlePartStringHash => "AB0551EA9FA84E128D4C483A04C86D99479E9408";

    public override string TrailPartStringHash => "6BEA40B3117FC2741C6D1DAA6C661835FC0594F7";

    public override string Algorithm => "haval-160-4";
}

public class Haval_160_5 : Hash
{
    public override string HashString => "8FF0C07890BE1CD2388DB65C85DA7B6C34E8A3D1";

    public override string EmptyStringHash => "255158CFC1EED1A7BE7C55DDD64D9790415B933B";

    public override string StartPartStringHash => "94302B716D76A079688C61AD457515026D803DFC";

    public override string MiddlePartStringHash => "B93D49BAAD60D772484E97A127217410867F2182";

    public override string TrailPartStringHash => "E6B094A29324BD1D5BEFA5222C948A363F8B7DFA";

    public override string Algorithm => "haval-160-5";
}

public class Haval_192_3 : Hash
{
    public override string HashString => "B00150CCD88C4404BBB4DE1D044D22CDE1D0AF78BFCFE911";

    public override string EmptyStringHash => "E9C48D7903EAF2A91C5B350151EFCB175C0FC82DE2289A4E";

    public override string StartPartStringHash => "70923D7E0C6B2E1E60921FF7A15C2FE979054CC2A9408FC0";

    public override string MiddlePartStringHash => "2F36EAB57A6750A26E531EB29AF68E8AB99DD7B0FFA0ED90";

    public override string TrailPartStringHash => "B761D8B30682CFA436A42F616F21E0423BB965E7D72479C6";

    public override string Algorithm => "haval-192-3";
}

public class Haval_192_4 : Hash
{
    public override string HashString => "47E4674075CB59C43DFF566B98B40F62F2652B5697B89C28";

    public override string EmptyStringHash => "4A8372945AFA55C7DEAD800311272523CA19D42EA47B72DA";

    public override string StartPartStringHash => "494BC12E35701EBEE02C1668312C157F1D718DDD15C68F79";

    public override string MiddlePartStringHash => "5E1EA11E4CEAE5A3434C9E833E04CAC0254F211693E7D62B";

    public override string TrailPartStringHash => "C78E5A9590E28E364E51530450F51B182FF5A1244B6F682F";

    public override string Algorithm => "haval-192-4";
}

public class Haval_192_5 : Hash
{
    public override string HashString => "575C8E28A5BCFBC10179020D70C6C367280B40FC7AD806C3";

    public override string EmptyStringHash => "4839D0626F95935E17EE2FC4509387BBE2CC46CB382FFE85";

    public override string StartPartStringHash => "4F2B554760AE4A7F36F4439C0C39BECBBAA198CDF936B7EC";

    public override string MiddlePartStringHash => "D4BFF180C4598DF62B227F5A540837EBAA616EE6C61C5F05";

    public override string TrailPartStringHash => "F8573316B48D2C417EC3F79A234E5690066A0FA5947694BF";

    public override string Algorithm => "haval-192-5";
}

public class Haval_224_3 : Hash
{
    public override string HashString => "A294D60D7351B4BC2E5962F5FF5A620B430B5069F27923E70D8AFBF0";

    public override string EmptyStringHash => "C5AAE9D47BFFCAAF84A8C6E7CCACD60A0DD1932BE7B1A192B9214B6D";

    public override string StartPartStringHash => "3371D568ED929816A63D9A5EF162FD8B3DB1AF983EB9513612D14D25";

    public override string MiddlePartStringHash => "2E9B0C63E53755C70F926E3CE7C1BA57511D78E6AD83DF9751B36A52";

    public override string TrailPartStringHash => "92A067B4D7E1812BEC3087354943882BB2C3CEC34DD396B87948534F";

    public override string Algorithm => "haval-224-3";
}

public class Haval_224_4 : Hash
{
    public override string HashString => "B9E3BCFBC5EA72626CACFBEB0E055CB89ADF2CE9B0E24A3C8A32CB34";

    public override string EmptyStringHash => "3E56243275B3B81561750550E36FCD676AD2F5DD9E15F2E89E6ED78E";

    public override string StartPartStringHash => "11F905EAE1EA61672970041C2074CF98703AF963999909C2A9DE84B3";

    public override string MiddlePartStringHash => "88FA26CAE0ECDC529F81905E9A336D99AE39986692989B72E05905C9";

    public override string TrailPartStringHash => "29A3368506242C5B35BA859077BA8810147F3DDE200301270364C514";

    public override string Algorithm => "haval-224-4";
}

public class Haval_224_5 : Hash
{
    public override string HashString => "FC2D1B6F27FB775D8E7030715AF85B646239C9D9D675CCFF309B49B7";

    public override string EmptyStringHash => "4A0513C032754F5582A758D35917AC9ADF3854219B39E3AC77D1837E";

    public override string StartPartStringHash => "41C1B8A6AC60949AA2A50313F19D100910881BBF0BC5761F88CDEBC6";

    public override string MiddlePartStringHash => "DAB55FE059D3DBCACF9E8C5A55C21D850391582CB2E4831AAA4E75D1";

    public override string TrailPartStringHash => "458827F727F77C599B452FDBC657245CB8226B4487F9987BD755E1DA";

    public override string Algorithm => "haval-224-5";
}

public class Haval_256_3 : Hash
{
    public override string HashString => "E3891CB6FD1A883A1AE723F13BA336F586FA8C10506C4799C209D10113675BC1";

    public override string EmptyStringHash => "4F6938531F0BC8991F62DA7BBD6F7DE3FAD44562B8C6F4EBF146D5B4E46F7C17";

    public override string StartPartStringHash => "F96418428C992DAB2139CFDB82D89725A192AB53F1F4563D59C0473A15B3418B";

    public override string MiddlePartStringHash => "A1055E7620768718DC9635D0358F3E4AF845F596C0BAED6A1BF0132A33F0F59A";

    public override string TrailPartStringHash => "066DEC0561FD9E2E89A24BC2DE241B2CA099AD5B360C33876F84B262631A4DAC";

    public override string Algorithm => "haval-256-3";
}

public class Haval_256_4 : Hash
{
    public override string HashString => "A16D7FCD48CED7B612FF2C35D78241EB89A752EFF2931647A32C2C3C22F8D747";

    public override string EmptyStringHash => "C92B2E23091E80E375DADCE26982482D197B1A2521BE82DA819F8CA2C579B99B";

    public override string StartPartStringHash => "50AA70038496D65EF6DA866025B31EF493FE33DC5289B615EA3FCA9442705146";

    public override string MiddlePartStringHash => "72AEF38030403F9143002BF1FF8BFC393B0A51A60B27F3C331DCB844A37D1EFC";

    public override string TrailPartStringHash => "8DD972314909F89C8C41026E53288C126FCA5762BF8530028B47790C6224A86F";

    public override string Algorithm => "haval-256-4";
}

public class Haval_256_5 : Hash
{
    public override string HashString => "386DBED5748A4B9E9409D8CE94ACFE8DF324A166EAC054E9817F85F7AEC8AED5";

    public override string EmptyStringHash => "BE417BB4DD5CFB76C7126F4F8EEB1553A449039307B1A3CD451DBFDC0FBBE330";

    public override string StartPartStringHash => "95CCCBF651A772BCE1270C14C262292E973362C06B871D2FA1DDA092DFC2908D";

    public override string MiddlePartStringHash => "2D7584D413364CB958B63D74B4972B97FC3E1154A302D93782C19E49489B964F";

    public override string TrailPartStringHash => "F119285F0556724635892BA10F40400C0F7140905A4A65D28F51063B3518EFFD";

    public override string Algorithm => "haval-256-5";
}

public class Edonr256 : Hash
{
    public override string HashString => "2DBADC39B5189B24479A766F87AC68DA5CB0C0AFF5D692DF3CECAB7B4F423CF1";

    public override string EmptyStringHash => "86E7C84024C55DBDC9339B395C95E88DB8F781719851AD1D237C6E6A8E370B80";

    public override string StartPartStringHash => "8064DFEEC1658E454F7DB8EBFC6AE11F6D9F65552CD4546C765787DF22863419";

    public override string MiddlePartStringHash => "24849CD6594AD41995A3B20193B066F56AB89416770B57A24916AE93EA6050D6";

    public override string TrailPartStringHash => "84EDA6162ADF8131F6B2276750D8DDCDD1679079EF4340186B2C5DC5DBA291C1";

    public override string Algorithm => "edonr256";
}

public class Edonr512 : Hash
{
    public override string HashString => "9A40FA8740E3E0E6475B83BABF1B78B1A38AC3F8DB081723C53E611F2513D68C52BDF641BCC856D7321ACE59FC5181ECC0D5CA6A311D7DF4C7FA80CE4DF8FBA5";

    public override string EmptyStringHash => "C7AFBDF3E5B4590EB0B25000BF83FB16D4F9B722EE7F9A2DC2BD382035E8EE38D6F6F15C7B8EEC85355AC59AF989799950C64557EAB0E687D0FCBDBA90AE9704";

    public override string StartPartStringHash => "4AB73ED70C25F76504DA7B917D3D8952C0B28E4F46FA9BEF1B986786BFE9AD0E1FCCDBB3FDAC2C1C293B97E73887660DF1E97F91FCC3FDF8502A9450ED22C6F2";

    public override string MiddlePartStringHash => "043AF8755799C65A97B94A6BA13FE1A1F92E2A0A4558664068DA8F04213B350D8F3CD9666C67DE8421DF5CC0B4B350D29A985EF6A7E511E6655FA0F8ECE1437A";

    public override string TrailPartStringHash => "0D1949018A6CB9A6E6C6CDAD99A1A59BB085DF4C879B06EAE101C7F4BBBB418AFDD437F207103E4981145591600E0A0CA7E2055B64466290180FC0045C86867D";

    public override string Algorithm => "edonr512";
}

public class Ntlm : Hash
{
    public override string HashString => "3DBDE697D71690A769204BEB12283678";

    public override string EmptyStringHash => "31D6CFE0D16AE931B73C59D7E0C089C0";

    public override string StartPartStringHash => "588FEB889288FB953B5F094D47D1565C";

    public override string MiddlePartStringHash => "8F33E2EBE5960B8738D98A80363786B0";

    public override string TrailPartStringHash => "FB59FE2EBEA80EC80458FF533094884C";

    public override string Algorithm => "ntlm";
}

public class Sha_3_224 : Hash
{
    public override string HashString => "602BDC204140DB016BEE5374895E5568CE422FABE17E064061D80097";

    public override string EmptyStringHash => "6B4E03423667DBB73B6E15454F0EB1ABD4597F9A1B078E3F5B5A6BC7";

    public override string StartPartStringHash => "95A8F823A2E12C1C9D6BE7378BA7BF29AAF9345C4CAA20C7405C8464";

    public override string MiddlePartStringHash => "F3FF4F073ED24D62051C8D7BB73418B95DB2F6FF9E4441AF466F6D98";

    public override string TrailPartStringHash => "71A022FC02222D9214AEE3641BBFD35A706F3E66975A1F949A80ABC3";

    public override string Algorithm => "sha-3-224";
}

public class Sha_3_256 : Hash
{
    public override string HashString => "A03AB19B866FC585B5CB1812A2F63CA861E7E7643EE5D43FD7106B623725FD67";

    public override string EmptyStringHash => "A7FFC6F8BF1ED76651C14756A061D662F580FF4DE43B49FA82D80A4B80F8434A";

    public override string StartPartStringHash => "1A9A118CB653759C3FCB3BD5060E6F9910C8C27008DD11FE4315F4635C9CAA98";

    public override string MiddlePartStringHash => "B1B1BD1ED240B1496C81CCF19CECCF2AF6FD24FAC10AE42023628ABBE2687310";

    public override string TrailPartStringHash => "39604BDFA135910DE937CD3CA347347A1E22C735877C21591D29FE8D2B5844F7";

    public override string Algorithm => "sha-3-256";
}

public class Sha_3_384 : Hash
{
    public override string HashString => "9BD942D1678A25D029B114306F5E1DAE49FE8ABEEACD03CFAB0F156AA2E363C988B1C12803D4A8C9BA38FDC873E5F007";

    public override string EmptyStringHash => "0C63A75B845E4F7D01107D852E4C2485C51A50AAAA94FC61995E71BBEE983A2AC3713831264ADB47FB6BD1E058D5F004";

    public override string StartPartStringHash => "8AD2282A10C5690BF8D59DADD7DCF08A42E3AE6339548848AF4A9DCD274FE5C023243EB34A2DFBE0EC0A13AB8DF2A06C";

    public override string MiddlePartStringHash => "39773563A8FC5C19BA80F0DC0F57BF49BA0E804ABE8E68A1ED067252C30EF499D54AB4EB4E8F4CFA2CFAC6C83798997E";

    public override string TrailPartStringHash => "1618C8E3044A1D03B8AD0088EFCA5CFCD8B30FC99E5C8FB7EF1FEF368C196D2F14FCEC4A5EF074B0D7D145D98573E6CD";

    public override string Algorithm => "sha-3-384";
}

public class Sha_3_512 : Hash
{
    public override string HashString => "48C8947F69C054A5CAA934674CE8881D02BB18FB59D5A63EEADDFF735B0E9801E87294783281AE49FC8287A0FD86779B27D7972D3E84F0FA0D826D7CB67DFEFC";

    public override string EmptyStringHash => "A69F73CCA23A9AC5C8B567DC185A756E97C982164FE25859E0D1DCC1475C80A615B2123AF1F5F94C11E3E9402C3AC558F500199D95B6D3E301758586281DCD26";

    public override string StartPartStringHash => "F235C129089233CE3C9C85F1D1554B9CB21952B27E0765BCBCF75D550DD4D2874E546889DA5C44DB9C066E05E268F4742D672889FF62FB9CB18A3D1B57F00658";

    public override string MiddlePartStringHash => "564E1971233E098C26D412F2D4E652742355E616FED8BA88FC9750F869AAC1C29CB944175C374A7B6769989AA7A4216198EE12F53BF7827850DFE28540587A97";

    public override string TrailPartStringHash => "4F1466999E95B9767883209830AB4E0AB1CF70CD0FC8D18A24EE45EBF9C9CFE691808DCFE3FC1B2EFE557A243303960C73F9825AD72F85A3312271B3FD64F7B6";

    public override string Algorithm => "sha-3-512";
}

public class Sha_3k_224 : Hash
{
    public override string HashString => "5C52615361CE4C5469F9D8C90113C7A543A4BF43490782D291CB32D8";

    public override string EmptyStringHash => "F71837502BA8E10837BDD8D365ADB85591895602FC552B48B7390ABD";

    public override string StartPartStringHash => "7CD0E0471CFFDBC4CCD3038693562640DC0CA7767C5ADFAC5FDE0B62";

    public override string MiddlePartStringHash => "3A8BC59EB4ED4B22328DB6C9476E63E7E76E2411225D9B5304E42807";

    public override string TrailPartStringHash => "3D0C4C221CA61E6E560667EBB53D6214D060A9E1230DAADEC26DC88C";

    public override string Algorithm => "sha-3k-224";
}

public class Sha_3k_256 : Hash
{
    public override string HashString => "64E604787CBF194841E7B68D7CD28786F6C9A0A3AB9F8B0A0E87CB4387AB0107";

    public override string EmptyStringHash => "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470";

    public override string StartPartStringHash => "7F8B6B088B6D74C2852FC86C796DCA07B44EED6FB3DAF5E6B59F7C364DB14528";

    public override string MiddlePartStringHash => "AD7C5BEF027816A800DA1736444FB58A807EF4C9603B7848673F7E3A68EB14A5";

    public override string TrailPartStringHash => "1572B593C53D839D80004AA4B8C51211864104F06ACE9E22BE9C4365B50655EA";

    public override string Algorithm => "sha-3k-256";
}

public class Sha_3k_384 : Hash
{
    public override string HashString => "7DD34CCAAE92BFC7EB541056D200DB23B6BBEEFE95BE0D2BB43625113361906F0AFC701DBEF1CFB615BF98B1535A84C1";

    public override string EmptyStringHash => "2C23146A63A29ACF99E73B88F8C24EAA7DC60AA771780CCC006AFBFA8FE2479B2DD2B21362337441AC12B515911957FF";

    public override string StartPartStringHash => "999F79C5E691445900D0E3411AFB7F7FB64A751514F29F2B11DBFD990D2D43D4782F10CAF90B4C6F613768BD2101B3C7";

    public override string MiddlePartStringHash => "D59D5976E48498472B1B55A5989F1C88768EC47A901193C9FB81F05394A996BF8E4D7DD9EBF9EE94E32EFF92C3E4EA9C";

    public override string TrailPartStringHash => "6321A9C06F0C1A15D21A314AFEE8585CB2B569EDFF920C79DEA40978AFACD4AB8B73D4D70480D31E2A5947E98C862C06";

    public override string Algorithm => "sha-3k-384";
}

public class Sha_3k_512 : Hash
{
    public override string HashString => "8CA32D950873FD2B5B34A7D79C4A294B2FD805ABE3261BEB04FAB61A3B4B75609AFD6478AA8D34E03F262D68BB09A2BA9D655E228C96723B2854838A6E613B9D";

    public override string EmptyStringHash => "0EAB42DE4C3CEB9235FC91ACFFE746B29C29A8C366B7C60E4E67C466F36A4304C00FA9CAF9D87976BA469BCBE06713B435F091EF2769FB160CDAB33D3670680E";

    public override string StartPartStringHash => "AA42ACA73BD7F8A17E987F281422B266E44F0DE1615D2D393C620C8C5A2C80B4F06178C8455BF98179603F2F1BCB30B2559F282C799E40533B0665F97A2A706A";

    public override string MiddlePartStringHash => "AC3B6998AC9C5E2C7EE8330010A7B0F87AC9DEE7EA547D4D8CD00AB7AD1BD5F57F80AF2BA711A9EB137B4E83B503D24CD7665399A48734D47FFF324FB74551E2";

    public override string TrailPartStringHash => "520B0637CA42F1F6380322E2DFB810E9E514679FCB2B982A6DA237EDEB6DD102B58C5853ECE024EDDF73972EC74C5585433AB9A28FAB9851054ECD63AD913522";

    public override string Algorithm => "sha-3k-512";
}

public class Blake2b : Hash
{
    public override string HashString => "E64CB91C7C1819BDCDA4DCA47A2AAE98E737DF75DDB0287083229DC0695064616DF676A0C95AE55109FE0A27BA9DEE79EA9A5C9D90CCEB0CF8AE80B4F61AB4A3";

    public override string EmptyStringHash => "786A02F742015903C6C6FD852552D272912F4740E15847618A86E217F71F5419D25E1031AFEE585313896444934EB04B903A685B1448B755D56F701AFE9BE2CE";

    public override string StartPartStringHash => "B7A5A0F0FB0C4A128B8A3E042FC860775D68D825BB3BF180479D0E12B1884E2652FE51DDB9C991B73824FC15609D82CB1CC19053DB7DC7637288091F6027BBCE";

    public override string MiddlePartStringHash => "C5FACA15AC2F93578B39EF4B6BBB871BDEDCE4DDD584FD31F0BB66FADE3947E6BB1353E562414ED50638A8829FF3DACCAC7EF4A50ACEE72A5384BA9AEB604FC9";

    public override string TrailPartStringHash => "08949F758439C6293FE5924DEFAF3E32BB79B9A93C1331F019C51B386557A9412B27F5A60A80BFA1F524C0D0C2E1F63C5B93D108A9A3AF8CDB7FC87C765FCA3F";

    public override string Algorithm => "blake2b";
}

public class Blake2s : Hash
{
    public override string HashString => "E906644AD861B58D47500E6C636EE3BF4CB4BB00016BB352B1D2D03D122C1605";

    public override string EmptyStringHash => "69217A3079908094E11121D042354A7C1F55B6482CA1A51E1B250DFD1ED0EEF9";

    public override string StartPartStringHash => "109A7947E24479DD8391DF82EB9AF87D135DE808EDDA86E5B9C0F78D7D05B170";

    public override string MiddlePartStringHash => "CD7AEC459FB9C9FD67D89E6B733C394DD0503DF3AB3D08E80894C9A4A14D086D";

    public override string TrailPartStringHash => "C974D441F2D4398B0F1E1F2CDCD0AD1181773A2F7108BB75649538C24FD89B20";

    public override string Algorithm => "blake2s";
}

public class Blake3 : Hash
{
    public override string HashString => "B3D4F8803F7E24B8F389B072E75477CDBCFBE074080FB5E500E53E26E054158E";

    public override string EmptyStringHash => "AF1349B9F5F9A1A6A0404DEA36DCC9499BCB25C9ADC112B7CC9A93CAE41F3262";

    public override string StartPartStringHash => "B944A0A3B20CF5927E594FF306D256D16CD5B0BA3E27B3285F40D7EF5E19695B";

    public override string MiddlePartStringHash => "813E9B729141E7F385AFA0A2D0DF3E6C3789E427FFE4AEEF566A565BC8F2FE3D";

    public override string TrailPartStringHash => "E0812C6818E340ABBE3C63CCE5C52CACB70758C0C002CF1A85BC3C9A806EF522";

    public override string Algorithm => "blake3";
}
