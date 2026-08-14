#define GPU_ATTEMPT_SIZE 16
#define BLOCK_LEN 64
#define HASH_LEN 20
#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))

static void prrmd160_compress(uint* state, const uchar* block) {
    uint X[16];
    for (int j = 0; j < 16; j++) {
        const int i = j * 4;
        X[j] = (uint)(block[i + 0])
            | ((uint)(block[i + 1]) << 8)
            | ((uint)(block[i + 2]) << 16)
            | ((uint)(block[i + 3]) << 24);
    }

    uint al = state[0], ar = state[0];
    uint bl = state[1], br = state[1];
    uint cl = state[2], cr = state[2];
    uint dl = state[3], dr = state[3];
    uint el = state[4], er = state[4];
    uint temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[0] + 0x00000000u, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[5] + 0x50A28BE6u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[1] + 0x00000000u, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[14] + 0x50A28BE6u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[2] + 0x00000000u, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[7] + 0x50A28BE6u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[3] + 0x00000000u, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[0] + 0x50A28BE6u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[4] + 0x00000000u, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[9] + 0x50A28BE6u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[5] + 0x00000000u, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[2] + 0x50A28BE6u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[6] + 0x00000000u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[11] + 0x50A28BE6u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[7] + 0x00000000u, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[4] + 0x50A28BE6u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[8] + 0x00000000u, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[13] + 0x50A28BE6u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[9] + 0x00000000u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[6] + 0x50A28BE6u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[10] + 0x00000000u, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[15] + 0x50A28BE6u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[11] + 0x00000000u, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[8] + 0x50A28BE6u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[12] + 0x00000000u, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[1] + 0x50A28BE6u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[13] + 0x00000000u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[10] + 0x50A28BE6u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[14] + 0x00000000u, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[3] + 0x50A28BE6u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ cl ^ dl) + X[15] + 0x00000000u, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ (cr | ~dr)) + X[12] + 0x50A28BE6u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[7] + 0x5A827999u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[6] + 0x5C4DD124u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[4] + 0x5A827999u, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[11] + 0x5C4DD124u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[13] + 0x5A827999u, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[3] + 0x5C4DD124u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[1] + 0x5A827999u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[7] + 0x5C4DD124u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[10] + 0x5A827999u, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[0] + 0x5C4DD124u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[6] + 0x5A827999u, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[13] + 0x5C4DD124u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[15] + 0x5A827999u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[5] + 0x5C4DD124u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[3] + 0x5A827999u, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[10] + 0x5C4DD124u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[12] + 0x5A827999u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[14] + 0x5C4DD124u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[0] + 0x5A827999u, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[15] + 0x5C4DD124u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[9] + 0x5A827999u, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[8] + 0x5C4DD124u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[5] + 0x5A827999u, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[12] + 0x5C4DD124u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[2] + 0x5A827999u, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[4] + 0x5C4DD124u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[14] + 0x5A827999u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[9] + 0x5C4DD124u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[11] + 0x5A827999u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[1] + 0x5C4DD124u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & cl) | (~bl & dl)) + X[8] + 0x5A827999u, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & dr) | (cr & ~dr)) + X[2] + 0x5C4DD124u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[3] + 0x6ED9EBA1u, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[15] + 0x6D703EF3u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[10] + 0x6ED9EBA1u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[5] + 0x6D703EF3u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[14] + 0x6ED9EBA1u, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[1] + 0x6D703EF3u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[4] + 0x6ED9EBA1u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[3] + 0x6D703EF3u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[9] + 0x6ED9EBA1u, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[7] + 0x6D703EF3u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[15] + 0x6ED9EBA1u, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[14] + 0x6D703EF3u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[8] + 0x6ED9EBA1u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[6] + 0x6D703EF3u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[1] + 0x6ED9EBA1u, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[9] + 0x6D703EF3u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[2] + 0x6ED9EBA1u, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[11] + 0x6D703EF3u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[7] + 0x6ED9EBA1u, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[8] + 0x6D703EF3u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[0] + 0x6ED9EBA1u, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[12] + 0x6D703EF3u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[6] + 0x6ED9EBA1u, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[2] + 0x6D703EF3u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[13] + 0x6ED9EBA1u, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[10] + 0x6D703EF3u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[11] + 0x6ED9EBA1u, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[0] + 0x6D703EF3u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[5] + 0x6ED9EBA1u, 7) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[4] + 0x6D703EF3u, 7) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl | ~cl) ^ dl) + X[12] + 0x6ED9EBA1u, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br | ~cr) ^ dr) + X[13] + 0x6D703EF3u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[1] + 0x8F1BBCDCu, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[8] + 0x7A6D76E9u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[9] + 0x8F1BBCDCu, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[6] + 0x7A6D76E9u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[11] + 0x8F1BBCDCu, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[4] + 0x7A6D76E9u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[10] + 0x8F1BBCDCu, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[1] + 0x7A6D76E9u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[0] + 0x8F1BBCDCu, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[3] + 0x7A6D76E9u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[8] + 0x8F1BBCDCu, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[11] + 0x7A6D76E9u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[12] + 0x8F1BBCDCu, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[15] + 0x7A6D76E9u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[4] + 0x8F1BBCDCu, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[0] + 0x7A6D76E9u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[13] + 0x8F1BBCDCu, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[5] + 0x7A6D76E9u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[3] + 0x8F1BBCDCu, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[12] + 0x7A6D76E9u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[7] + 0x8F1BBCDCu, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[2] + 0x7A6D76E9u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[15] + 0x8F1BBCDCu, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[13] + 0x7A6D76E9u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[14] + 0x8F1BBCDCu, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[9] + 0x7A6D76E9u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[5] + 0x8F1BBCDCu, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[7] + 0x7A6D76E9u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[6] + 0x8F1BBCDCu, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[10] + 0x7A6D76E9u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + ((bl & dl) | (cl & ~dl)) + X[2] + 0x8F1BBCDCu, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + ((br & cr) | (~br & dr)) + X[14] + 0x7A6D76E9u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[4] + 0xA953FD4Eu, 9) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[12] + 0x00000000u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[0] + 0xA953FD4Eu, 15) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[15] + 0x00000000u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[5] + 0xA953FD4Eu, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[10] + 0x00000000u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[9] + 0xA953FD4Eu, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[4] + 0x00000000u, 9) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[7] + 0xA953FD4Eu, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[1] + 0x00000000u, 12) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[12] + 0xA953FD4Eu, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[5] + 0x00000000u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[2] + 0xA953FD4Eu, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[8] + 0x00000000u, 14) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[10] + 0xA953FD4Eu, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[7] + 0x00000000u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[14] + 0xA953FD4Eu, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[6] + 0x00000000u, 8) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[1] + 0xA953FD4Eu, 12) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[2] + 0x00000000u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[3] + 0xA953FD4Eu, 13) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[13] + 0x00000000u, 6) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[8] + 0xA953FD4Eu, 14) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[14] + 0x00000000u, 5) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[11] + 0xA953FD4Eu, 11) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[0] + 0x00000000u, 15) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[6] + 0xA953FD4Eu, 8) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[3] + 0x00000000u, 13) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[15] + 0xA953FD4Eu, 5) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[9] + 0x00000000u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = ROTL32(0U + al + (bl ^ (cl | ~dl)) + X[13] + 0xA953FD4Eu, 6) + el;
    al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = temp;
    temp = ROTL32(0U + ar + (br ^ cr ^ dr) + X[11] + 0x00000000u, 11) + er;
    ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = temp;

    temp = 0U + state[1] + cl + dr;
    state[1] = 0U + state[2] + dl + er;
    state[2] = 0U + state[3] + el + ar;
    state[3] = 0U + state[4] + al + br;
    state[4] = 0U + state[0] + bl + cr;
    state[0] = temp;
}

static void prrmd160_hash(const uchar* message, uint len, uchar* hash) {
    uint state[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    uchar block[BLOCK_LEN];
    for (uint __i = 0; __i < BLOCK_LEN; ++__i) block[__i] = 0;
    for (uint __i = 0; __i < len; ++__i) block[__i] = message[__i];
    block[len] = 0x80;
    const ulong bitlen = (ulong)(len) << 3;
    block[BLOCK_LEN - 8] = (uchar)(bitlen);
    block[BLOCK_LEN - 7] = (uchar)(bitlen >> 8);
    block[BLOCK_LEN - 6] = (uchar)(bitlen >> 16);
    block[BLOCK_LEN - 5] = (uchar)(bitlen >> 24);
    block[BLOCK_LEN - 4] = (uchar)(bitlen >> 32);
    block[BLOCK_LEN - 3] = (uchar)(bitlen >> 40);
    block[BLOCK_LEN - 2] = (uchar)(bitlen >> 48);
    block[BLOCK_LEN - 1] = (uchar)(bitlen >> 56);
    prrmd160_compress(state, block);
    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = (uchar)(state[i >> 2] >> ((i & 3) << 3));
}

static int prrmd160_compare(__global const uchar* k_hash, uchar* password, const int length) {
  uchar hash[20];
  prrmd160_hash(password, (uint)length, hash);
  for (int i = 0; i < 20; ++i) {
    if (hash[i] != k_hash[i]) return 0;
  }
  return 1;
}

__kernel void prrmd160_kernel(__global uchar* result,
                          __global const uchar* k_dict,
                          __global const uchar* k_hash,
                          __global int* g_found,
                          const ulong start,
                          const uint count,
                          const uint pass_len,
                          const uint dict_length,
                          const uint min_len) {
  const uint ix = get_global_id(0);
  if (ix >= count || *g_found) return;
  ulong idx = start + (ulong)ix;
  uchar attempt[GPU_ATTEMPT_SIZE];
  for (int pos = (int)pass_len - 1; pos >= 0; --pos) {
    attempt[pos] = k_dict[idx % dict_length];
    idx /= dict_length;
  }
  for (uint i = 0; i < dict_length; ++i) {
    attempt[pass_len] = k_dict[i];
    if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {
      if (*g_found) return;
      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 1u))) {
        for (uint k = 0; k < pass_len + 1u; ++k) result[k] = attempt[k];
        result[pass_len + 1u] = 0;
        *g_found = 1;
        return;
      }
    }
    if (pass_len + 2u < min_len) continue;
    for (uint j = 0; j < dict_length; ++j) {
      attempt[pass_len + 1u] = k_dict[j];
      if (*g_found) return;
      if (prrmd160_compare(k_hash, attempt, (int)(pass_len + 2u))) {
        for (uint k = 0; k < pass_len + 2u; ++k) result[k] = attempt[k];
        result[pass_len + 2u] = 0;
        *g_found = 1;
        return;
      }
    }
  }
}
