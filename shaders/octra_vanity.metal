/*
 * octra_vanity.metal — Octra vanity address miner (Metal port of octra_vanity.cu)
 *
 * Pipeline per thread:
 *   seed[32] -> SHA-512 (first 32 bytes only) -> clamp -> Ed25519 scalarbase
 *   -> pack -> pk[32] -> SHA-256(pk) -> base58 -> "oct" + base58
 *
 * Original CUDA implementation: https://github.com/0x02937/octra_vanity
 */

#include <metal_stdlib>
using namespace metal;

typedef uchar  u8;
typedef uint   u32;
typedef ulong  u64;
typedef long   i64;

/* SHA-512 round constants (FIPS 180-4) */
constant u64 C512K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

constant u32 C256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

/* GF(2^255-19) Ed25519 base-point related constants */
constant int GF_D2[16] = {
    0xf159,0x26b2,0x9b94,0xebd6,0xb156,0x8283,0x149a,0x00e0,
    0xd130,0xeef3,0x80f2,0x198e,0xfce7,0x56df,0xd9dc,0x2406
};
constant int GF_BY_MINUS_BX[16] = {-28354,-10431,14598,-25328,-16716,-11967,-24826,-710,-30198,-38768,-31691,-23102,4712,-26376,12179,17661};
constant int GF_BY_PLUS_BX[16]  = {80754,62859,37830,77756,69144,64395,77254,53138,82626,91196,84119,75530,47716,78804,40249,34767};
constant int GF_BT_D2[16]       = {43605,34682,4613,43977,50334,52394,59427,9945,22924,56643,32203,23067,26024,40716,31592,61201};

constant char B58[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* ---- byteswap helper (replaces CUDA __byte_perm(x, 0, 0x0123)) ---- */
static inline u32 bswap32(u32 x) {
    return ((x >> 24) & 0x000000ffu) |
           ((x >>  8) & 0x0000ff00u) |
           ((x <<  8) & 0x00ff0000u) |
           ((x << 24) & 0xff000000u);
}

/* SHA-512 over 32 bytes (passed as 8 LE u32 words) — only first 32 bytes of output */
static void sha512_half_32(thread const u32 *data, thread u32 *out32)
{
    u64 h[8] = {
        0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
        0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
        0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
    };
    u64 w[16];

    for (int i = 0; i < 4; i++) {
        u64 hi = bswap32(data[i*2]);
        u64 lo = bswap32(data[i*2+1]);
        w[i] = (hi << 32) | lo;
    }
    w[4]  = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) w[i] = 0UL;
    w[15] = 256UL;

    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    for (int i = 0; i < 16; i++) {
        u64 S1 = ((e>>14)|(e<<50)) ^ ((e>>18)|(e<<46)) ^ ((e>>41)|(e<<23));
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = hv + S1 + ch + C512K[i] + w[i];
        u64 S0 = ((a>>28)|(a<<36)) ^ ((a>>34)|(a<<30)) ^ ((a>>39)|(a<<25));
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    for (int i = 16; i < 80; i++) {
        u64 v15 = w[(i+ 1)&15];
        u64 v2  = w[(i+14)&15];
        u64 s0  = ((v15>>1)|(v15<<63)) ^ ((v15>>8)|(v15<<56)) ^ (v15>>7);
        u64 s1  = ((v2>>19)|(v2<<45)) ^ ((v2>>61)|(v2<<3))   ^ (v2>>6);
        w[i&15] += s0 + w[(i+9)&15] + s1;
        u64 wi  = w[i&15];
        u64 S1  = ((e>>14)|(e<<50)) ^ ((e>>18)|(e<<46)) ^ ((e>>41)|(e<<23));
        u64 ch  = (e & f) ^ (~e & g);
        u64 t1  = hv + S1 + ch + C512K[i] + wi;
        u64 S0  = ((a>>28)|(a<<36)) ^ ((a>>34)|(a<<30)) ^ ((a>>39)|(a<<25));
        u64 mj  = (a & b) ^ (a & c) ^ (b & c);
        u64 t2  = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;

    out32[0] = ((u32)(h[0]>>56)) | (((u32)(h[0]>>48)&0xff)<<8) | (((u32)(h[0]>>40)&0xff)<<16) | (((u32)(h[0]>>32)&0xff)<<24);
    out32[1] = ((u32)(h[0]>>24)&0xff) | (((u32)(h[0]>>16)&0xff)<<8) | (((u32)(h[0]>>8)&0xff)<<16) | (((u32)(h[0])&0xff)<<24);
    out32[2] = ((u32)(h[1]>>56)) | (((u32)(h[1]>>48)&0xff)<<8) | (((u32)(h[1]>>40)&0xff)<<16) | (((u32)(h[1]>>32)&0xff)<<24);
    out32[3] = ((u32)(h[1]>>24)&0xff) | (((u32)(h[1]>>16)&0xff)<<8) | (((u32)(h[1]>>8)&0xff)<<16) | (((u32)(h[1])&0xff)<<24);
    out32[4] = ((u32)(h[2]>>56)) | (((u32)(h[2]>>48)&0xff)<<8) | (((u32)(h[2]>>40)&0xff)<<16) | (((u32)(h[2]>>32)&0xff)<<24);
    out32[5] = ((u32)(h[2]>>24)&0xff) | (((u32)(h[2]>>16)&0xff)<<8) | (((u32)(h[2]>>8)&0xff)<<16) | (((u32)(h[2])&0xff)<<24);
    out32[6] = ((u32)(h[3]>>56)) | (((u32)(h[3]>>48)&0xff)<<8) | (((u32)(h[3]>>40)&0xff)<<16) | (((u32)(h[3]>>32)&0xff)<<24);
    out32[7] = ((u32)(h[3]>>24)&0xff) | (((u32)(h[3]>>16)&0xff)<<8) | (((u32)(h[3]>>8)&0xff)<<16) | (((u32)(h[3])&0xff)<<24);
}

/* SHA-256 over exactly 32 bytes */
static void sha256_32_from32(thread const u32 *data, thread u32 *out)
{
    u32 h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    u32 w[16];

    for (int i = 0; i < 8; i++) w[i] = bswap32(data[i]);
    w[8]  = 0x80000000u;
    for (int i = 9; i < 15; i++) w[i] = 0u;
    w[15] = 256u;

    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    for (int i = 0; i < 16; i++) {
        u32 S1 = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = hv + S1 + ch + C256K[i] + w[i];
        u32 S0 = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    for (int i = 16; i < 64; i++) {
        u32 v15 = w[(i+ 1)&15];
        u32 v2  = w[(i+14)&15];
        u32 s0  = ((v15>>7)|(v15<<25)) ^ ((v15>>18)|(v15<<14)) ^ (v15>>3);
        u32 s1  = ((v2>>17)|(v2<<15))  ^ ((v2>>19)|(v2<<13))   ^ (v2>>10);
        w[i&15] += s0 + w[(i+9)&15] + s1;
        u32 wi  = w[i&15];
        u32 S1  = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
        u32 ch  = (e & f) ^ (~e & g);
        u32 t1  = hv + S1 + ch + C256K[i] + wi;
        u32 S0  = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
        u32 mj  = (a & b) ^ (a & c) ^ (b & c);
        u32 t2  = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hv;

    for (int i = 0; i < 8; i++) out[i] = h[i];
}

/* ---- GF(2^255-19) field arithmetic (16 limbs, ~16 bits each) ---- */
static void gf_set(thread int *r, thread const int *a) {
    for (int i = 0; i < 16; i++) r[i] = a[i];
}
static void gf_zero(thread int *r) {
    for (int i = 0; i < 16; i++) r[i] = 0;
}
static void gf_one(thread int *r) { gf_zero(r); r[0] = 1; }
static void gf_add(thread int *o, thread const int *a, thread const int *b) {
    for (int i = 0; i < 16; i++) o[i] = a[i] + b[i];
}
static void gf_sub(thread int *o, thread const int *a, thread const int *b) {
    for (int i = 0; i < 16; i++) o[i] = a[i] - b[i];
}

/* Carry-reduce: normalise limbs to <= 16 bits after add/sub. */
static void gf_car(thread int *o) {
    int c;
    for (int i = 0; i < 16; i++) {
        o[i] += (1<<16);
        c = o[i] >> 16;
        o[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
        o[i] -= c << 16;
    }
}

/* Constant-time conditional swap */
static void gf_sel(thread int *p, thread int *q, int b) {
    int t, c = ~(b - 1);
    for (int i = 0; i < 16; i++) {
        t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

/* Field multiplication (accumulates in i64, two carry passes). */
static void gf_mul(thread int *o, thread const int *a, thread const int *b) {
    i64 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            i64 p = (i64)a[i] * (i64)b[j];
            if (i + j < 16) t[i + j]      += p;
            else            t[i + j - 16] += 38L * p;
        }
    }
    for (int pass = 0; pass < 2; pass++) {
        i64 c;
        for (int i = 0; i < 16; i++) {
            t[i] += (1L<<16);
            c = t[i] >> 16;
            t[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
            t[i] -= c << 16;
        }
    }
    for (int i = 0; i < 16; i++) o[i] = (int)t[i];
}

/* Same but b is in constant address space (for fixed base-point coefficients). */
static void gf_mul_const(thread int *o, thread const int *a, constant int *b) {
    i64 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            i64 p = (i64)a[i] * (i64)b[j];
            if (i + j < 16) t[i + j]      += p;
            else            t[i + j - 16] += 38L * p;
        }
    }
    for (int pass = 0; pass < 2; pass++) {
        i64 c;
        for (int i = 0; i < 16; i++) {
            t[i] += (1L<<16);
            c = t[i] >> 16;
            t[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
            t[i] -= c << 16;
        }
    }
    for (int i = 0; i < 16; i++) o[i] = (int)t[i];
}

static void gf_sqr(thread int *o, thread const int *a) {
    i64 t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = i; j < 16; j++) {
            i64 p = (i64)a[i] * (i64)a[j];
            if (i != j) p += p;
            int k = i + j;
            if (k < 16) t[k]      += p;
            else        t[k - 16] += 38L * p;
        }
    }
    for (int pass = 0; pass < 2; pass++) {
        i64 c;
        for (int i = 0; i < 16; i++) {
            t[i] += (1L<<16);
            c = t[i] >> 16;
            t[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
            t[i] -= c << 16;
        }
    }
    for (int i = 0; i < 16; i++) o[i] = (int)t[i];
}

/* Inversion via Fermat */
static void gf_inv(thread int *o, thread const int *inp) {
    int c[16];
    gf_set(c, inp);
    for (int a = 253; a >= 0; a--) {
        gf_sqr(c, c);
        if (a != 2 && a != 4) gf_mul(c, c, inp);
    }
    gf_set(o, c);
}

/* Pack 16 limbs to 32 LE bytes */
static void gf_pack25519(thread u8 *o, thread const int *n) {
    int b;
    int m[16], t[16];
    gf_set(t, n);
    gf_car(t); gf_car(t); gf_car(t);
    for (int j = 0; j < 2; j++) {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; i++) {
            m[i] = t[i] - 0xffff - ((m[i-1]>>16)&1);
            m[i-1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14]>>16)&1);
        b = (m[15]>>16)&1;
        m[14] &= 0xffff;
        gf_sel(t, m, 1-b);
    }
    for (int i = 0; i < 16; i++) {
        o[2*i]   = (u8)(t[i] & 0xff);
        o[2*i+1] = (u8)(t[i] >> 8);
    }
}

static void gf_pack25519_32(thread u32 *o, thread const int *n) {
    int b;
    int m[16], t[16];
    gf_set(t, n);
    gf_car(t); gf_car(t); gf_car(t);
    for (int j = 0; j < 2; j++) {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; i++) {
            m[i] = t[i] - 0xffff - ((m[i-1]>>16)&1);
            m[i-1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14]>>16)&1);
        b = (m[15]>>16)&1;
        m[14] &= 0xffff;
        gf_sel(t, m, 1-b);
    }
    for (int i = 0; i < 8; i++) {
        o[i] = (t[2*i] & 0xffff) | ((t[2*i+1] & 0xffff) << 16);
    }
}

/* Parity bit of (a mod p). Must do the full pack25519-style reduction —
 * carry-reduce alone leaves the value in [0, ~2p), and since p is odd the
 * low bit flips on subtraction. tweetnacl's par25519 does the same.        */
static u8 gf_par(thread const int *a) {
    u8 d[32];
    gf_pack25519(d, a);
    return d[0] & 1;
}

/* Precomputed point entry — packed 32-byte representations of y+x, y-x, t*d2 */
struct PrecompPoint {
    u32 y_plus_x[8];
    u32 y_minus_x[8];
    u32 t_d2[8];
};

static void pack_limb32(thread u32 *out, thread const int *in) {
    u8 bytes[32];
    gf_pack25519(bytes, in);
    for (int i = 0; i < 8; i++) {
        out[i] = ((u32)bytes[i*4]) | ((u32)bytes[i*4+1]<<8) | ((u32)bytes[i*4+2]<<16) | ((u32)bytes[i*4+3]<<24);
    }
}

static void unpack_limb32(thread int *out, device const u32 *in) {
    for (int i = 0; i < 8; i++) {
        u32 v = in[i];
        out[2*i]   = v & 0xffff;
        out[2*i+1] = v >> 16;
    }
}

/* ---- Ed25519 group ops, extended twisted Edwards coords ---- */

/* p ← 2 * p */
static void ed_double(thread int p[4][16]) {
    int A[16], B[16], C[16], E[16];

    gf_sqr(A, p[0]);
    gf_sqr(B, p[1]);
    gf_sqr(C, p[2]);
    gf_add(C, C, C);

    gf_add(E, p[0], p[1]);
    gf_sqr(E, E);

    gf_add(p[3], A, B);
    gf_sub(E, E, p[3]);

    gf_sub(p[2], B, A);
    gf_sub(C, C, p[2]);

    gf_mul(A, E, C);
    gf_mul(B, p[2], p[3]);
    gf_mul(p[0], E, p[3]);
    gf_mul(p[1], C, p[2]);

    gf_set(p[3], p[0]);
    gf_set(p[2], p[1]);
    gf_set(p[0], A);
    gf_set(p[1], B);
}

/* p ← p + base-point (used during precomputation only) */
static void ed_add_base(thread int p[4][16]) {
    int a[16], b[16], c[16], d[16];

    gf_sub(a, p[1], p[0]);
    gf_mul_const(a, a, GF_BY_MINUS_BX);

    gf_add(b, p[0], p[1]);
    gf_mul_const(b, b, GF_BY_PLUS_BX);

    gf_mul_const(c, p[3], GF_BT_D2);

    for (int i = 0; i < 16; i++) d[i] = p[2][i] + p[2][i];

    gf_sub(p[0], b, a);
    gf_add(p[3], b, a);
    gf_sub(p[1], d, c);
    gf_add(p[2], d, c);

    gf_mul(a, p[0], p[1]);
    gf_mul(b, p[3], p[2]);
    gf_mul(c, p[2], p[1]);
    gf_mul(d, p[0], p[3]);

    gf_set(p[0], a);
    gf_set(p[1], b);
    gf_set(p[2], c);
    gf_set(p[3], d);
}

static void ed_pack32(thread u32 *r, thread int p[4][16]) {
    int tx[16], ty[16], zi[16];
    gf_inv(zi, p[2]);
    gf_mul(tx, p[0], zi);
    gf_mul(ty, p[1], zi);
    gf_pack25519_32(r, ty);
    r[7] ^= (u32)gf_par(tx) << 31;
}

/* ---- Precompute table kernel ---- */
/* grid: 32 threadgroups × 256 threads. Each thread writes one PrecompPoint entry. */
kernel void init_table_kernel(
    device PrecompPoint *table          [[buffer(0)]],
    uint                 tgid           [[threadgroup_position_in_grid]],
    uint                 tid_in_tg      [[thread_position_in_threadgroup]])
{
    int i = (int)tgid;       // byte position 0..31
    int j = (int)tid_in_tg;  // byte value    0..255

    u32 s32[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    s32[i/4] = (u32)j << ((i%4)*8);

    int p[4][16];
    gf_zero(p[0]); gf_one(p[1]); gf_one(p[2]); gf_zero(p[3]);

    for (int word = 7; word >= 0; --word) {
        u32 w = s32[word];
        for (int bit = 31; bit >= 0; --bit) {
            ed_double(p);
            if ((w >> bit) & 1u) ed_add_base(p);
        }
    }

    int zi[16];
    gf_inv(zi, p[2]);

    int x[16], y[16], t[16];
    gf_mul(x, p[0], zi);
    gf_mul(y, p[1], zi);
    gf_mul(t, p[3], zi);

    int y_plus_x[16], y_minus_x[16], t_d2[16];
    gf_add(y_plus_x, y, x);
    gf_sub(y_minus_x, y, x);
    gf_mul_const(t_d2, t, GF_D2);

    int idx = i * 256 + j;

    u32 ypx[8], ymx[8], td2[8];
    pack_limb32(ypx, y_plus_x);
    pack_limb32(ymx, y_minus_x);
    pack_limb32(td2, t_d2);
    for (int k = 0; k < 8; k++) {
        table[idx].y_plus_x[k]  = ypx[k];
        table[idx].y_minus_x[k] = ymx[k];
        table[idx].t_d2[k]      = td2[k];
    }
}

/* Scalar-base multiply using precomputed 8-bit window table */
static void ed_scalarbase_table(thread int p[4][16], thread const u32 *s32, device const PrecompPoint *table) {
    gf_zero(p[0]); gf_one(p[1]); gf_one(p[2]); gf_zero(p[3]);

    /* Byte-wise scan over s32 (little-endian) */
    for (int i = 0; i < 32; i++) {
        u32 w = s32[i/4];
        u8 j = (u8)((w >> ((i%4)*8)) & 0xff);
        if (j == 0) continue;

        int idx = i * 256 + j;

        int y_plus_x[16], y_minus_x[16], t_d2[16];
        unpack_limb32(y_plus_x, table[idx].y_plus_x);
        unpack_limb32(y_minus_x, table[idx].y_minus_x);
        unpack_limb32(t_d2, table[idx].t_d2);

        int a[16], b[16], c[16], d[16];

        gf_sub(a, p[1], p[0]);
        gf_mul(a, a, y_minus_x);

        gf_add(b, p[0], p[1]);
        gf_mul(b, b, y_plus_x);

        gf_mul(c, p[3], t_d2);

        for (int k = 0; k < 16; k++) d[k] = p[2][k] + p[2][k];

        gf_sub(p[0], b, a);
        gf_add(p[3], b, a);
        gf_sub(p[1], d, c);
        gf_add(p[2], d, c);

        gf_mul(a, p[0], p[1]);
        gf_mul(b, p[3], p[2]);
        gf_mul(c, p[2], p[1]);
        gf_mul(d, p[0], p[3]);

        gf_set(p[0], a);
        gf_set(p[1], b);
        gf_set(p[2], c);
        gf_set(p[3], d);
    }
}

static void derive_pubkey32(thread const u32 *seed, thread u32 *pk, device const PrecompPoint *table) {
    u32 s32[8];
    sha512_half_32(seed, s32);
    s32[0] &= 0xFFFFFFF8u;
    s32[7] &= 0x7FFFFFFFu;
    s32[7] |= 0x40000000u;

    int p[4][16];
    ed_scalarbase_table(p, s32, table);
    ed_pack32(pk, p);
}

/* ---- Pattern matching ---- */
#define MODE_PREFIX    0
#define MODE_SUFFIX    1
#define MODE_ANYWHERE  2
#define MODE_REP_START 3
#define MODE_REP_END   4
#define MODE_REP_ANY   5

struct VanityCfg {
    int  mode;
    char pat[64];
    int  plen;
    int  rep;
    int  case_insensitive;  /* bool; explicit int to match Swift layout */
    int  rep_bonus;
    int  rep_bonus_mode;
};

/* Base58-encode 32 bytes (passed as 8 LE u32) to a string */
static int b58enc(thread const u32 *t_in, thread char *out) {
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    char buf[48];
    int  blen = 0;

    int leading = 0;
    for (int i = 0; i < 8; i++) {
        if (t_in[i] == 0) leading += 4;
        else { leading += clz(t_in[i]) / 8; break; }
    }

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    while (begin < 8) {
        int new_begin = 8;
        u32 rem = 0;
        for (int i = 0; i < 8; i++) {
            if (i >= begin) {
                u64 cur = ((u64)rem << 32) | t[i];
                t[i] = (u32)(cur / 58);
                rem = (u32)(cur % 58);
                if (t[i] != 0 && new_begin == 8) new_begin = i;
            }
        }
        buf[blen++] = B58[rem];
        begin = new_begin;
    }
    for (int i = 0; i < leading; i++) buf[blen++] = '1';

    for (int i = 0; i < blen; i++) out[i] = buf[blen-1-i];
    out[blen] = '\0';
    return blen;
}

static bool char_match(char c1, char c2, bool ci) {
    if (ci) {
        if (c1 >= 'A' && c1 <= 'Z') c1 ^= 0x20;
        if (c2 >= 'A' && c2 <= 'Z') c2 ^= 0x20;
    }
    return c1 == c2;
}

static bool b58_suffix_match(thread const u32 *t_in, constant VanityCfg *c) {
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for (int i = 0; i < 8; i++) {
        if (t_in[i] == 0) leading += 4;
        else { leading += clz(t_in[i]) / 8; break; }
    }
    if (leading == 32) {
        if (c->plen > 32) return false;
        bool ci = c->case_insensitive != 0;
        for (int i = 0; i < c->plen; i++)
            if (!char_match(c->pat[i], '1', ci)) return false;
        return true;
    }

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    bool ci = c->case_insensitive != 0;
    for (int matched = 0; matched < c->plen; matched++) {
        if (begin >= 8) return false;

        int new_begin = 8;
        u32 rem = 0;
        for (int i = 0; i < 8; i++) {
            if (i >= begin) {
                u64 cur = ((u64)rem << 32) | t[i];
                t[i] = (u32)(cur / 58);
                rem = (u32)(cur % 58);
                if (t[i] != 0 && new_begin == 8) new_begin = i;
            }
        }
        if (!char_match(B58[rem], c->pat[c->plen - 1 - matched], ci)) return false;
        begin = new_begin;
    }
    return true;
}

static bool b58_rep_end_match(thread const u32 *t_in, int rep) {
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for (int i = 0; i < 8; i++) {
        if (t_in[i] == 0) leading += 4;
        else { leading += clz(t_in[i]) / 8; break; }
    }
    if (leading == 32) return rep <= 32;

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    char want = '\0';
    for (int matched = 0; matched < rep; matched++) {
        if (begin >= 8) return false;

        int new_begin = 8;
        u32 rem = 0;
        for (int i = 0; i < 8; i++) {
            if (i >= begin) {
                u64 cur = ((u64)rem << 32) | t[i];
                t[i] = (u32)(cur / 58);
                rem = (u32)(cur % 58);
                if (t[i] != 0 && new_begin == 8) new_begin = i;
            }
        }
        char digit = B58[rem];
        if (matched == 0) want = digit;
        else if (digit != want) return false;
        begin = new_begin;
    }
    return true;
}

static bool b58_rep_any_match(thread const u32 *t_in, int rep) {
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for (int i = 0; i < 8; i++) {
        if (t_in[i] == 0) leading += 4;
        else { leading += clz(t_in[i]) / 8; break; }
    }
    if (leading == 32) return rep <= 32;

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    char prev = '\0';
    int run = 0;

    while (begin < 8) {
        int new_begin = 8;
        u32 rem = 0;
        for (int i = 0; i < 8; i++) {
            if (i >= begin) {
                u64 cur = ((u64)rem << 32) | t[i];
                t[i] = (u32)(cur / 58);
                rem = (u32)(cur % 58);
                if (t[i] != 0 && new_begin == 8) new_begin = i;
            }
        }
        char digit = B58[rem];
        if (run > 0 && digit == prev) run++;
        else { prev = digit; run = 1; }
        if (run >= rep) return true;
        begin = new_begin;
    }
    for (int i = 0; i < leading; i++) {
        if (run > 0 && prev == '1') run++;
        else { prev = '1'; run = 1; }
        if (run >= rep) return true;
    }
    return false;
}

static bool check_rep_bonus(thread const u32 *t_in, constant VanityCfg *c) {
    if (c->rep_bonus <= 0) return false;
    if (c->rep_bonus_mode == 2) return b58_rep_end_match(t_in, c->rep_bonus);
    if (c->rep_bonus_mode == 3) return b58_rep_any_match(t_in, c->rep_bonus);

    /* Prefix mode: encode fully and look at the start of the address */
    u32 t[8];
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for (int i = 0; i < 8; i++) {
        if (t_in[i] == 0) leading += 4;
        else { leading += clz(t_in[i]) / 8; break; }
    }
    if (leading >= c->rep_bonus) return true;

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    char buf[48];
    int blen = 0;
    while (begin < 8) {
        int new_begin = 8;
        u32 rem = 0;
        for (int i = 0; i < 8; i++) {
            if (i >= begin) {
                u64 cur = ((u64)rem << 32) | t[i];
                t[i] = (u32)(cur / 58);
                rem = (u32)(cur % 58);
                if (t[i] != 0 && new_begin == 8) new_begin = i;
            }
        }
        buf[blen++] = B58[rem];
        begin = new_begin;
    }
    for (int i = 0; i < leading; i++) buf[blen++] = '1';

    if (blen < c->rep_bonus) return false;
    char first = buf[blen - 1];
    for (int i = 1; i < c->rep_bonus; i++) {
        if (buf[blen - 1 - i] != first) return false;
    }
    return true;
}

static bool pat_match(thread const char *s, int slen, constant VanityCfg *c) {
    bool ci = c->case_insensitive != 0;
    switch (c->mode) {
    case MODE_PREFIX:
        if (slen < c->plen) return false;
        for (int i = 0; i < c->plen; i++)
            if (!char_match(s[i], c->pat[i], ci)) return false;
        return true;

    case MODE_SUFFIX: {
        if (slen < c->plen) return false;
        int off = slen - c->plen;
        for (int i = 0; i < c->plen; i++)
            if (!char_match(s[off+i], c->pat[i], ci)) return false;
        return true;
    }

    case MODE_ANYWHERE:
        for (int i = 0; i <= slen - c->plen; i++) {
            bool ok = true;
            for (int j = 0; j < c->plen; j++)
                if (!char_match(s[i+j], c->pat[j], ci)) { ok = false; break; }
            if (ok) return true;
        }
        return false;

    case MODE_REP_START:
        if (slen < c->rep) return false;
        for (int i = 1; i < c->rep; i++)
            if (s[i] != s[0]) return false;
        return true;

    case MODE_REP_END: {
        if (slen < c->rep) return false;
        int st = slen - c->rep;
        for (int i = 1; i < c->rep; i++)
            if (s[st+i] != s[st]) return false;
        return true;
    }

    case MODE_REP_ANY:
        for (int i = 0; i <= slen - c->rep; i++) {
            bool ok = true;
            for (int j = 1; j < c->rep; j++)
                if (s[i+j] != s[i]) { ok = false; break; }
            if (ok) return true;
        }
        return false;
    }
    return false;
}

/* ---- Result writers ---- */
struct FoundOut {
    atomic_uint flag;
    uint        _pad;
    u8          seed[32];
    char        addr[64];
};

static void write_found(device FoundOut *out, thread const u32 *seed, thread const char *addr, int alen) {
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(&out->flag, &expected, 1u,
                                              memory_order_relaxed, memory_order_relaxed)) {
        for (int i = 0; i < 8; i++) {
            out->seed[i*4]   = seed[i]        & 0xff;
            out->seed[i*4+1] = (seed[i] >> 8) & 0xff;
            out->seed[i*4+2] = (seed[i] >>16) & 0xff;
            out->seed[i*4+3] = (seed[i] >>24) & 0xff;
        }
        for (int i = 0; i < alen; i++) out->addr[i] = addr[i];
        out->addr[alen] = '\0';
    }
}

/* ---- Debug: derive pk from a fixed seed and dump it (for verification) ---- */
kernel void debug_derive(
    device const u8           *seed_in           [[buffer(0)]],
    device       u8           *pk_out            [[buffer(1)]],
    device const PrecompPoint *table             [[buffer(2)]],
    uint                       tid               [[thread_position_in_grid]])
{
    if (tid != 0) return;
    u32 seed[8];
    for (int i = 0; i < 8; i++) {
        seed[i] = ((u32)seed_in[i*4]) | ((u32)seed_in[i*4+1]<<8) | ((u32)seed_in[i*4+2]<<16) | ((u32)seed_in[i*4+3]<<24);
    }
    u32 pk[8];
    derive_pubkey32(seed, pk, table);
    for (int i = 0; i < 8; i++) {
        pk_out[i*4]   = pk[i]        & 0xff;
        pk_out[i*4+1] = (pk[i] >> 8) & 0xff;
        pk_out[i*4+2] = (pk[i] >>16) & 0xff;
        pk_out[i*4+3] = (pk[i] >>24) & 0xff;
    }
}

/* ---- Generic kernel (prefix / anywhere / rep_start / rep_any) ---- */
kernel void vanity_kernel(
    constant ulong            &base_seed         [[buffer(0)]],
    constant int              &iters_per_thread  [[buffer(1)]],
    constant VanityCfg        *cfg               [[buffer(2)]],
    device   FoundOut         *out_main          [[buffer(3)]],
    device   FoundOut         *out_bonus         [[buffer(4)]],
    device   const PrecompPoint *table           [[buffer(5)]],
    uint                       tid               [[thread_position_in_grid]])
{
    if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

    /* splitmix64 init then xorshift64 per call */
    u64 rng = base_seed + (u64)tid * 6364136223846793005UL + 1442695040888963407UL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9UL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebUL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            char baddr[48];
            int blen = b58enc(pk, baddr);
            write_found(out_bonus, seed, baddr, blen);
        }

        bool matched = false;
        switch (cfg->mode) {
        case MODE_SUFFIX:
            matched = b58_suffix_match(pk, cfg);
            break;
        case MODE_REP_END:
            matched = b58_rep_end_match(pk, cfg->rep);
            break;
        case MODE_REP_ANY:
            matched = b58_rep_any_match(pk, cfg->rep);
            break;
        default: {
            char addr[48];
            int alen = b58enc(pk, addr);
            if (pat_match(addr, alen, cfg)) {
                write_found(out_main, seed, addr, alen);
                return;
            }
            continue;
        }
        }

        if (matched) {
            char addr[48];
            int alen = b58enc(pk, addr);
            write_found(out_main, seed, addr, alen);
            return;
        }
    }
}

/* ---- Specialised suffix kernel ---- */
kernel void vanity_kernel_suffix(
    constant ulong            &base_seed         [[buffer(0)]],
    constant int              &iters_per_thread  [[buffer(1)]],
    constant VanityCfg        *cfg               [[buffer(2)]],
    device   FoundOut         *out_main          [[buffer(3)]],
    device   FoundOut         *out_bonus         [[buffer(4)]],
    device   const PrecompPoint *table           [[buffer(5)]],
    uint                       tid               [[thread_position_in_grid]])
{
    if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

    u64 rng = base_seed + (u64)tid * 6364136223846793005UL + 1442695040888963407UL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9UL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebUL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            char baddr[48];
            int blen = b58enc(pk, baddr);
            write_found(out_bonus, seed, baddr, blen);
        }

        if (!b58_suffix_match(pk, cfg)) continue;

        char addr[48];
        int alen = b58enc(pk, addr);
        write_found(out_main, seed, addr, alen);
        return;
    }
}

/* ---- Specialised rep_end kernel ---- */
kernel void vanity_kernel_rep_end(
    constant ulong            &base_seed         [[buffer(0)]],
    constant int              &iters_per_thread  [[buffer(1)]],
    constant VanityCfg        *cfg               [[buffer(2)]],
    device   FoundOut         *out_main          [[buffer(3)]],
    device   FoundOut         *out_bonus         [[buffer(4)]],
    device   const PrecompPoint *table           [[buffer(5)]],
    uint                       tid               [[thread_position_in_grid]])
{
    if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

    u64 rng = base_seed + (u64)tid * 6364136223846793005UL + 1442695040888963407UL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9UL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebUL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (atomic_load_explicit(&out_main->flag, memory_order_relaxed) != 0u) return;

        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            char baddr[48];
            int blen = b58enc(pk, baddr);
            write_found(out_bonus, seed, baddr, blen);
        }

        if (!b58_rep_end_match(pk, cfg->rep)) continue;

        char addr[48];
        int alen = b58enc(pk, addr);
        write_found(out_main, seed, addr, alen);
        return;
    }
}
