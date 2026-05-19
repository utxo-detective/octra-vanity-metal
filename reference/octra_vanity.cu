/*
 * octra_vanity.cu — Octra vanity address miner (CUDA GPU)
 *
 * Pipeline per thread:
 *   seed[32] → SHA-512 → clamp → Ed25519 scalarbase → pack → pk[32]
 *   → SHA-256(pk) → base58 → "oct" + base58_str
 *
 * Build:
 *   nvcc -O3 -arch=sm_89 octra_vanity.cu -o octra_vanity.exe
 *   (sm_52 Maxwell, sm_61 Pascal, sm_75 Turing, sm_86 Ampere, sm_89 Ada Lovelace)
 *
 * Usage:
 *   octra_vanity.exe --prefix   HELLO       → octHELLO...
 *   octra_vanity.exe --suffix   END         → ...END
 *   octra_vanity.exe --anywhere TEST        → ...TEST...
 *   octra_vanity.exe --rep-start 5          → octXXXXX... (5 same chars)
 *   octra_vanity.exe --rep-end   5          → ...XXXXX
 *   octra_vanity.exe --rep-any   5          → 5 consecutive identical chars
 *
 * Base58 alphabet: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
 * Patterns are case-sensitive and must use only valid base58 characters.
 */

 #include <cuda_runtime.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <stdint.h>
 #include <time.h>
 
 /* types */
 typedef unsigned char      u8;
 typedef unsigned int       u32;
 typedef unsigned long long u64;
 typedef long long          i64;
 
 /* SHA-512 round constants
    Sourced from tweetnacl.c K[80] (identical to FIPS 180-4).                  */
 __device__ __constant__ u64 C512K[80] = {
     0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL,
     0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
     0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
     0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
     0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
     0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
     0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL,
     0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
     0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
     0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
     0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL,
     0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
     0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL,
     0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
     0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
     0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
     0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL,
     0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
     0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL,
     0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
     0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
     0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
     0xd192e819d6ef5218ULL, 0xd69906245565a910ULL,
     0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
     0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
     0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
     0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
     0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
     0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL,
     0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
     0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL,
     0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
     0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
     0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
     0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
     0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
     0x28db77f523047d84ULL, 0x32caab7b40c72493ULL,
     0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
     0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
     0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
 };
 
 /* SHA-256 round constants */
 __device__ __constant__ u32 C256K[64] = {
     0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
     0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
     0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
     0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
     0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
     0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
     0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
     0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
     0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
     0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
     0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
     0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
     0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
     0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
     0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
     0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
 };
 
/* GF(2^255-19) constants — 16 × int limbs (each ~16 bits)
   D2 = 2*d where d = -121665/121666 (mod p)
   BX, BY = Ed25519 base point coordinates
   Using int (32-bit) instead of i64 halves field-element register cost.       */
__device__ __constant__ int GF_D2[16] = {
    0xf159,0x26b2,0x9b94,0xebd6,0xb156,0x8283,0x149a,0x00e0,
    0xd130,0xeef3,0x80f2,0x198e,0xfce7,0x56df,0xd9dc,0x2406
};
__device__ __constant__ int GF_BX[16] = {
    0xd51a,0x8f25,0x2d60,0xc956,0xa7b2,0x9525,0xc760,0x692c,
    0xdc5c,0xfdd6,0xe231,0xc0a4,0x53fe,0xcd6e,0x36d3,0x2169
};
__device__ __constant__ int GF_BY[16] = {
    0x6658,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,
    0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666
};
__device__ __constant__ int GF_BT[16] = {
    0xdda3,0xa5b7,0x8ab3,0x6dde,0x52f5,0x7751,0x9f80,0x20f0,
    0xe37d,0x64ab,0x4e8e,0x66ea,0x7665,0xd78b,0x5f0f,0x6787
};
__device__ __constant__ int GF_BY_MINUS_BX[16] = {-28354,-10431,14598,-25328,-16716,-11967,-24826,-710,-30198,-38768,-31691,-23102,4712,-26376,12179,17661};
__device__ __constant__ int GF_BY_PLUS_BX[16] = {80754,62859,37830,77756,69144,64395,77254,53138,82626,91196,84119,75530,47716,78804,40249,34767};
__device__ __constant__ int GF_BT_D2[16] = {43605,34682,4613,43977,50334,52394,59427,9945,22924,56643,32203,23067,26024,40716,31592,61201};
 
 /* Base58 alphabet (Bitcoin/Octra) */
 __device__ __constant__ char B58[59] =
     "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
 
/* SHA-512 — specialized for exactly 32-byte input.
   Outputs only the first 32 bytes as 8x u32 to avoid local memory usage.*/
__device__ static void sha512_half(const u8 *data, u32 *out32)
{
    u64 h[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    u64 w[16];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        w[i] = ((u64)data[i*8+0]<<56)|((u64)data[i*8+1]<<48)
              |((u64)data[i*8+2]<<40)|((u64)data[i*8+3]<<32)
              |((u64)data[i*8+4]<<24)|((u64)data[i*8+5]<<16)
              |((u64)data[i*8+6]<< 8)|(u64)data[i*8+7];
    w[4]  = 0x8000000000000000ULL;
    #pragma unroll
    for (int i = 5; i < 15; i++) w[i] = 0ULL;
    w[15] = 256ULL;

    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    /* Rounds 0-15: schedule already in w[] */
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u64 S1 = ((e>>14)|(e<<50)) ^ ((e>>18)|(e<<46)) ^ ((e>>41)|(e<<23));
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = hv + S1 + ch + C512K[i] + w[i];
        u64 S0 = ((a>>28)|(a<<36)) ^ ((a>>34)|(a<<30)) ^ ((a>>39)|(a<<25));
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    /* Rounds 16-79: extend schedule in-place using rolling window */
    #pragma unroll
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

__device__ static void sha512_half_32(const u32 *data, u32 *out32)
{
    u64 h[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    u64 w[16];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 hi = __byte_perm(data[i*2], 0, 0x0123);
        u64 lo = __byte_perm(data[i*2+1], 0, 0x0123);
        w[i] = (hi << 32) | lo;
    }
    w[4]  = 0x8000000000000000ULL;
    #pragma unroll
    for (int i = 5; i < 15; i++) w[i] = 0ULL;
    w[15] = 256ULL;

    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u64 S1 = ((e>>14)|(e<<50)) ^ ((e>>18)|(e<<46)) ^ ((e>>41)|(e<<23));
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = hv + S1 + ch + C512K[i] + w[i];
        u64 S0 = ((a>>28)|(a<<36)) ^ ((a>>34)|(a<<30)) ^ ((a>>39)|(a<<25));
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    #pragma unroll
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
 
/* SHA-256 — specialized for exactly 32-byte input.
   Uses a 16-entry rolling window instead of w[64] to save ~48 u32 registers.*/
__device__ static void sha256_32(const u8 *data, u8 *out)
{
    u32 h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    u32 w[16];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = ((u32)data[i*4]<<24)|((u32)data[i*4+1]<<16)
              |((u64)data[i*4+2]<<8)|(u32)data[i*4+3];
    w[8]  = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0u;
    w[15] = 256u;

    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    /* Rounds 0-15 */
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u32 S1 = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = hv + S1 + ch + C256K[i] + w[i];
        u32 S0 = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    /* Rounds 16-63: extend schedule in-place */
    #pragma unroll
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

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4]  =(u8)(h[i]>>24); out[i*4+1]=(u8)(h[i]>>16);
        out[i*4+2]=(u8)(h[i]>> 8); out[i*4+3]=(u8)(h[i]);
    }
}

__device__ static void sha256_32_from32(const u32 *data, u32 *out)
{
    u32 h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    u32 w[16];

    #pragma unroll
    for (int i = 0; i < 8; i++)
        w[i] = __byte_perm(data[i], 0, 0x0123);
    w[8]  = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0u;
    w[15] = 256u;

    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u32 S1 = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = hv + S1 + ch + C256K[i] + w[i];
        u32 S0 = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    #pragma unroll
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

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = h[i];
    }
}
 
/* GF(2^255-19) arithmetic
   Limb representation: 16 × int (32-bit) limbs, each holding ~16 bits.
   Switching from i64 to int cuts field-element size from 32 to 16 u32 registers,
   roughly halving register pressure in ed_add and across all GF operations.
   Multiplication uses i64 accumulators internally, then reduces to int.*/
__device__ static void gf_set(int *r, const int *a)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) r[i] = a[i];
}

__device__ static void gf_zero(int *r)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) r[i] = 0;
}

__device__ static void gf_one(int *r)
{
    gf_zero(r); r[0] = 1;
}

__device__ static void gf_add(int *o, const int *a, const int *b)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = a[i] + b[i];
}

__device__ static void gf_sub(int *o, const int *a, const int *b)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = a[i] - b[i];
}

/* Carry-reduce: after add/sub limbs can exceed 16 bits; this normalises them. */
__device__ static void gf_car(int *o)
{
    int c;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] += (1<<16);
        c = o[i] >> 16;
        o[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
        o[i] -= c << 16;
    }
}

/* Constant-time conditional swap: if b==1 swap p and q element-wise */
__device__ static void gf_sel(int *p, int *q, int b)
{
    int t, c = ~(b - 1);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

/* Field multiplication.
   Accumulates in i64 (avoids int overflow: max term ≈ 39×16×(2^16)² ≈ 2^42),
   then carry-reduces twice inside the i64 before casting back to int.
   Result limbs are in [0, 2^16) and fit safely in int.*/
__device__ static void gf_mul(int *o, const int *a, const int *b)
{
    i64 t[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            i64 p = (i64)a[i] * b[j];
            if (i + j < 16) t[i + j]      += p;
            else            t[i + j - 16] += 38LL * p;
        }
    }
    /* Two carry passes in i64 before narrowing to int */
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        i64 c;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            t[i] += (1LL<<16);
            c = t[i] >> 16;
            t[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
            t[i] -= c << 16;
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = (int)t[i];
}

/* Squaring: computes each off-diagonal product once and doubles (~47% fewer mults). */
__device__ static void gf_sqr(int *o, const int *a)
{
    i64 t[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        #pragma unroll
        for (int j = i; j < 16; j++) {
            i64 p = (i64)a[i] * a[j];
            if (i != j) p += p;
            int k = i + j;
            if (k < 16) t[k]      += p;
            else        t[k - 16] += 38LL * p;
        }
    }
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        i64 c;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            t[i] += (1LL<<16);
            c = t[i] >> 16;
            t[(i+1)*(i<15)] += c - 1 + 37*(c-1)*(i==15);
            t[i] -= c << 16;
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) o[i] = (int)t[i];
}

/* Field inversion via Fermat: a^(p-2) mod p */
__device__ static void gf_inv(int *o, const int *inp)
{
    int c[16];
    gf_set(c, inp);
    for (int a = 253; a >= 0; a--) {
        gf_sqr(c, c);
        if (a != 2 && a != 4) gf_mul(c, c, inp);
    }
    gf_set(o, c);
}

/* Pack field element to 32 little-endian bytes */
__device__ static void gf_pack25519(u8 *o, const int *n)
{
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
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[2*i]   = (u8)(t[i] & 0xff);
        o[2*i+1] = (u8)(t[i] >> 8);
    }
}

__device__ static void gf_pack25519_32(u32 *o, const int *n)
{
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
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        o[i] = (t[2*i] & 0xffff) | ((t[2*i+1] & 0xffff) << 16);
    }
}

/* Parity bit: just normalise and return low bit of limb[0].
   Avoids the full 32-byte pack that gf_pack25519 would do. */
__device__ static u8 gf_par(const int *a)
{
    int t[16];
    gf_set(t, a);
    gf_car(t); gf_car(t); gf_car(t);
    return (u8)(t[0] & 1);
}

/* Precomputed table for scalar multiplication*/
struct PrecompPoint {
    u32 y_plus_x[8];
    u32 y_minus_x[8];
    u32 t_d2[8];
};

__device__ static void pack_limb32(u32 *out, const int *in) {
    u8 bytes[32];
    gf_pack25519(bytes, in);
    #pragma unroll
    for(int i=0; i<8; i++){
        out[i] = ((u32)bytes[i*4]) | ((u32)bytes[i*4+1]<<8) | ((u32)bytes[i*4+2]<<16) | ((u32)bytes[i*4+3]<<24);
    }
}

__device__ static void unpack_limb32(int *out, const u32 *in) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u32 v = in[i];
        out[2*i]   = v & 0xffff;
        out[2*i+1] = v >> 16;
    }
}

/* Ed25519 group operations — extended twisted Edwards coordinates
   Point p = (X:Y:Z:T)  stored as  p[0..3][0..15]  (4 × gf, int limbs) */

/* Dedicated point doubling: p ← 2 * p */
__device__ static void ed_double(int p[4][16])
{
    int A[16], B[16], C[16], E[16];

    gf_sqr(A, p[0]); // A = X1^2
    gf_sqr(B, p[1]); // B = Y1^2
    gf_sqr(C, p[2]); 
    gf_add(C, C, C); // C = 2 * Z1^2

    gf_add(E, p[0], p[1]); // E = X1 + Y1
    gf_sqr(E, E);          // E = (X1 + Y1)^2
    
    gf_add(p[3], A, B);    // H = A + B
    gf_sub(E, E, p[3]);    // E = (X1 + Y1)^2 - H
    
    gf_sub(p[2], B, A);    // G = B - A
    gf_sub(C, C, p[2]);    // F = C - G
    
    gf_mul(A, E, C);       // X3 = E * F
    gf_mul(B, p[2], p[3]); // Y3 = G * H
    
    gf_mul(p[0], E, p[3]); // T3 = E * H
    gf_mul(p[1], C, p[2]); // Z3 = F * G

    gf_set(p[3], p[0]); // T3
    gf_set(p[2], p[1]); // Z3
    gf_set(p[0], A);    // X3
    gf_set(p[1], B);    // Y3
}

/* Add base point B without allocating it */
__device__ static void ed_add_base(int p[4][16])
{
    int a[16], b[16], c[16], d[16];

    gf_sub(a, p[1], p[0]);
    gf_mul(a, a, GF_BY_MINUS_BX); 

    gf_add(b, p[0], p[1]);
    gf_mul(b, b, GF_BY_PLUS_BX); 

    gf_mul(c, p[3], GF_BT_D2); 

    #pragma unroll
    for (int i = 0; i < 16; i++) d[i] = p[2][i] + p[2][i]; // D = 2 * Z1 * 1

    gf_sub(p[0], b, a); // E = b - a
    gf_add(p[3], b, a); // H = b + a
    gf_sub(p[1], d, c); // F = d - c
    gf_add(p[2], d, c); // G = d + c

    gf_mul(a, p[0], p[1]); // E * F
    gf_mul(b, p[3], p[2]); // H * G
    gf_mul(c, p[2], p[1]); // G * F
    gf_mul(d, p[0], p[3]); // E * H

    gf_set(p[0], a);
    gf_set(p[1], b);
    gf_set(p[2], c);
    gf_set(p[3], d);
}

/* Compress point to 32 bytes */
__device__ static void ed_pack(u8 *r, int p[4][16])
{
    int tx[16], ty[16], zi[16];
    gf_inv(zi, p[2]);
    gf_mul(tx, p[0], zi);
    gf_mul(ty, p[1], zi);
    gf_pack25519(r, ty);
    r[31] ^= gf_par(tx) << 7;
}

__device__ static void ed_pack32(u32 *r, int p[4][16])
{
    int tx[16], ty[16], zi[16];
    gf_inv(zi, p[2]);
    gf_mul(tx, p[0], zi);
    gf_mul(ty, p[1], zi);
    gf_pack25519_32(r, ty);
    r[7] ^= (u32)gf_par(tx) << 31;
}

__global__ void init_table_kernel(PrecompPoint *table) {
    int i = blockIdx.x; // 0..31
    int j = threadIdx.x; // 0..255

    u32 s32[8] = {0};
    s32[i/4] = j << ((i%4)*8);

    int p[4][16];
    gf_zero(p[0]); gf_one(p[1]); gf_one(p[2]); gf_zero(p[3]);

    for (int word = 7; word >= 0; --word) {
        u32 w = s32[word];
        for (int bit = 31; bit >= 0; --bit) {
            ed_double(p);
            if ((w >> bit) & 1) ed_add_base(p);
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
    gf_mul(t_d2, t, GF_D2);

    int idx = i * 256 + j;
    pack_limb32(table[idx].y_plus_x, y_plus_x);
    pack_limb32(table[idx].y_minus_x, y_minus_x);
    pack_limb32(table[idx].t_d2, t_d2);
}

/* Scalar-base multiply using precomputed 8-bit window table */
__device__ static void ed_scalarbase_table(int p[4][16], const u32 *s32, const PrecompPoint * __restrict__ table)
{
    // Initialize p to the neutral element (0, 1, 1, 0)
    gf_zero(p[0]); gf_one(p[1]); gf_one(p[2]); gf_zero(p[3]);

    const u8 *s8 = (const u8*)s32;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        u8 j = s8[i];
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

        #pragma unroll
        for (int k = 0; k < 16; k++) d[k] = p[2][k] + p[2][k];

        gf_sub(p[0], b, a); // E = b - a
        gf_add(p[3], b, a); // H = b + a
        gf_sub(p[1], d, c); // F = d - c
        gf_add(p[2], d, c); // G = d + c

        gf_mul(a, p[0], p[1]); // E * F
        gf_mul(b, p[3], p[2]); // H * G
        gf_mul(c, p[2], p[1]); // G * F
        gf_mul(d, p[0], p[3]); // E * H

        gf_set(p[0], a);
        gf_set(p[1], b);
        gf_set(p[2], c);
        gf_set(p[3], d);
    }
}

 /* Public-key derivation: seed[32] → pk[32] — Mirrors tweetnacl crypto_sign_seed_keypair (pubkey part only).*/
 __device__ static void derive_pubkey(const u8 *seed, u8 *pk, const PrecompPoint * __restrict__ table)
 {
    u32 s32[8];
    sha512_half(seed, s32);
    s32[0] &= 0xFFFFFFF8u;
    s32[7] &= 0x7FFFFFFFu;
    s32[7] |= 0x40000000u;

    int p[4][16];
    ed_scalarbase_table(p, s32, table);
    ed_pack(pk, p);
 }

 __device__ static void derive_pubkey32(const u32 *seed, u32 *pk, const PrecompPoint * __restrict__ table)
 {
    u32 s32[8];
    sha512_half_32(seed, s32);
    s32[0] &= 0xFFFFFFF8u;
    s32[7] &= 0x7FFFFFFFu;
    s32[7] |= 0x40000000u;

    int p[4][16];
    ed_scalarbase_table(p, s32, table);
    ed_pack32(pk, p);
 }

/* Pattern matching modes */
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
    bool case_insensitive;
    int  rep_bonus;
    int  rep_bonus_mode;
};
 
 /* Base58 encoding of exactly 32 bytes → null-terminated string
    Returns length (typically 43 or 44 for SHA-256 output). */
 __device__ static int b58enc(const u32 *t_in, char *out)
 {
     u32 t[8];
     #pragma unroll
     for (int i = 0; i < 8; i++) t[i] = t_in[i];
 
     char buf[48];
     int  blen = 0;
 
     int leading = 0;
     for(int i=0; i<8; i++) {
         if (t_in[i] == 0) leading += 4;
         else {
             leading += __clz(t_in[i]) / 8;
             break;
         }
     }
 
     int begin = 0;
     while (begin < 8 && t[begin] == 0) begin++;
 
     while (begin < 8) {
         int new_begin = 8;
         u32 rem = 0;
         #pragma unroll
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


__device__ static bool char_match(char c1, char c2, bool case_insensitive) {
    if (case_insensitive) {
        if (c1 >= 'A' && c1 <= 'Z') c1 ^= 0x20;
        if (c2 >= 'A' && c2 <= 'Z') c2 ^= 0x20;
    }
    return c1 == c2;
}

__device__ static bool b58_suffix_match(const u32 *t_in, const VanityCfg *c)
{
    u32 t[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for(int i=0; i<8; i++) {
        if (t_in[i] == 0) leading += 4;
        else {
            leading += __clz(t_in[i]) / 8;
            break;
        }
    }
    if (leading == 32) {
        if (c->plen > 32) return false;
        bool ci = c->case_insensitive;
        for (int i = 0; i < c->plen; i++)
            if (!char_match(c->pat[i], '1', ci)) return false;
        return true;
    }

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    bool ci = c->case_insensitive;
    for (int matched = 0; matched < c->plen; matched++) {
        if (begin >= 8) return false;

        int new_begin = 8;
        u32 rem = 0;
        #pragma unroll
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

__device__ static bool b58_rep_end_match(const u32 *t_in, int rep)
{
    u32 t[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for(int i=0; i<8; i++) {
        if (t_in[i] == 0) leading += 4;
        else {
            leading += __clz(t_in[i]) / 8;
            break;
        }
    }
    if (leading == 32) return rep <= 32;

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    char want = '\0';
    for (int matched = 0; matched < rep; matched++) {
        if (begin >= 8) return false;

        int new_begin = 8;
        u32 rem = 0;
        #pragma unroll
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

__device__ static bool b58_rep_any_match(const u32 *t_in, int rep)
{
    u32 t[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) t[i] = t_in[i];

    int leading = 0;
    for(int i=0; i<8; i++) {
        if (t_in[i] == 0) leading += 4;
        else {
            leading += __clz(t_in[i]) / 8;
            break;
        }
    }
    if (leading == 32) return rep <= 32;

    int begin = 0;
    while (begin < 8 && t[begin] == 0) begin++;

    char prev = '\0';
    int run = 0;

    while (begin < 8) {
        int new_begin = 8;
        u32 rem = 0;
        #pragma unroll
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

__device__ static bool check_rep_bonus(const u32 *t_in, const VanityCfg *c) {
    if (c->rep_bonus <= 0) return false;
    if (c->rep_bonus_mode == 2) {
        return b58_rep_end_match(t_in, c->rep_bonus);
    } else if (c->rep_bonus_mode == 3) {
        return b58_rep_any_match(t_in, c->rep_bonus);
    } else {
        u32 t[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) t[i] = t_in[i];

        int leading = 0;
        for(int i=0; i<8; i++) {
            if (t_in[i] == 0) leading += 4;
            else {
                leading += __clz(t_in[i]) / 8;
                break;
            }
        }
        if (leading >= c->rep_bonus) return true;

        int begin = 0;
        while (begin < 8 && t[begin] == 0) begin++;

        char buf[48];
        int blen = 0;
        while (begin < 8) {
            int new_begin = 8;
            u32 rem = 0;
            #pragma unroll
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
}
 
 /* s is the base58 part only (without "oct" prefix) */
 __device__ static bool pat_match(const char *s, int slen, const VanityCfg *c)
 {
     bool ci = c->case_insensitive;
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
 
/* Main kernel — iters_per_thread keys per thread per launch*/
__global__ __launch_bounds__(128, 4) void vanity_kernel(
    u64           base_seed,
    int           iters_per_thread,
    VanityCfg    *cfg,
    volatile int *found_flag,
    u8           *found_seed,
    char         *found_addr,
    volatile int *rep_found_flag,
    u8           *rep_seed,
    char         *rep_addr,
    const PrecompPoint * __restrict__ table)
{
    if (*found_flag) return;

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Per-thread RNG — splitmix64 seeding then xorshift64 */
    u64 rng = base_seed + (u64)tid * 6364136223846793005ULL + 1442695040888963407ULL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9ULL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebULL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (*found_flag) return;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            if (atomicCAS((int*)rep_found_flag, 0, 1) == 0) {
                for (int i = 0; i < 8; i++) {
                    rep_seed[i*4]   = seed[i] & 0xff;
                    rep_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                    rep_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                    rep_seed[i*4+3] = (seed[i] >> 24) & 0xff;
                }
                b58enc(pk, rep_addr);
            }
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
            matched = pat_match(addr, alen, cfg);
            if (!matched) break;
            if (atomicCAS((int*)found_flag, 0, 1) == 0) {
                for (int i = 0; i < 8; i++) {
                    found_seed[i*4]   = seed[i] & 0xff;
                    found_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                    found_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                    found_seed[i*4+3] = (seed[i] >> 24) & 0xff;
                }
                for (int i = 0; i < alen; i++) found_addr[i] = addr[i];
                found_addr[alen] = '\0';
            }
            return;
        }
        }

        if (matched) {
            char addr[48];
            int alen = b58enc(pk, addr);
            if (atomicCAS((int*)found_flag, 0, 1) == 0) {
                for (int i = 0; i < 8; i++) {
                    found_seed[i*4]   = seed[i] & 0xff;
                    found_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                    found_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                    found_seed[i*4+3] = (seed[i] >> 24) & 0xff;
                }
                for (int i = 0; i < alen; i++) found_addr[i] = addr[i];
                found_addr[alen] = '\0';
            }
            return;
        }
    }
}

__global__ __launch_bounds__(128, 4) void vanity_kernel_rep_end(
    u64           base_seed,
    int           iters_per_thread,
    VanityCfg    *cfg,
    volatile int *found_flag,
    u8           *found_seed,
    char         *found_addr,
    volatile int *rep_found_flag,
    u8           *rep_seed,
    char         *rep_addr,
    const PrecompPoint * __restrict__ table)
{
    if (*found_flag) return;

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    u64 rng = base_seed + (u64)tid * 6364136223846793005ULL + 1442695040888963407ULL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9ULL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebULL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (*found_flag) return;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            if (atomicCAS((int*)rep_found_flag, 0, 1) == 0) {
                for (int i = 0; i < 8; i++) {
                    rep_seed[i*4]   = seed[i] & 0xff;
                    rep_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                    rep_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                    rep_seed[i*4+3] = (seed[i] >> 24) & 0xff;
                }
                b58enc(pk, rep_addr);
            }
        }

        if (!b58_rep_end_match(pk, cfg->rep)) continue;

        if (atomicCAS((int*)found_flag, 0, 1) == 0) {
            for (int i = 0; i < 8; i++) {
                found_seed[i*4]   = seed[i] & 0xff;
                found_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                found_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                found_seed[i*4+3] = (seed[i] >> 24) & 0xff;
            }
            b58enc(pk, found_addr);
        }
        return;
    }
}

__global__ __launch_bounds__(128, 4) void vanity_kernel_suffix(
    u64           base_seed,
    int           iters_per_thread,
    VanityCfg    *cfg,
    volatile int *found_flag,
    u8           *found_seed,
    char         *found_addr,
    volatile int *rep_found_flag,
    u8           *rep_seed,
    char         *rep_addr,
    const PrecompPoint * __restrict__ table)
{
    if (*found_flag) return;

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    u64 rng = base_seed + (u64)tid * 6364136223846793005ULL + 1442695040888963407ULL;
    rng ^= rng >> 30; rng *= 0xbf58476d1ce4e5b9ULL;
    rng ^= rng >> 27; rng *= 0x94d049bb133111ebULL;
    rng ^= rng >> 31;

    u32 seed[8], pk[8];

    for (int iter = 0; iter < iters_per_thread; iter++) {
        if (*found_flag) return;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            u64 v = rng;
            seed[i*2]   = (u32)v;
            seed[i*2+1] = (u32)(v >> 32);
        }

        derive_pubkey32(seed, pk, table);
        sha256_32_from32(pk, pk);

        if (check_rep_bonus(pk, cfg)) {
            if (atomicCAS((int*)rep_found_flag, 0, 1) == 0) {
                for (int i = 0; i < 8; i++) {
                    rep_seed[i*4]   = seed[i] & 0xff;
                    rep_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                    rep_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                    rep_seed[i*4+3] = (seed[i] >> 24) & 0xff;
                }
                b58enc(pk, rep_addr);
            }
        }

        if (!b58_suffix_match(pk, cfg)) continue;

        if (atomicCAS((int*)found_flag, 0, 1) == 0) {
            for (int i = 0; i < 8; i++) {
                found_seed[i*4]   = seed[i] & 0xff;
                found_seed[i*4+1] = (seed[i] >> 8) & 0xff;
                found_seed[i*4+2] = (seed[i] >> 16) & 0xff;
                found_seed[i*4+3] = (seed[i] >> 24) & 0xff;
            }
            b58enc(pk, found_addr);
        }
        return;
    }
}
 
/* Host utilities*/
#include <math.h>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "\nCUDA error %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/* Base64-encode exactly 32 bytes → 44 chars + '\0' (RFC 4648, with padding) */
static void b64enc32(const u8 *in, char *out)
{
    static const char T[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 10; i++) {
        u32 n = ((u32)in[i*3]<<16)|((u32)in[i*3+1]<<8)|in[i*3+2];
        out[i*4+0]=T[(n>>18)&63]; out[i*4+1]=T[(n>>12)&63];
        out[i*4+2]=T[(n>> 6)&63]; out[i*4+3]=T[n&63];
    }
    u32 n = ((u32)in[30]<<16)|((u32)in[31]<<8);
    out[40]=T[(n>>18)&63]; out[41]=T[(n>>12)&63];
    out[42]=T[(n>> 6)&63]; out[43]='=';
    out[44]='\0';
}

/* Returns true if every character in pat is a valid base58 character */
static bool valid_b58_str(const char *pat)
{
    static const char *alph =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (int i = 0; pat[i]; i++) {
        bool ok = false;
        for (int j = 0; alph[j]; j++)
            if (pat[i] == alph[j]) { ok = true; break; }
        if (!ok) return false;
    }
    return true;
}

/* Read a line from stdin, strip newline, return length */
static int read_line(char *buf, int maxlen)
{
    if (!fgets(buf, maxlen, stdin)) { buf[0]='\0'; return 0; }
    int n = (int)strlen(buf);
    while (n > 0 && (buf[n-1]=='\n' || buf[n-1]=='\r')) buf[--n]='\0';
    return n;
}

/* Read an integer; return default_val if user just presses Enter */
static int read_int_default(int default_val)
{
    char buf[32];
    read_line(buf, sizeof(buf));
    if (buf[0] == '\0') return default_val;
    int v = atoi(buf);
    return (v > 0) ? v : default_val;
}

/* Print estimated difficulty and search time */
static void print_estimate(int mode, int n, int blocks, int threads)
{
    /* Difficulty = expected number of keys to try */
    double diff;
    switch (mode) {
        case MODE_PREFIX:
        case MODE_SUFFIX:    diff = pow(58.0, n);          break;
        case MODE_ANYWHERE:  diff = pow(58.0, n) / 38.0;   break; /* ~38 positions */
        case MODE_REP_START:
        case MODE_REP_END:   diff = pow(58.0, n - 1);      break;
        case MODE_REP_ANY:   diff = pow(58.0, n - 1) / 38.0; break;
        default:             diff = 1.0;
    }

    double speed_est = 3e6;
    double seconds         = diff / speed_est;

    printf("  Difficulty:  ~%.3g keys on average\n", diff);
    printf("  Estimated:   ");
    if      (seconds < 2)       printf("< 1 second\n");
    else if (seconds < 120)     printf("~%.0f seconds\n",     seconds);
    else if (seconds < 7200)    printf("~%.0f minutes\n",     seconds / 60.0);
    else if (seconds < 172800)  printf("~%.1f hours\n",       seconds / 3600.0);
    else                        printf("~%.1f days\n",        seconds / 86400.0);
}

/* Separator line */
static void sep() { printf("  ***********************************************\n"); }

/* main — interactive*/
int main(void)
{
    /* GPU info */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    /* Banner */
    printf("\n");
    printf("  *************************************************\n");
    printf("  *             Octra Vanity Address Miner         *\n");
    printf("  *************************************************\n");
    printf("  GPU : %s\n", prop.name);
    printf("  SM  : %d.%d  |  %d multiprocessors\n",
           prop.major, prop.minor, prop.multiProcessorCount);
    printf("\n");

    /* Auto-tune blocks and threads based on GPU architecture */
    int threads = 128; // Optimal to avoid register fragmentation for 128-reg kernels
    int blocks = prop.multiProcessorCount * 64; // High occupancy: 64 blocks per SM to queue up
    if (blocks < 4096) blocks = 4096;
    if (blocks > 65535) blocks = 65535;
    
    /* Higher end GPUs can handle more iterations internally without timeout */
    int iters = (prop.major >= 8) ? 16 : 8; 

    VanityCfg cfg    = {};
    char rpc[256]    = "https://devnet.octra.com";
    char pat_buf[64] = {};

    /* STEP 1 — Search type*/
    sep();
    printf("  Where should the pattern appear?\n\n");
    printf("    1.  Prefix      oct[PATTERN]...\n");
    printf("    2.  Suffix      ...[PATTERN]\n");
    printf("    3.  Anywhere    ...[PATTERN]...\n");
    printf("    4.  Repeating   consecutive identical characters\n");
    printf("\n");

    int type = 0;
    while (type < 1 || type > 4) {
        printf("  > ");
        fflush(stdout);
        char buf[16];
        read_line(buf, sizeof(buf));
        type = atoi(buf);
        if (type < 1 || type > 4)
            printf("  Please enter 1, 2, 3 or 4.\n");
    }

    /* STEP 2 — Pattern or repeat count*/
    printf("\n");
    sep();

    if (type <= 3) {
        /* Specific string */
        static const char *type_label[] = { "", "prefix", "suffix", "anywhere" };
        printf("  Enter the pattern (%s).\n", type_label[type]);
        printf("  Valid chars: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz\n");
        printf("  Note: 0 O I l are NOT in base58.\n\n");

        bool valid = false;
        while (!valid) {
            printf("  Pattern > ");
            fflush(stdout);
            int n = read_line(pat_buf, sizeof(pat_buf));
            if (n == 0) {
                printf("  Pattern cannot be empty.\n");
                continue;
            }
            if (!valid_b58_str(pat_buf)) {
                printf("  Contains invalid base58 characters. Try again.\n");
                continue;
            }
            valid = true;
        }

        cfg.mode = (type == 1) ? MODE_PREFIX
                 : (type == 2) ? MODE_SUFFIX
                 :                MODE_ANYWHERE;
        strncpy(cfg.pat, pat_buf, 63);
        cfg.plen = (int)strlen(cfg.pat);

        printf("\n");
        printf("  Case sensitive? (Y/n) > ");
        fflush(stdout);
        char ci_buf[16];
        read_line(ci_buf, sizeof(ci_buf));
        cfg.case_insensitive = (ci_buf[0] == 'n' || ci_buf[0] == 'N');

        printf("\n");
        printf("  Also record long repeating patterns? (y/N) > ");
        fflush(stdout);
        char bonus_buf[16];
        read_line(bonus_buf, sizeof(bonus_buf));
        
        if (bonus_buf[0] == 'y' || bonus_buf[0] == 'Y') {
            printf("\n");
            int rep_b = 0;
            while (rep_b < 2) {
                printf("  What's the length of these patterns? (min 2) > ");
                fflush(stdout);
                char r_buf[16];
                read_line(r_buf, sizeof(r_buf));
                rep_b = atoi(r_buf);
                if (rep_b < 2) printf("  Must be at least 2.\n");
            }
            cfg.rep_bonus = rep_b;

            printf("\n");
            int r_mode = 0;
            while (r_mode < 1 || r_mode > 3) {
                printf("  Where these patterns must occur? (1 - prefix, 2 - suffix, 3 - anywhere) > ");
                fflush(stdout);
                char rm_buf[16];
                read_line(rm_buf, sizeof(rm_buf));
                r_mode = atoi(rm_buf);
                if (r_mode < 1 || r_mode > 3) printf("  Please enter 1, 2, or 3.\n");
            }
            cfg.rep_bonus_mode = r_mode;
        } else {
            cfg.rep_bonus = 0;
            cfg.rep_bonus_mode = 0;
        }

    } else {
        /* Repeating characters */
        printf("  Where should the repeating characters appear?\n\n");
        printf("    1.  At start   oct[XXXXX]...\n");
        printf("    2.  At end     ...[XXXXX]\n");
        printf("    3.  Anywhere   ...[XXXXX]...\n\n");

        int sub = 0;
        while (sub < 1 || sub > 3) {
            printf("  > ");
            fflush(stdout);
            char buf[16];
            read_line(buf, sizeof(buf));
            sub = atoi(buf);
            if (sub < 1 || sub > 3)
                printf("  Please enter 1, 2 or 3.\n");
        }
        cfg.mode = (sub == 1) ? MODE_REP_START
                 : (sub == 2) ? MODE_REP_END
                 :               MODE_REP_ANY;

        printf("\n");
        int rep = 0;
        while (rep < 2) {
            printf("  How many identical chars? (min 2) > ");
            fflush(stdout);
            char buf[16];
            read_line(buf, sizeof(buf));
            rep = atoi(buf);
            if (rep < 2)
                printf("  Must be at least 2.\n");
        }
        cfg.rep = rep;
        cfg.rep_bonus = 0;
        cfg.rep_bonus_mode = 0;
    }

    /* Show what we'll search for + difficulty */
    printf("\n");
    sep();
    printf("  Target:  ");
    switch (cfg.mode) {
        case MODE_PREFIX:    printf("oct%s... %s\n",       cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_SUFFIX:    printf("...%s %s\n",          cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_ANYWHERE:  printf("...%s... %s\n",       cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_REP_START: printf("oct[%dx same]...\n",  cfg.rep); break;
        case MODE_REP_END:   printf("...[%dx same]\n",     cfg.rep); break;
        case MODE_REP_ANY:   printf("...[%dx same]...\n",  cfg.rep); break;
    }
    int n_for_est = (cfg.mode <= MODE_ANYWHERE) ? cfg.plen : cfg.rep;
    print_estimate(cfg.mode, n_for_est, blocks, threads);

        /* STEP 3 — Performance settings (optional)*/
        /* STEP 3 — Performance settings */
    printf("\n");
    sep();
    printf("  Performance settings\n\n");
    printf("  Auto-tune performance parameters? (Y/n) > ");
    fflush(stdout);
    char at_buf[16];
    read_line(at_buf, sizeof(at_buf));
    bool do_auto_tune = (at_buf[0] != 'n' && at_buf[0] != 'N');

    if (!do_auto_tune) {
        printf("\n  CUDA blocks    [%d] > ", blocks);
        fflush(stdout);
        blocks = read_int_default(blocks);

        printf("  Threads/block  [%d] > ", 128);
        fflush(stdout);
        threads = read_int_default(128);

        printf("  Keys/thread    [%d] > ", iters);
        fflush(stdout);
        iters = read_int_default(iters);

        /* Clamp threads to valid warp multiple */
        threads = ((threads + 31) / 32) * 32;
        if (threads > 1024) threads = 1024;
        if (blocks  < 1)    blocks  = 1;
        if ((cfg.mode == MODE_REP_END || cfg.mode == MODE_SUFFIX) && threads > 128)
            threads = 128;
    }

    /* STEP 4 — RPC URL (optional)*/
    printf("\n");
    sep();
    printf("  RPC URL for saved wallet\n");
    printf("  [%s] > ", rpc);
    fflush(stdout);
    char rpc_buf[256];
    int rlen = read_line(rpc_buf, sizeof(rpc_buf));
    if (rlen > 0) strncpy(rpc, rpc_buf, 255);

    /* Summary + confirm*/
    printf("\n");
    sep();
    printf("  Ready to search\n\n");
    printf("  Target  : ");
    switch (cfg.mode) {
        case MODE_PREFIX:    printf("oct%s... %s\n",         cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_SUFFIX:    printf("...%s %s\n",            cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_ANYWHERE:  printf("...%s... %s\n",         cfg.pat, cfg.case_insensitive ? "(case-insensitive)" : "(case-sensitive)"); break;
        case MODE_REP_START: printf("oct[%dx same]...\n", cfg.rep); break;
        case MODE_REP_END:   printf("...[%dx same]\n",    cfg.rep); break;
        case MODE_REP_ANY:   printf("...[%dx same]...\n", cfg.rep); break;
    }
    if (do_auto_tune) {
        printf("  Grid    : Auto-tuned at runtime\n");
    } else {
        printf("  Grid    : %d blocks x %d threads x %d iters = %llu keys/launch\n",
               blocks, threads, iters, (u64)blocks * threads * iters);
    }
    printf("  RPC     : %s\n\n", rpc);
    printf("  Press Enter to start, Ctrl+C to cancel...");
    fflush(stdout);
    getchar();

    /* Search*/
    printf("\n");
    sep();
    printf("  Searching...\n\n");

    /* Initialize precomputed scalar table */
    PrecompPoint *d_table;
    CUDA_CHECK(cudaMalloc((void**)&d_table, 32 * 256 * sizeof(PrecompPoint)));
    init_table_kernel<<<32, 256>>>(d_table);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VanityCfg    *d_cfg;
    volatile int *d_found;
    u8           *d_seed;
    char         *d_addr;
    volatile int *d_rep_found;
    u8           *d_rep_seed;
    char         *d_rep_addr;

    CUDA_CHECK(cudaMalloc((void**)&d_cfg,   sizeof(VanityCfg)));
    CUDA_CHECK(cudaMalloc((void**)&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_seed,  32));
    CUDA_CHECK(cudaMalloc((void**)&d_addr,  64));
    CUDA_CHECK(cudaMalloc((void**)&d_rep_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_rep_seed,  32));
    CUDA_CHECK(cudaMalloc((void**)&d_rep_addr,  64));

    CUDA_CHECK(cudaMemcpy((void*)d_cfg, &cfg, sizeof(VanityCfg), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void*)d_found, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset((void*)d_rep_found, 0, sizeof(int)));

    if (do_auto_tune) {
        printf("  Auto-tuning performance parameters...\n");
        int best_blocks = blocks;
        int best_threads = threads;
        int best_iters = iters;
        float max_kps = 0.0f;

        cudaEvent_t tune_ev_start, tune_ev_stop;
        CUDA_CHECK(cudaEventCreate(&tune_ev_start));
        CUDA_CHECK(cudaEventCreate(&tune_ev_stop));

        /* Warm-up */
        if (cfg.mode == MODE_REP_END) {
            vanity_kernel_rep_end<<<blocks, threads>>>(0, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        } else if (cfg.mode == MODE_SUFFIX) {
            vanity_kernel_suffix<<<blocks, threads>>>(0, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        } else {
            vanity_kernel<<<blocks, threads>>>(0, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        int tune_threads[] = {64, 128, 256, 512, 1024};
        int tune_mults[] = {1, 2, 4, 8, 16};
        int tune_iters[] = {1, 2, 4, 8, 16};

        for(int t_idx = 0; t_idx < 5; t_idx++) {
            int t = tune_threads[t_idx];
            if ((cfg.mode == MODE_REP_END || cfg.mode == MODE_SUFFIX) && t > 128) continue;
            
            for(int m_idx = 0; m_idx < 5; m_idx++) {
                int b = tune_mults[m_idx] * prop.multiProcessorCount;
                if (b < 1) b = 1;
                
                for(int i_idx = 0; i_idx < 5; i_idx++) {
                    int i = tune_iters[i_idx];
                    
                    CUDA_CHECK(cudaEventRecord(tune_ev_start));
                    if (cfg.mode == MODE_REP_END) {
                        vanity_kernel_rep_end<<<b, t>>>(0, i, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
                    } else if (cfg.mode == MODE_SUFFIX) {
                        vanity_kernel_suffix<<<b, t>>>(0, i, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
                    } else {
                        vanity_kernel<<<b, t>>>(0, i, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
                    }
                    CUDA_CHECK(cudaEventRecord(tune_ev_stop));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, tune_ev_start, tune_ev_stop));
                    if (ms > 0.0f) {
                        float kps = (float)((u64)b * t * i) / (ms / 1000.0f);
                        if (kps > max_kps) {
                            max_kps = kps;
                            best_blocks = b;
                            best_threads = t;
                            best_iters = i;
                        }
                    }
                }
            }
        }
        
        printf("  Auto-tune complete: best %d blocks, %d threads, %d keys/thread (%.2f MK/s)\n\n", 
               best_blocks, best_threads, best_iters, max_kps / 1000000.0f);
        blocks = best_blocks;
        threads = best_threads;
        iters = best_iters;

        CUDA_CHECK(cudaEventDestroy(tune_ev_start));
        CUDA_CHECK(cudaEventDestroy(tune_ev_stop));

        CUDA_CHECK(cudaMemset((void*)d_found, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset((void*)d_rep_found, 0, sizeof(int)));
    }

    /* Kernel diagnostics */
    {
        cudaFuncAttributes attr;
        if (cfg.mode == MODE_REP_END) {
            CUDA_CHECK(cudaFuncGetAttributes(&attr, vanity_kernel_rep_end));
        } else if (cfg.mode == MODE_SUFFIX) {
            CUDA_CHECK(cudaFuncGetAttributes(&attr, vanity_kernel_suffix));
        } else {
            CUDA_CHECK(cudaFuncGetAttributes(&attr, vanity_kernel));
        }
        int regs      = attr.numRegs;
        int smem      = (int)attr.sharedSizeBytes;
        int max_thr   = attr.maxThreadsPerBlock;
        int regs_per_sm = prop.regsPerBlock;          /* total regs per SM   */
        int warps_per_sm= prop.maxThreadsPerMultiProcessor / 32;
        int blocks_per_sm = regs > 0
            ? (regs_per_sm / (regs * threads))        /* limited by regs     */
            : prop.maxBlocksPerMultiProcessor;
        if (blocks_per_sm > prop.maxBlocksPerMultiProcessor)
            blocks_per_sm = prop.maxBlocksPerMultiProcessor;
        int active_warps  = blocks_per_sm * (threads / 32);
        double occupancy  = (double)active_warps / warps_per_sm * 100.0;

        printf("  [diag] Registers/thread : %d\n", regs);
        printf("  [diag] Shared mem/block : %d bytes\n", smem);
        printf("  [diag] Max threads/blk  : %d\n", max_thr);
        printf("  [diag] Blocks/SM (regs) : %d  (SM has %d regs)\n",
               blocks_per_sm, regs_per_sm);
        printf("  [diag] Active warps/SM  : %d / %d  (%.0f%% occupancy)\n",
               active_warps, warps_per_sm, occupancy);
        printf("\n");
    }

    u64    base_seed   = (u64)time(NULL) ^ 0xc0ffeedeadbeefULL;
    int    found       = 0;
    u64    total_keys  = 0;
    u64    launch_num  = 0;
    time_t t_start     = time(NULL);
    time_t t_last_prog = t_start;

    /* CUDA events for per-launch timing */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    while (!found) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (cfg.mode == MODE_REP_END) {
            vanity_kernel_rep_end<<<blocks, threads>>>(
                base_seed, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        } else if (cfg.mode == MODE_SUFFIX) {
            vanity_kernel_suffix<<<blocks, threads>>>(
                base_seed, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        } else {
            vanity_kernel<<<blocks, threads>>>(
                base_seed, iters, d_cfg, d_found, d_seed, d_addr, d_rep_found, d_rep_seed, d_rep_addr, d_table);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float launch_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&launch_ms, ev_start, ev_stop));

        base_seed  += (u64)blocks * (u64)threads * (u64)iters;
        total_keys += (u64)blocks * (u64)threads * (u64)iters;
        launch_num++;

        CUDA_CHECK(cudaMemcpy(&found, (void*)d_found, sizeof(int), cudaMemcpyDeviceToHost));

        int rep_found = 0;
        CUDA_CHECK(cudaMemcpy(&rep_found, (void*)d_rep_found, sizeof(int), cudaMemcpyDeviceToHost));
        if (rep_found) {
            u8   res_rep_seed[32] = {0};
            char res_rep_addr[64] = {0};
            CUDA_CHECK(cudaMemcpy(res_rep_seed, (void*)d_rep_seed, 32, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(res_rep_addr, (void*)d_rep_addr, 64, cudaMemcpyDeviceToHost));

            char priv_b64[48];
            b64enc32(res_rep_seed, priv_b64);

            printf("\r                                                                                                    \r");
            printf("  Found bonus pattern: oct%s\n", res_rep_addr);

            char fname[128];
            char short_id[9];
            strncpy(short_id, res_rep_addr, 8);
            short_id[8] = '\0';
            snprintf(fname, sizeof(fname), "wallet_bonus_%s.json", short_id);

            FILE *fp = fopen(fname, "w");
            if (fp) {
                fprintf(fp,
                    "{\n"
                    "  \"priv\": \"%s\",\n"
                    "  \"addr\": \"oct%s\",\n"
                    "  \"rpc\": \"%s\"\n"
                    "}\n",
                    priv_b64, res_rep_addr, rpc);
                fclose(fp);
            }

            CUDA_CHECK(cudaMemset((void*)d_rep_found, 0, sizeof(int)));
            t_last_prog = 0; // force progress redraw immediately
        }

        time_t now = time(NULL);
        if (!found && (now - t_last_prog) >= 2) {
            double elapsed  = (double)(now - t_start);
            double kps      = (elapsed > 0) ? (double)total_keys / elapsed : 0.0;
            u64    keys_this = (u64)blocks * threads * iters;
            double kps_inst  = (launch_ms > 0) ? (double)keys_this / (launch_ms / 1000.0) : 0.0;
            printf("\r  Tried: %-14llu  %5.0fs  avg %9.0f k/s  last launch %.0fms (%.0f k/s inst)",
                   (unsigned long long)total_keys, elapsed, kps,
                   (double)launch_ms, kps_inst / 1000.0);
            fflush(stdout);
            t_last_prog = now;
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* Result*/
    u8   res_seed[32] = {0};
    char res_addr[64] = {0};
    CUDA_CHECK(cudaMemcpy(res_seed, (void*)d_seed, 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(res_addr, (void*)d_addr, 64, cudaMemcpyDeviceToHost));

    time_t t_end     = time(NULL);
    double elapsed   = (double)(t_end - t_start);
    double kps_final = (elapsed > 0) ? (double)total_keys / elapsed : 0.0;

    char priv_b64[48];
    b64enc32(res_seed, priv_b64);

    printf("\n\n");
    printf("  *************************************************\n");
    printf("  *                 FOUND!                    *\n");
    printf("  *************************************************\n\n");
    printf("  Address     oct%s\n", res_addr);
    printf("  Private key %s\n",    priv_b64);
    printf("  Tried       %llu keys in %.0f s  (%.0f keys/s)\n\n",
           (unsigned long long)total_keys, elapsed, kps_final);

    /* Save wallet JSON */
    char fname[128];
    char short_id[9];
    strncpy(short_id, res_addr, 8);
    short_id[8] = '\0';
    snprintf(fname, sizeof(fname), "wallet_vanity_%s.json", short_id);

    FILE *fp = fopen(fname, "w");
    if (fp) {
        fprintf(fp,
            "{\n"
            "  \"priv\": \"%s\",\n"
            "  \"addr\": \"oct%s\",\n"
            "  \"rpc\": \"%s\"\n"
            "}\n",
            priv_b64, res_addr, rpc);
        fclose(fp);
        printf("  Saved       %s\n", fname);
    } else {
        fprintf(stderr, "  Warning: could not save wallet file.\n");
    }

    printf("\n");
    sep();
    printf("  Press Enter to exit...");
    fflush(stdout);
    getchar();

    cudaFree((void*)d_cfg);
    cudaFree((void*)d_found);
    cudaFree((void*)d_seed);
    cudaFree((void*)d_addr);
    cudaFree((void*)d_rep_found);
    cudaFree((void*)d_rep_seed);
    cudaFree((void*)d_rep_addr);
    cudaFree((void*)d_table);
    return 0;
}