// TurboQuant (Zandieh et al., ICLR 2026) prototype: WHT + Lloyd-Max.
// Codec is SELF-CONTAINED: the Hadamard rotation is applied at encode and
// undone at decode, so the ggml type stays a drop-in -- attention math is
// untouched. That is the whole reason this can be a plain GGML_TYPE_*.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Lloyd-Max 3-bit optimal quantizer for a unit Gaussian (Max, 1960).
// The WHT is what makes assuming a Gaussian legitimate: averaging 2^k
// arbitrary values drives each coefficient toward normal (CLT).
static const float LM3[8] = {-2.1520f,-1.3439f,-0.7560f,-0.2451f,
                              0.2451f, 0.7560f, 1.3439f, 2.1520f};
static const float LM3_B[7] = {-1.7480f,-1.0500f,-0.5006f, 0.0f,
                               0.5006f, 1.0500f, 1.7480f};
static const float LM4[16] = {-2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9423f,-0.6568f,-0.3881f,-0.1284f,
                               0.1284f, 0.3881f, 0.6568f, 0.9423f, 1.2562f, 1.6180f, 2.0690f, 2.7326f};


// round-trip a float through IEEE fp16 (the scales really are stored as fp16)
static float f16(float f) {
    union { float f; unsigned u; } v = { .f = f };
    unsigned u = v.u, sign = (u >> 16) & 0x8000;
    int exp = (int)((u >> 23) & 0xff) - 127 + 15;
    unsigned man = u & 0x7fffff;
    unsigned h;
    if (exp <= 0)       h = sign;                                  // underflow -> 0
    else if (exp >= 31) h = sign | 0x7c00;                         // overflow  -> inf
    else                h = sign | ((unsigned) exp << 10) | (man >> 13);
    unsigned s2 = (h & 0x8000) << 16;
    int e2 = (h >> 10) & 0x1f; unsigned m2 = h & 0x3ff;
    if (e2 == 0) { v.u = s2; return v.f; }
    v.u = s2 | ((unsigned)(e2 - 15 + 127) << 23) | (m2 << 13);
    return v.f;
}

static int lm3_code(float v) { int c=0; for (int i=0;i<7;i++) if (v>LM3_B[i]) c=i+1; return c; }
static int lm4_code(float v) { // nearest of 16
    int b=0; float bd=fabsf(v-LM4[0]);
    for (int i=1;i<16;i++){ float d=fabsf(v-LM4[i]); if(d<bd){bd=d;b=i;} } return b; }

// in-place normalized Walsh-Hadamard transform, n a power of two.
// Self-inverse once scaled by 1/sqrt(n), so encode and decode call the same fn.
static void wht(float * x, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float a = x[j], b = x[j + len];
                x[j] = a + b; x[j + len] = a - b;
            }
    const float s = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) x[i] *= s;
}

// --- tbq3: n values -> fp16 scale + n*3 bits ---
static void tbq_round(const float * src, float * dst, int n, int bits) {
    float t[1024]; memcpy(t, src, n*sizeof(float));
    wht(t, n);
    float ss = 0; for (int i=0;i<n;i++) ss += t[i]*t[i];
    float d = sqrtf(ss/n);                 // RMS: matches the unit-variance codebook
    if (d == 0) { memset(dst,0,n*sizeof(float)); return; }
    d = f16(d);             // scale really is stored as fp16
    for (int i=0;i<n;i++) {
        float v = t[i]/d;
        t[i] = d * (bits==3 ? LM3[lm3_code(v)] : LM4[lm4_code(v)]);
    }
    wht(t, n);                             // inverse == forward for normalized WHT
    memcpy(dst, t, n*sizeof(float));
}

// --- baselines: exactly what llama.cpp does today ---
static void q4_0_round(const float * src, float * dst, int n) {
    for (int b=0;b<n;b+=32) {
        float amax=0, max=0;
        for (int i=0;i<32;i++){ float a=fabsf(src[b+i]); if(a>amax){amax=a;max=src[b+i];} }
        float d = max/-8.0f; d = f16(d);
        float id = d ? 1.0f/d : 0.0f;
        for (int i=0;i<32;i++) {
            float x = src[b+i]*id + 8.5f;
            int q = (int)fminf(fmaxf(x,0.0f),15.0f);
            dst[b+i] = (q-8)*d;
        }
    }
}
static void q8_0_round(const float * src, float * dst, int n) {
    for (int b=0;b<n;b+=32) {
        float amax=0; for (int i=0;i<32;i++) amax=fmaxf(amax,fabsf(src[b+i]));
        float d = amax/127.0f; d = f16(d);
        float id = d ? 1.0f/d : 0.0f;
        for (int i=0;i<32;i++) dst[b+i] = roundf(src[b+i]*id)*d;
    }
}

static double nrmse(const float * a, const float * b, int n) {
    double se=0, ss=0;
    for (int i=0;i<n;i++){ double e=a[i]-b[i]; se+=e*e; ss+=(double)a[i]*a[i]; }
    return sqrt(se/(ss>0?ss:1));
}

static unsigned long rs=88172645463325252ULL;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (double)(rs>>11)/9007199254740992.0; }
static double gauss(void){ double u=urand(), v=urand(); if(u<1e-12)u=1e-12; return sqrt(-2*log(u))*cos(6.283185307*v); }

int main(int argc, char** argv) {
    int HD = 128;                // head_dim: the row length V is quantized along
    int ROWS = 20000;
    if (argc>1 && !strcmp(argv[1],"file")) {   // read real dims from the dump header
        FILE * h = fopen(argv[2],"rb"); if(!h){perror("open");return 1;}
        long long hd=0; if(fread(&hd,sizeof(hd),1,h)!=1) return 1;
        fseek(h,0,SEEK_END); long long nf=(ftell(h)-8)/4; fclose(h);
        HD=(int)hd; ROWS=(int)(nf/hd); if(ROWS>20000) ROWS=20000;
    }
    float *src = malloc(sizeof(float)*HD*ROWS);
    float *d3_32=malloc(sizeof(float)*HD*ROWS), *d3_128=malloc(sizeof(float)*HD*ROWS);
    float *d4_32=malloc(sizeof(float)*HD*ROWS);
    float *q4=malloc(sizeof(float)*HD*ROWS), *q8=malloc(sizeof(float)*HD*ROWS);

    const char * mode = argc>1 ? argv[1] : "outlier";
    if (!strcmp(mode,"file")) {
        // real V-cache rows captured from a running model via llama-tbq-dump
        FILE * f = fopen(argv[2], "rb");
        if (!f) { perror("open"); return 1; }
        long long hd = 0; if (fread(&hd,sizeof(hd),1,f)!=1) return 1;
        size_t got = fread(src, sizeof(float), (size_t)HD*ROWS, f);
        fclose(f);
        if (got < (size_t)HD*ROWS) { fprintf(stderr,"only %zu floats\n", got); return 1; }
    } else {
    for (int r=0;r<ROWS;r++) {
        float *row = src + (size_t)r*HD;
        for (int i=0;i<HD;i++) row[i] = (float) gauss();
    }
    if (!strcmp(mode,"outlier")) {
        int oc[4]; for (int k=0;k<4;k++) oc[k] = (int)(urand()*HD);
        float om[4]; for (int k=0;k<4;k++) om[k] = 20.0f + 60.0f*(float)urand();
        for (int r=0;r<ROWS;r++)
            for (int k=0;k<4;k++) src[(size_t)r*HD + oc[k]] *= om[k];
    }
    }
    for (int r=0;r<ROWS;r++) {
        size_t o=(size_t)r*HD;
        for (int b=0;b<HD;b+=32)  tbq_round(src+o+b, d3_32+o+b, 32, 3);
        for (int b=0;b<HD;b+=32)  tbq_round(src+o+b, d4_32+o+b, 32, 4);
        for (int b=0;b<HD;b+=128) tbq_round(src+o+b, d3_128+o+b, 128, 3);
        q4_0_round(src+o, q4+o, HD);
        q8_0_round(src+o, q8+o, HD);
    }
    int n = HD*ROWS;

    // What actually matters for a V cache is not per-element error but the
    // error in  out = softmax(scores) @ V  -- i.e. inner-product distortion.
    // TurboQuant's theorem is stated for exactly this, so measure it.
    const int NQ = 512, CTX = 2048;
    double eo[5] = {0}, so = 0;
    float * outs[5]; const float * cand[5] = { q8, q4, d4_32, d3_32, d3_128 };
    for (int k=0;k<5;k++) outs[k] = malloc(sizeof(float)*HD);
    float * ref = malloc(sizeof(float)*HD);
    float * p   = malloc(sizeof(float)*CTX);
    for (int qi=0; qi<NQ; qi++) {
        int base = (int)(urand()*(ROWS-CTX));
        double sum=0;
        for (int i=0;i<CTX;i++) { p[i] = (float) exp(gauss()*1.5); sum += p[i]; }
        for (int i=0;i<CTX;i++) p[i] /= (float) sum;      // a real softmax profile
        memset(ref,0,sizeof(float)*HD);
        for (int k=0;k<5;k++) memset(outs[k],0,sizeof(float)*HD);
        for (int i=0;i<CTX;i++) {
            size_t o=(size_t)(base+i)*HD; float w=p[i];
            for (int j=0;j<HD;j++) ref[j] += w*src[o+j];
            for (int k=0;k<5;k++) for (int j=0;j<HD;j++) outs[k][j] += w*cand[k][o+j];
        }
        for (int j=0;j<HD;j++) so += (double)ref[j]*ref[j];
        for (int k=0;k<5;k++) for (int j=0;j<HD;j++) { double e=outs[k][j]-ref[j]; eo[k]+=e*e; }
    }
    const char * nm[5] = {"q8_0","q4_0","tbq4 b=32","tbq3 b=32","tbq3 b=128"};
    const double bp[5] = {8.5,4.5,4.5,3.5,3.125};
    { // how outlier-heavy is this data, per channel?
        double best=0; int bc=0;
        for (int c=0;c<HD;c++){ double m=0,v=0; for(int r=0;r<ROWS;r++) m+=src[(size_t)r*HD+c]; m/=ROWS;
            for(int r=0;r<ROWS;r++){double d=src[(size_t)r*HD+c]-m; v+=d*d;} v=sqrt(v/ROWS);
            if(v>best){best=v;bc=c;} }
        double avg=0; for (int c=0;c<HD;c++){ double m=0,v=0; for(int r=0;r<ROWS;r++) m+=src[(size_t)r*HD+c]; m/=ROWS;
            for(int r=0;r<ROWS;r++){double d=src[(size_t)r*HD+c]-m; v+=d*d;} avg+=sqrt(v/ROWS); } avg/=HD;
        printf("  channel-outlier ratio (max sd / mean sd) = %.1fx  (channel %d)\n\n", best/avg, bc);
    }
    printf("data = %s, head_dim=%d, rows=%d\n\n", mode, HD, ROWS);
    printf("  %-12s %7s  %-9s  %s\n", "scheme", "bpw", "elem-NRMSE", "attn-out NRMSE (what matters)");
    const float * el[5] = { q8, q4, d4_32, d3_32, d3_128 };
    for (int k=0;k<5;k++)
        printf("  %-12s %7.4f  %-9.5f  %.5f\n", nm[k], bp[k], nrmse(src,el[k],n), sqrt(eo[k]/so));
    return 0;
}
