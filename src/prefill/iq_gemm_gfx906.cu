// src/prefill/iq_gemm_gfx906.cu - see include/strata/prefill/iq_gemm_gfx906.hpp.
//
// The prompt expert products on Strata's own kernel for gfx906: one unpack pass per chunk turns each expert's
// GGUF blocks into signed int8 plus per-block float scales (codebook, signs and llama.cpp's post-dot integer
// transform folded into the scale - exact in real arithmetic, the only difference being its <= 1 LSB integer
// truncation), then a register-tiled int8 GEMM (8x4 per lane, 64 lanes, 64x32 tile, 8 V_DOT4 per K-step).
// 1.23-1.32x llama.cpp's MMQ at the model's shapes (16 experts a group, R = 40..626).
#include "strata/prefill/iq_gemm_gfx906.hpp"

#if defined(STRATA_HIP_GFX906)

#include "common.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::iqgfx {
namespace {

__device__ __forceinline__ int sdot4(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
__device__ __forceinline__ int vsub4_sat(int a, int b) {   // __vsubss4, per byte, saturating
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int x = (int) (int8_t) ((a >> (8 * i)) & 0xFF) - (int) (int8_t) ((b >> (8 * i)) & 0xFF);
        x = x < -128 ? -128 : (x > 127 ? 127 : x);
        r |= (x & 0xFF) << (8 * i);
    }
    return r;
}
__device__ __forceinline__ unsigned vcmpne4(unsigned a, unsigned b) {
    unsigned r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned x = (a >> (8 * i)) & 0xFF, y = (b >> (8 * i)) & 0xFF;
        r |= (x == y ? 0u : 0xFFu) << (8 * i);
    }
    return r;
}

// one 32-value block (sub-block s of a 256-value block) -> 8 signed-int8 dwords; returns the block's scale
__device__ __forceinline__ float deq3s(const block_iq3_s& b, int s, int* o) {
    const uint8_t* qs = b.qs + 8 * s;
    const int qh = b.qh[s];
    const uint8_t* sg = b.signs + 4 * s;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned g0 = iq3s_grid[qs[2 * j + 0] | ((qh << (8 - 2 * j)) & 0x100)];
        const unsigned g1 = iq3s_grid[qs[2 * j + 1] | ((qh << (7 - 2 * j)) & 0x100)];
        const unsigned sb = sg[j];
        const unsigned m0 = vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0u);
        const unsigned m1 = vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0u);
        o[2 * j + 0] = vsub4_sat((int) (g0 ^ m0), (int) m0);
        o[2 * j + 1] = vsub4_sat((int) (g1 ^ m1), (int) m1);
    }
    return __half2float(__ushort_as_half(b.d)) * (float) (1 + 2 * ((b.scales[s >> 1] >> (4 * (s & 1))) & 0xF));
}
__device__ __forceinline__ float deq3xxs(const block_iq3_xxs& b, int s, int* o) {
    const uint8_t* q3 = b.qs + 8 * s;
    const uint32_t aux = *(const uint32_t*) (b.qs + 64 + 4 * s);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned g0 = iq3xxs_grid[q3[2 * j + 0]];
        const unsigned g1 = iq3xxs_grid[q3[2 * j + 1]];
        const unsigned ks = (unsigned) ksigns_iq2xs[(aux >> (7 * j)) & 127] * 0x01010101u;
        const unsigned m0 = vcmpne4(ks & 0x08040201u, 0u);
        const unsigned m1 = vcmpne4(ks & 0x80402010u, 0u);
        o[2 * j + 0] = vsub4_sat((int) (g0 ^ m0), (int) m0);
        o[2 * j + 1] = vsub4_sat((int) (g1 ^ m1), (int) m1);
    }
    return __half2float(__ushort_as_half(b.d)) * (float) (2 * (aux >> 28) + 1) * 0.25f;
}
__device__ __forceinline__ void deq2s(const block_iq2_s& b, int s, int* o, float sc[2]) {
    const uint8_t* qs = b.qs + 4 * s;
    const int qh = b.qh[s];
    const uint8_t* sg = b.qs + 32 + 4 * s;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = qs[j] | ((qh << (8 - 2 * j)) & 0x300);
        const int g0 = *(const int*) &iq2s_grid[2 * idx + 0];
        const int g1 = *(const int*) &iq2s_grid[2 * idx + 1];
        const unsigned sb = sg[j];
        const unsigned m0 = vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0u);
        const unsigned m1 = vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0u);
        o[2 * j + 0] = vsub4_sat((int) (g0 ^ (int) m0), (int) m0);
        o[2 * j + 1] = vsub4_sat((int) (g1 ^ (int) m1), (int) m1);
    }
    const float d = __half2float(__ushort_as_half(b.d));
    sc[0] = d * (float) (2 * (b.scales[s] & 0xF) + 1) * 0.25f;
    sc[1] = d * (float) (2 * (b.scales[s] >> 4) + 1) * 0.25f;
}
__device__ __forceinline__ float deq4nl(const block_iq4_nl& b, int* o) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t byte = b.qs[4 * k + i];
            lo |= (unsigned) (uint8_t) kvalues_iq4nl[byte & 0x0F] << (8 * i);
            hi |= (unsigned) (uint8_t) kvalues_iq4nl[byte >> 4] << (8 * i);
        }
        o[k] = (int) lo;
        o[k + 4] = (int) hi;
    }
    return __half2float(__ushort_as_half(b.d));
}

// ---- unpack: one matrix (rows x cols) of one expert ----------------------------------------------------------
__global__ void unpack_kernel(int t, const unsigned char* __restrict__ W, signed char* __restrict__ U,
                              float* __restrict__ S, int rows, int nsub) {
    const int sb = blockIdx.x * 64 + threadIdx.x;
    if (sb >= nsub) return;
    const int r = blockIdx.y;
    int o[8];
    float s0 = 1.0f, s1 = 1.0f;
    const int kb = sb >> 3, s = sb & 7;
    if (t == GGML_TYPE_IQ3_S) {
        s0 = s1 = deq3s(((const block_iq3_s*) W)[(size_t) r * (nsub >> 3) + kb], s, o);
    } else if (t == GGML_TYPE_IQ3_XXS) {
        s0 = s1 = deq3xxs(((const block_iq3_xxs*) W)[(size_t) r * (nsub >> 3) + kb], s, o);
    } else if (t == GGML_TYPE_IQ2_S) {
        float sc2[2] = {1.0f, 1.0f};
        deq2s(((const block_iq2_s*) W)[(size_t) r * (nsub >> 3) + kb], s, o, sc2);
        s0 = sc2[0];
        s1 = sc2[1];
    } else {
        s0 = s1 = deq4nl(((const block_iq4_nl*) W)[(size_t) r * nsub + sb], o);
    }
    int* d = (int*) (U + ((size_t) r * nsub + sb) * 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = o[i];
    const size_t si = ((size_t) r * nsub + sb) * 2;
    S[si + 0] = s0;
    S[si + 1] = s1;
}

// ---- q8_1 activations in this kernel's own layout -------------------------------------------------------------
__global__ void quantize_kernel(const float* __restrict__ x, int64_t ld, int64_t rows, int64_t cols,
                                block_q8_1* __restrict__ out) {
    const int64_t blk = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one 32-value block
    const int64_t nsub = cols / 32;
    const int64_t r = blk / nsub, sb = blk % nsub;
    if (r >= rows) return;
    const float* src = x + r * ld + sb * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(src[i]));
    const float d = amax / 127.0f, id = amax != 0.0f ? 127.0f / amax : 0.0f;
    block_q8_1& b = out[r * nsub + sb];
    b.ds = __halves2half2(__float2half(d), __float2half(0.0f));
    for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) roundf(src[i] * id);
}

// ---- the GEMM: 8x4 per lane, 64 lanes, tile 64 x 32 ----------------------------------------------------------
constexpr int TH = 8, TW = 4, MT = 64, NT = 32;

template <bool SPLIT>   // IQ2_S: two 16-value accumulators with their own scales
__global__ void __launch_bounds__(64, 8) gemm_kernel(const signed char* __restrict__ U, const float* __restrict__ S,
                                                     const block_q8_1* __restrict__ X, float* __restrict__ C,
                                                     int nsub, int R, int N, int E) {
    __shared__ signed char sx[MT][36];
    __shared__ signed char sw[NT][36];
    __shared__ float sxs[MT][2], sws[NT][2];
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int n0 = blockIdx.x * NT, m0 = blockIdx.y * MT;
    const int mg = (t >> 3) * 8, ng = (t & 7) * 4;

    float acc[TH][TW];
#pragma unroll
    for (int i = 0; i < TH; ++i)
#pragma unroll
        for (int j = 0; j < TW; ++j) acc[i][j] = 0.0f;

    for (int sb = 0; sb < nsub; ++sb) {
        for (int r = t; r < MT + NT; r += 64) {
            if (r < MT) {
                const int m = m0 + r;
                int4 a = make_int4(0, 0, 0, 0), bb = make_int4(0, 0, 0, 0);
                float xs = 0.0f;
                if (m < R) {
                    const block_q8_1& blk = X[(size_t) (e * R + m) * nsub + sb];
                    a = ((const int4*) blk.qs)[0];
                    bb = ((const int4*) blk.qs)[1];
                    xs = __half2float(__low2half(blk.ds));
                }
                int* d = (int*) sx[r];
                d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w;
                d[4] = bb.x; d[5] = bb.y; d[6] = bb.z; d[7] = bb.w;
                sxs[r][0] = sxs[r][1] = xs;
            } else {
                const int n = n0 + (r - MT);
                int4 a = make_int4(0, 0, 0, 0), bb = make_int4(0, 0, 0, 0);
                float w0 = 0.0f, w1 = 0.0f;
                if (n < N) {
                    const long long off = ((long long) (e * N + n) * nsub + sb) * 32;
                    a = ((const int4*) (U + off))[0];
                    bb = ((const int4*) (U + off))[1];
                    const long long so = ((long long) (e * N + n) * nsub + sb) * 2;
                    w0 = S[so + 0];
                    w1 = S[so + 1];
                }
                int* d = (int*) sw[r - MT];
                d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w;
                d[4] = bb.x; d[5] = bb.y; d[6] = bb.z; d[7] = bb.w;
                sws[r - MT][0] = w0;
                sws[r - MT][1] = w1;
            }
        }
        __syncthreads();
        if (!SPLIT) {
            int sum[TH][TW];
#pragma unroll
            for (int i = 0; i < TH; ++i)
#pragma unroll
                for (int j = 0; j < TW; ++j) sum[i][j] = 0;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int xv[TH], wv[TW];
#pragma unroll
                for (int i = 0; i < TH; ++i) xv[i] = *(const int*) &sx[mg + i][kk * 4];
#pragma unroll
                for (int j = 0; j < TW; ++j) wv[j] = *(const int*) &sw[ng + j][kk * 4];
#pragma unroll
                for (int i = 0; i < TH; ++i)
#pragma unroll
                    for (int j = 0; j < TW; ++j) sum[i][j] = sdot4(wv[j], xv[i], sum[i][j]);
            }
            float wq[TW];
#pragma unroll
            for (int j = 0; j < TW; ++j) wq[j] = sws[ng + j][0];
#pragma unroll
            for (int i = 0; i < TH; ++i) {
                const float xq = sxs[mg + i][0];
#pragma unroll
                for (int j = 0; j < TW; ++j) acc[i][j] = fmaf((float) sum[i][j], xq * wq[j], acc[i][j]);
            }
        } else {
            int sum0[TH][TW], sum1[TH][TW];
#pragma unroll
            for (int i = 0; i < TH; ++i)
#pragma unroll
                for (int j = 0; j < TW; ++j) { sum0[i][j] = 0; sum1[i][j] = 0; }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int xv[TH], wv[TW];
#pragma unroll
                for (int i = 0; i < TH; ++i) xv[i] = *(const int*) &sx[mg + i][kk * 4];
#pragma unroll
                for (int j = 0; j < TW; ++j) wv[j] = *(const int*) &sw[ng + j][kk * 4];
                if (kk < 4) {
#pragma unroll
                    for (int i = 0; i < TH; ++i)
#pragma unroll
                        for (int j = 0; j < TW; ++j) sum0[i][j] = sdot4(wv[j], xv[i], sum0[i][j]);
                } else {
#pragma unroll
                    for (int i = 0; i < TH; ++i)
#pragma unroll
                        for (int j = 0; j < TW; ++j) sum1[i][j] = sdot4(wv[j], xv[i], sum1[i][j]);
                }
            }
#pragma unroll
            for (int i = 0; i < TH; ++i) {
                const float xq = sxs[mg + i][0];
#pragma unroll
                for (int j = 0; j < TW; ++j) {
                    acc[i][j] = fmaf((float) sum0[i][j], xq * sws[ng + j][0], acc[i][j]);
                    acc[i][j] = fmaf((float) sum1[i][j], xq * sws[ng + j][1], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TH; ++i) {
        const int m = m0 + mg + i;
        if (m >= R) continue;
#pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int n = n0 + ng + j;
            if (n < N) C[(size_t) (e * R + m) * N + n] = acc[i][j];
        }
    }
}

}  // namespace

bool built() { return true; }
bool supported(int t) {
    return t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ4_NL;
}
size_t values_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) cols; }
size_t scales_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 32) * 2 * sizeof(float); }
size_t image_bytes(int t, int64_t rows, int64_t cols) { (void) t; return values_bytes(rows, cols) + scales_bytes(rows, cols); }
size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) (cols / 32) * sizeof(block_q8_1); }

void quantize_acts(const float* x, int64_t ld, int64_t rows, int64_t cols, void* out, void* stream) {
    const int64_t total = rows * (cols / 32);
    const int th = 256;
    quantize_kernel<<<(int) ((total + th - 1) / th), th, 0, (cudaStream_t) stream>>>(x, ld, rows, cols,
                                                                                    (block_q8_1*) out);
}

void unpack_matrix(int t, const void* w, int64_t rows, int64_t cols, void* values, float* scales, void* stream) {
    const int nsub = (int) (cols / 32);
    dim3 g((nsub + 63) / 64, (unsigned) rows);
    unpack_kernel<<<g, 64, 0, (cudaStream_t) stream>>>(t, (const unsigned char*) w, (signed char*) values, scales,
                                                       (int) rows, nsub);
}

void unpack_group(int t, const uint8_t* const* blob, size_t up_off, size_t down_off, int n, int64_t n_embd, int64_t n_ff,
                  void* gu_values, float* gu_scales, void* d_values, float* d_scales, void* stream) {
    const int nsub_gu = (int) (n_embd / 32), nsub_d = (int) (n_ff / 32);
    for (int q = 0; q < n; ++q) {
        unpack_matrix(t, blob[q], 2 * n_ff, n_embd, (signed char*) gu_values + (size_t) q * 2 * n_ff * n_embd,
                      gu_scales + (size_t) q * 2 * n_ff * nsub_gu * 2, stream);
        unpack_matrix(t, blob[q] + down_off, n_embd, n_ff, (signed char*) d_values + (size_t) q * n_embd * n_ff,
                      d_scales + (size_t) q * n_embd * nsub_d * 2, stream);
    }
    (void) up_off;
}

void gemm_group(int t, const void* w_values, const float* w_scales, const void* xq, float* out, int64_t rows,
                int64_t cols, int max_rows, int n_experts, void* stream) {
    const int nsub = (int) (cols / 32);
    dim3 g((unsigned) ((rows + NT - 1) / NT), (unsigned) ((max_rows + MT - 1) / MT), (unsigned) n_experts);
    if (t == GGML_TYPE_IQ2_S)
        gemm_kernel<true><<<g, 64, 0, (cudaStream_t) stream>>>((const signed char*) w_values, w_scales,
                                                              (const block_q8_1*) xq, out, nsub, max_rows, (int) rows,
                                                              n_experts);
    else
        gemm_kernel<false><<<g, 64, 0, (cudaStream_t) stream>>>((const signed char*) w_values, w_scales,
                                                               (const block_q8_1*) xq, out, nsub, max_rows, (int) rows,
                                                               n_experts);
}

}  // namespace strata::prefill::iqgfx

#else

namespace strata::prefill::iqgfx {
bool built() { return false; }
bool supported(int) { return false; }
size_t values_bytes(int64_t, int64_t) { return 0; }
size_t scales_bytes(int64_t, int64_t) { return 0; }
size_t image_bytes(int, int64_t, int64_t) { return 0; }
size_t act_bytes(int64_t, int64_t) { return 0; }
void quantize_acts(const float*, int64_t, int64_t, int64_t, void*, void*) {}
void unpack_matrix(int, const void*, int64_t, int64_t, void*, float*, void*) {}
void unpack_group(int, const uint8_t* const*, size_t, size_t, int, int64_t, int64_t, void*, float*, void*, float*, void*) {}
void gemm_group(int, const void*, const float*, const void*, float*, int64_t, int64_t, int, int, void*) {}
}  // namespace strata::prefill::iqgfx

#endif
