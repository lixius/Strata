// tests/hip/attn_chunk_bench.cpp - micro-benchmark and parity for the INT8-KV batched QSA attention kernel
// (`qsa_decode_attn_batch`, src/kernels/cuda/qsa_decode_attn.cu), which is also what the gfx906 prompt path runs
// (qsa_prompt_attn_batch refuses gfx906, so prefill.cpp falls back to this one, 32 queries at a time).
//
// It builds int8 pools with selections shaped like a real prompt (the 2,051-cell cap: a recent window plus a set
// that drifts), runs the kernel over `queries` of them in the prefill batch of 32, times it, and checks the
// output against an FP64 host reference and against a stored A/B hash.
//
// Usage: attn_chunk_bench [context=32768] [queries=2048] [reps=5]
// Env:   ACB_HASH=<file>   write the output bit-hash (for a bitwise A/B of two builds/switches)
//        ACB_REF=<file>    compare against a reference dump written by an earlier run
//        ACB_DUMP=<file>   write this run's output dump
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "attn_chunk_bench: %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "cudaMalloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float h2f(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}
uint16_t f2h(float f) {
    __half h = __float2half(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    const int fmt = argc > 4 ? std::atoi(argv[4]) : 1;   // 1 = INT8 pools (the engine's --kv int8), 0 = FP16 pools

    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    const int64_t cap = k::qsa_selection_width(ctx, s);

    // two generators: the pools' format must not move the selections, so an A/B across formats shares them
    std::mt19937 rng_sel(1234), rng_pool(999);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);

    std::vector<int8_t> kq((size_t) rows * HD), vq((size_t) rows * HD);
    std::vector<uint16_t> ks((size_t) rows * 4), vs((size_t) rows * 4);
    std::vector<uint16_t> kh((size_t) rows * HD), vh((size_t) rows * HD);
    if (fmt == 1) {
        for (auto& x : kq) x = (int8_t) code(rng_pool);
        for (auto& x : vq) x = (int8_t) code(rng_pool);
        for (auto& x : ks) x = f2h(sc(rng_pool));
        for (auto& x : vs) x = f2h(sc(rng_pool));
    } else {
        for (auto& x : kh) x = f2h(nd(rng_pool) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng_pool));
    }
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;

    // selections: query i sits at position ctx - nq + i; the widest cap cells, a recent window plus an older set
    // that drifts ~3% per query (as a real prompt's do)
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng_sel) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) w;
        if (w == nkv) {
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[(size_t) c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng_sel);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng_sel);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[(size_t) pick(rng_sel)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }

    k::QsaAttnPools pl;
    if (fmt == 1) {
        pl.k_q = up(kq);
        pl.v_q = up(vq);
        pl.k_scale = up(ks);
        pl.v_scale = up(vs);
    } else {
        pl.k_pool = up(kh);
        pl.v_pool = up(vh);
    }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);

    const int64_t batch = 32;
    float* d_out = nullptr;
    float* scratch = nullptr;
    ck(cudaMalloc(&d_out, (size_t) (nq * NH * HD) * 4), "cudaMalloc out");
    ck(cudaMalloc(&scratch, (size_t) (batch * k::qsa_decode_attn_scratch_floats(cap, s)) * 4), "cudaMalloc scratch");

    auto run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_out + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };

    run();
    ck(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev");
    ck(cudaEventCreate(&e1), "ev");
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        ck(cudaEventRecord(e0), "rec");
        run();
        ck(cudaEventRecord(e1), "rec");
        ck(cudaEventSynchronize(e1), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, e0, e1), "elapsed");
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    // cells actually attended, for a MAC rate
    double macs = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        macs += (double) w * (double) NH * (double) HD * 2.0;   // q.k and p.v
    }
    std::printf("attn_chunk: ctx %lld  nq %lld  cap %lld  %s/%s kernel: %.3f ms  (%.2f TFLOP/s eff)\n", (long long) ctx,
                (long long) nq, (long long) cap, fmt == 1 ? "int8" : "fp16",
                getenv("STRATA_GFX906_ATTN_DOT2") ? "DOT2" : "base", med, macs * 2.0 / (med * 1e-3) / 1e12);

    std::vector<float> out((size_t) (nq * NH * HD));
    ck(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost), "down");

    uint64_t hh = 1469598103934665603ull;
    for (float x : out) {
        uint32_t u;
        std::memcpy(&u, &x, 4);
        hh = (hh ^ u) * 1099511628211ull;
    }
    std::printf("OUTHASH %016llx\n", (unsigned long long) hh);
    if (const char* f = std::getenv("ACB_HASH")) {
        std::FILE* fp = std::fopen(f, "w");
        if (fp) {
            std::fprintf(fp, "%016llx\n", (unsigned long long) hh);
            std::fclose(fp);
        }
    }
    if (const char* f = std::getenv("ACB_DUMP")) {
        std::FILE* fp = std::fopen(f, "wb");
        if (fp) {
            std::fwrite(out.data(), 4, out.size(), fp);
            std::fclose(fp);
        }
    }
    if (const char* f = std::getenv("ACB_REF")) {
        std::FILE* fp = std::fopen(f, "rb");
        if (!fp) {
            std::printf("ACB_REF: cannot open %s\n", f);
            return 2;
        }
        std::vector<float> ref(out.size());
        const size_t got = std::fread(ref.data(), 4, ref.size(), fp);
        std::fclose(fp);
        if (got != ref.size()) {
            std::printf("ACB_REF: size mismatch\n");
            return 2;
        }
        double maxabs = 0, maxrel = 0, sum2 = 0, ref2 = 0;
        int exact = 0, nonfinite = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            if (!std::isfinite(out[i])) ++nonfinite;
            if (out[i] == ref[i]) ++exact;
            const double d = std::fabs((double) out[i] - (double) ref[i]);
            maxabs = std::max(maxabs, d);
            sum2 += d * d;
            ref2 += (double) ref[i] * (double) ref[i];
            const double den = std::fabs((double) ref[i]);
            if (den > 1e-3) maxrel = std::max(maxrel, d / den);
        }
        std::printf("ACB_REF: bit-identical %d/%zu (%.2f%%)  max|d| %.3e  rms %.3e (ref rms %.3e -> %.3f%%)  max rel %.3e  nonfinite %d\n",
                    exact, out.size(), 100.0 * exact / out.size(), maxabs, std::sqrt(sum2 / out.size()),
                    std::sqrt(ref2 / out.size()), 100.0 * std::sqrt(sum2 / out.size()) / std::sqrt(ref2 / out.size()),
                    maxrel, nonfinite);
    }
    return 0;
}
