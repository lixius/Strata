// include/strata/prefill/iq_gemm_gfx906.hpp - the gfx906 prompt expert products on Strata's own kernel.
//
// Optional replacement for llama.cpp's MMQ on the prompt path (STRATA_GFX906_EXPERT_GEMM=1): the experts are
// unpacked once per chunk into signed int8 plus per-block float scales (the codebook, the signs and the post-dot
// integer transform folded into the scale), then multiplied by a register-tiled int8 GEMM.  Measured on an MI50
// at the model's shapes (16 experts a group, R rows each): 14-16 TOPS against MMQ's 10.7-13.3, i.e. 1.23-1.32x.
//
// The activation layout here is the kernel's own (32 values a block, rows back to back); it does not use the
// padded block_q8_1_mmq layout of llama.cpp's MMQ.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::iqgfx {

/// The build has the kernel (gfx906 HIP only).
bool built();
/// This ggml type is covered: IQ3_S, IQ3_XXS, IQ2_S, IQ4_NL.
bool supported(int ggml_type);
/// Bytes of the unpacked image of one matrix: `rows * cols` int8 values and `rows * cols/32 * 2` floats of scales.
size_t image_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Values / scales inside an image (separate buffers).
size_t values_bytes(int64_t rows, int64_t cols);
size_t scales_bytes(int64_t rows, int64_t cols);

/// q8_1 activations in the kernel's layout: T rows of `cols` values, 32 values (one 36-byte block) each.
size_t act_bytes(int64_t rows, int64_t cols);
void quantize_acts(const float* x, int64_t ld, int64_t rows, int64_t cols, void* out, void* stream);

/// One expert's gate|up and down matrices from their GGUF blocks into the int8 image.
/// `gu` = [2*n_ff, n_embd] (gate rows then up rows), `d` = [n_embd, n_ff]; each is a contiguous GGUF matrix.
void unpack_matrix(int ggml_type, const void* w, int64_t rows, int64_t cols, void* values, float* scales,
                   void* stream);
/// `n` experts at once: blobs[q]'s gate at +0, up at +up_off, down at +down_off (the pack's native layout).
void unpack_group(int ggml_type, const uint8_t* const* blob, size_t up_off, size_t down_off, int n, int64_t n_embd,
                  int64_t n_ff, void* gu_values, float* gu_scales, void* d_values, float* d_scales, void* stream);

/// The products of one group: for expert e, C rows [e*max_rows, e*max_rows+R_e) = X rows e*max_rows.. times the
/// expert's matrix (rows = output width, cols = K).  `xq` is quantize_acts' buffer for the whole group.
void gemm_group(int ggml_type, const void* w_values, const float* w_scales, const void* xq, float* out, int64_t rows,
                int64_t cols, int max_rows, int n_experts, void* stream);

}  // namespace strata::prefill::iqgfx
