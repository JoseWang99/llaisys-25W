#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../../core/context/context.hpp"
#include "../../utils/check.hpp"

// Ops (using direct HPP inclusion for internal implementation)
#include "../../ops/embedding/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace llaisys;

// --- Helper Functions ---

// Convert int64 shape to size_t shape for Tensor API
std::vector<size_t> to_dims(const std::vector<int64_t>& shape) {
    std::vector<size_t> dims;
    dims.reserve(shape.size());
    for (auto s : shape) dims.push_back(static_cast<size_t>(s));
    return dims;
}



// Wrapper to create tensor using int64 shape
tensor_t create_tensor(const std::vector<int64_t>& shape, llaisysDataType_t dtype, llaisysDeviceType_t device) {
    return Tensor::create(to_dims(shape), dtype, device);
}

// Helper to simulate Tensor::zeros
tensor_t zeros(const std::vector<int64_t>& shape, llaisysDataType_t dtype, llaisysDeviceType_t device) {
    tensor_t t = create_tensor(shape, dtype, device);
    if (device == LLAISYS_DEVICE_CPU) {
        std::memset(t->data(), 0, t->numel() * t->elementSize());
    }
    return t;
}

// ------------------------

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;

    // KV Cache: [layer][k|v] -> Shape [max_seq, nkvh, dh]
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;

    size_t pos; // Current sequence position
    llaisysDeviceType_t device_type;

    // Helper: Convert opaque C handle to C++ tensor_t (shared_ptr)
    tensor_t t(llaisysTensor_t handle) {
        if (!handle) return nullptr;
        return *reinterpret_cast<tensor_t*>(handle);
    }
    
    // Helper: Check if tensor loaded
    bool has(llaisysTensor_t handle) {
        return handle != nullptr;
    }
};

extern "C" {

LlaisysQwen2Model* llaisysQwen2ModelCreate(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int* device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->meta = *meta;
    model->device_type = static_cast<llaisysDeviceType_t>(device);
    model->pos = 0;

    // Initialize weight arrays
    size_t n = meta->nlayer;
    model->weights.attn_norm_w = new llaisysTensor_t[n]();
    model->weights.attn_q_w = new llaisysTensor_t[n]();
    model->weights.attn_q_b = new llaisysTensor_t[n](); // Note: Qwen2 usually has bias for q, k, v, o? Actually Qwen1.5/2 has bias for QKV but strict Qwen2 might vary. Assuming loaded if present.
    model->weights.attn_k_w = new llaisysTensor_t[n]();
    model->weights.attn_k_b = new llaisysTensor_t[n]();
    model->weights.attn_v_w = new llaisysTensor_t[n]();
    model->weights.attn_v_b = new llaisysTensor_t[n]();
    model->weights.attn_o_w = new llaisysTensor_t[n]();
    model->weights.mlp_norm_w = new llaisysTensor_t[n]();
    model->weights.mlp_gate_w = new llaisysTensor_t[n]();
    model->weights.mlp_up_w = new llaisysTensor_t[n]();
    model->weights.mlp_down_w = new llaisysTensor_t[n]();

    // Allocate KV Cache (Pre-allocate max size)
    // Structure: Vector of layers, each containing a tensor for K and V
    llaisysDataType_t dtype = static_cast<llaisysDataType_t>(meta->dtype);
    for (size_t i = 0; i < n; ++i) {
        std::vector<int64_t> cache_shape = {
            static_cast<int64_t>(meta->maxseq), 
            static_cast<int64_t>(meta->nkvh), 
            static_cast<int64_t>(meta->dh)
        };
        
        // Use local zeros helper
        model->k_cache.push_back(zeros(cache_shape, dtype, model->device_type));
        model->v_cache.push_back(zeros(cache_shape, dtype, model->device_type));
    }

    return model;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model* model) {
    if (!model) return;
    delete[] model->weights.attn_norm_w;
    delete[] model->weights.attn_q_w;
    delete[] model->weights.attn_q_b;
    delete[] model->weights.attn_k_w;
    delete[] model->weights.attn_k_b;
    delete[] model->weights.attn_v_w;
    delete[] model->weights.attn_v_b;
    delete[] model->weights.attn_o_w;
    delete[] model->weights.mlp_norm_w;
    delete[] model->weights.mlp_gate_w;
    delete[] model->weights.mlp_up_w;
    delete[] model->weights.mlp_down_w;
    delete model;
}

LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model* model) {
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model* model, int64_t* token_ids, size_t ntoken) {
    if (ntoken == 0) return -1;
    
    // --- 1. Prepare Inputs ---
    std::vector<int64_t> seq_shape = {static_cast<int64_t>(ntoken)};
    
    // Input Tokens [ntoken] - Use create_tensor wrapper
    tensor_t input = create_tensor(seq_shape, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    std::memcpy(input->data(), token_ids, ntoken * sizeof(int64_t));

    // Position IDs [ntoken]
    tensor_t pos_ids = create_tensor(seq_shape, ::LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    int64_t* pos_ptr = reinterpret_cast<int64_t*>(pos_ids->data());
    for(size_t i=0; i<ntoken; ++i) pos_ptr[i] = static_cast<int64_t>(model->pos + i);

    // --- 2. Embedding ---
    // Output: [ntoken, hidden_size]
    tensor_t hidden_states = create_tensor(
        {static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, 
        static_cast<llaisysDataType_t>(model->meta.dtype), 
        model->device_type
    );
    ops::embedding(hidden_states, input, model->t(model->weights.in_embed));

    // --- 3. Layers ---
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        tensor_t residual = hidden_states; // Shared pointer copy
        
        // RMSNorm expects same shape
        tensor_t norm_out = Tensor::create(hidden_states->shape(), hidden_states->dtype(), hidden_states->deviceType());
        ops::rms_norm(norm_out, hidden_states, model->t(model->weights.attn_norm_w[i]), model->meta.epsilon);

        // --- Attention Block ---
        // Q Projection: [ntoken, hs]
        tensor_t q_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(q_proj, norm_out, model->t(model->weights.attn_q_w[i]), model->t(model->weights.attn_q_b[i]));
        
        // K Projection: [ntoken, nkvh * dh]
        tensor_t k_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh * model->meta.dh)}, hidden_states->dtype(), model->device_type);
        ops::linear(k_proj, norm_out, model->t(model->weights.attn_k_w[i]), model->t(model->weights.attn_k_b[i]));

        // V Projection: [ntoken, nkvh * dh]
        tensor_t v_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh * model->meta.dh)}, hidden_states->dtype(), model->device_type);
        ops::linear(v_proj, norm_out, model->t(model->weights.attn_v_w[i]), model->t(model->weights.attn_v_b[i]));

        // Reshape for RoPE: [ntoken, nhead, dh]
        // Change: Tensor::reshape(t, shape) -> t->reshape(shape)
        tensor_t q = q_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nh), static_cast<int64_t>(model->meta.dh)}));
        tensor_t k = k_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh), static_cast<int64_t>(model->meta.dh)}));
        tensor_t v = v_proj->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nkvh), static_cast<int64_t>(model->meta.dh)}));
        // RoPE (In-place)
        ops::rope(q, q, pos_ids, model->meta.theta);
        ops::rope(k, k, pos_ids, model->meta.theta);

        // Update KV Cache
        tensor_t layer_k_cache = model->k_cache[i];
        tensor_t layer_v_cache = model->v_cache[i];
        
        {
            size_t row_size_bytes = model->meta.nkvh * model->meta.dh * k->elementSize();
            uint8_t* dst_k_base = reinterpret_cast<uint8_t*>(layer_k_cache->data());
            uint8_t* dst_v_base = reinterpret_cast<uint8_t*>(layer_v_cache->data());
            const uint8_t* src_k_base = reinterpret_cast<const uint8_t*>(k->data());
            const uint8_t* src_v_base = reinterpret_cast<const uint8_t*>(v->data());

             // Copy each token's KV
             // Important: assumes LLAISYS_DEVICE_CPU for memcpy
            for (size_t t = 0; t < ntoken; ++t) {
                size_t cache_idx = model->pos + t;
                if (cache_idx >= model->meta.maxseq) break; // Boundary check
                std::memcpy(dst_k_base + cache_idx * row_size_bytes, src_k_base + t * row_size_bytes, row_size_bytes);
                std::memcpy(dst_v_base + cache_idx * row_size_bytes, src_v_base + t * row_size_bytes, row_size_bytes);
            }
        }

        // View Hack: Modifying tensor shape in-place to emulate a view of the filled KV cache
        // We use const_cast because shape() returns const reference
        const std::vector<size_t> original_shape = layer_k_cache->shape(); 
        
        // This cast assumes Tensor stores shape in a standard container that can be cast to non-const
        std::vector<size_t>& mut_k_shape = const_cast<std::vector<size_t>&>(layer_k_cache->shape());
        std::vector<size_t>& mut_v_shape = const_cast<std::vector<size_t>&>(layer_v_cache->shape());
        
        mut_k_shape[0] = static_cast<size_t>(model->pos + ntoken);
        mut_v_shape[0] = static_cast<size_t>(model->pos + ntoken);

        // Attn Output Layout: [ntoken, nh, dh] -> Reshape to [ntoken, hs]
        tensor_t attn_out_view = create_tensor(
            {static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.nh), static_cast<int64_t>(model->meta.dh)}, 
            hidden_states->dtype(), model->device_type
        );
        
        float scale = 1.0f / sqrtf(static_cast<float>(model->meta.dh));
        
        // Run Attention
        ops::self_attention(attn_out_view, q, layer_k_cache, layer_v_cache, scale);

        // Restore Cache Shape
        mut_k_shape = original_shape;
        mut_v_shape = original_shape;

        // Merge Heads: [ntoken, hs]
        tensor_t attn_out = attn_out_view->reshape(to_dims({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}));

        // Output Projection (O_Proj)
        tensor_t o_linear = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(o_linear, attn_out, model->t(model->weights.attn_o_w[i]), nullptr);

        // Residual Connection 1
        ops::add(hidden_states, residual, o_linear);

        
        // --- MLP Block ---
        tensor_t mlp_input = hidden_states; // Updates accumulation
        tensor_t residual_mlp = mlp_input;  // Save for skip connection

        // Pre-MLP Norm
        tensor_t mlp_norm = Tensor::create(mlp_input->shape(), mlp_input->dtype(), mlp_input->deviceType());
        ops::rms_norm(mlp_norm, mlp_input, model->t(model->weights.mlp_norm_w[i]), model->meta.epsilon);

        // Gate & Up Projections
        tensor_t gate = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::linear(gate, mlp_norm, model->t(model->weights.mlp_gate_w[i]), nullptr);

        tensor_t up = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::linear(up, mlp_norm, model->t(model->weights.mlp_up_w[i]), nullptr);

        // SwiGLU Activation
        tensor_t act_out = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.di)}, hidden_states->dtype(), model->device_type);
        ops::swiglu(act_out, gate, up);

        // Down Projection
        tensor_t down_proj = create_tensor({static_cast<int64_t>(ntoken), static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
        ops::linear(down_proj, act_out, model->t(model->weights.mlp_down_w[i]), nullptr);

        // Residual Connection 2
        ops::add(hidden_states, residual_mlp, down_proj);
    }

    // --- 4. Final Processing ---
    
    // Final Norm
    tensor_t final_norm_out = Tensor::create(hidden_states->shape(), hidden_states->dtype(), hidden_states->deviceType());
    ops::rms_norm(final_norm_out, hidden_states, model->t(model->weights.out_norm_w), model->meta.epsilon);

    // Get the last token embedding
    tensor_t last_token_emb = create_tensor({1, static_cast<int64_t>(model->meta.hs)}, hidden_states->dtype(), model->device_type);
    
    size_t d_bytes = model->meta.hs * final_norm_out->elementSize();
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(last_token_emb->data());
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(final_norm_out->data());
    std::memcpy(dst_ptr, src_ptr + (ntoken - 1) * d_bytes, d_bytes);

    // LM Head Linear
    tensor_t logits = create_tensor({1, static_cast<int64_t>(model->meta.voc)}, hidden_states->dtype(), model->device_type);
    ops::linear(logits, last_token_emb, model->t(model->weights.out_embed), nullptr);

    // Argmax
    tensor_t out_token = create_tensor({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    
    // Create dimension tensor for argmax axis parameter
    // Expects tensor_t as per compilation error
    tensor_t dim_tensor = create_tensor({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU);
    int64_t dim_val = -1;
    std::memcpy(dim_tensor->data(), &dim_val, sizeof(int64_t));

    ops::argmax(out_token, logits, dim_tensor /* axis as tensor */);

    // Update global position
    model->pos += ntoken;

    // Update global position
    model->pos += ntoken;

    return *reinterpret_cast<int64_t*>(out_token->data());
}

} // extern "C"