#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 检查维度
    if (in->ndim() != 2 || out->ndim() != 2) {
        EXCEPTION_INVALID_ARGUMENT("in and out must be 2D tensors");
    }
    if (weight->ndim() != 1) {
        EXCEPTION_INVALID_ARGUMENT("weight must be 1D tensor");
    }
    
    // 检查数据类型（假设所有张量数据类型相同）
    if (in->dtype() != out->dtype() || in->dtype() != weight->dtype()) {
        EXCEPTION_INVALID_ARGUMENT("All tensors must have the same data type");
    }
    
    // 检查设备（假设都在 CPU 上）
    if (in->deviceType() != LLAISYS_DEVICE_CPU || out->deviceType() != LLAISYS_DEVICE_CPU ||
        weight->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    // 获取形状
    size_t batch_size = in->shape()[0];
    size_t d = in->shape()[1];
    
    // 检查形状匹配
    if (out->shape()[0] != batch_size || out->shape()[1] != d) {
        EXCEPTION_INVALID_ARGUMENT("out shape must match in shape");
    }
    if (weight->shape()[0] != d) {
        EXCEPTION_INVALID_ARGUMENT("weight shape must be (d,)");
    }
    
    // 根据数据类型分支处理
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *in_data = reinterpret_cast<const float *>(in->data());
        const float *weight_data = reinterpret_cast<const float *>(weight->data());
        float *out_data = reinterpret_cast<float *>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            // 计算该行的 sum of squares
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                float x = in_data[i * d + j];
                sum_sq += x * x;
            }
            // 计算 RMS
            float rms = std::sqrt((sum_sq / d) + eps);
            // 应用 normalization 和 weight
            for (size_t j = 0; j < d; ++j) {
                out_data[i * d + j] = weight_data[j] * (in_data[i * d + j] / rms);
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        const fp16_t *in_data = reinterpret_cast<const fp16_t *>(in->data());
        const fp16_t *weight_data = reinterpret_cast<const fp16_t *>(weight->data());
        fp16_t *out_data = reinterpret_cast<fp16_t *>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                float x = utils::cast<float>(in_data[i * d + j]);
                sum_sq += x * x;
            }
            float rms = std::sqrt((sum_sq / d) + eps);
            for (size_t j = 0; j < d; ++j) {
                float x = utils::cast<float>(in_data[i * d + j]);
                float w = utils::cast<float>(weight_data[j]);
                out_data[i * d + j] = utils::cast<fp16_t>(w * (x / rms));
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *in_data = reinterpret_cast<const bf16_t *>(in->data());
        const bf16_t *weight_data = reinterpret_cast<const bf16_t *>(weight->data());
        bf16_t *out_data = reinterpret_cast<bf16_t *>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                float x = utils::cast<float>(in_data[i * d + j]);
                sum_sq += x * x;
            }
            float rms = std::sqrt((sum_sq / d) + eps);
            for (size_t j = 0; j < d; ++j) {
                float x = utils::cast<float>(in_data[i * d + j]);
                float w = utils::cast<float>(weight_data[j]);
                out_data[i * d + j] = utils::cast<bf16_t>(w * (x / rms));
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
