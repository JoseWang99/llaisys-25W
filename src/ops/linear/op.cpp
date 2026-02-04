#include "op.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 检查维度
    // in 和 out 必须是 2D 张量
    if (in->ndim() != 2 || out->ndim() != 2) {
        EXCEPTION_SHAPE_MISMATCH;
    }
    // weight 必须是 2D 张量
    if (weight->ndim() != 2) {
        EXCEPTION_SHAPE_MISMATCH;
    }
    // bias 如果提供，必须是 1D 张量
    if (bias && bias->ndim() != 1) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    // 检查数据类型（假设所有张量数据类型相同）
    if (in->dtype() != out->dtype() || in->dtype() != weight->dtype() || (bias && in->dtype() != bias->dtype())) {
        EXCEPTION_INVALID_ARGUMENT("All tensors must have the same data type");
    }

    // 检查设备（假设都在 CPU 上）
    if (in->deviceType() != LLAISYS_DEVICE_CPU || out->deviceType() != LLAISYS_DEVICE_CPU ||
        weight->deviceType() != LLAISYS_DEVICE_CPU || (bias && bias->deviceType() != LLAISYS_DEVICE_CPU)) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // get shape
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    size_t weight_cols = weight->shape()[1];
    
    // 检查形状匹配
    if (in_features != weight_cols) {
        EXCEPTION_SHAPE_MISMATCH;
    }
    if (out->shape()[0] != batch_size || out->shape()[1] != out_features) {
        EXCEPTION_INVALID_ARGUMENT("out shape must be (batch_size, out_features)");
    }
    if (bias && bias->shape()[0] != out_features) {
        EXCEPTION_INVALID_ARGUMENT("bias shape must be (out_features,)");
    }

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *in_data = reinterpret_cast<const float *>(in->data());
        const float *weight_data = reinterpret_cast<const float *>(weight->data());
        float *out_data = reinterpret_cast<float *>(out->data());
        const float *bias_data = bias ? reinterpret_cast<const float *>(bias->data()) : nullptr;
        
        // 计算 Y = X * W^T + b
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < in_features; ++k) {
                    sum += in_data[i * in_features + k] * weight_data[j * in_features + k];
                }
                if (bias_data) {
                    sum += bias_data[j];
                }
                out_data[i * out_features + j] = sum;
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        const fp16_t *in_data = reinterpret_cast<const fp16_t *>(in->data());
        const fp16_t *weight_data = reinterpret_cast<const fp16_t *>(weight->data());
        fp16_t *out_data = reinterpret_cast<fp16_t *>(out->data());
        const fp16_t *bias_data = bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr;
        
        // 计算时转换到 float 以正确运算
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < in_features; ++k) {
                    // 转换 uint16_t 到 float（假设项目支持此转换；如果有 half 库，可优化）
                    float x = utils::cast<float>(in_data[i * in_features + k]);
                    float w = utils::cast<float>(weight_data[j * in_features + k]);
                    sum += x * w;
                }
                if (bias_data) {
                    sum += utils::cast<float>(bias_data[j]);
                }
                // 转换回 uint16_t
                out_data[i * out_features + j] = utils::cast<fp16_t>(sum);
            }
        }
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        const bf16_t *in_data = reinterpret_cast<const bf16_t *>(in->data());
        const bf16_t *weight_data = reinterpret_cast<const bf16_t *>(weight->data());
        bf16_t *out_data = reinterpret_cast<bf16_t *>(out->data());
        const bf16_t *bias_data = bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr;
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < in_features; ++k) {
                    float x = utils::cast<float>(in_data[i * in_features + k]);
                    float w = utils::cast<float>(weight_data[j * in_features + k]);
                    sum += x * w;
                }
                if (bias_data) {
                    sum += utils::cast<float>(bias_data[j]);
                }
                out_data[i * out_features + j] = utils::cast<bf16_t>(sum);
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops
