#include "op.hpp"
#include <cmath> 

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 检查维度
    if (in->ndim() != 3 || out->ndim() != 3) {
        EXCEPTION_INVALID_ARGUMENT("in and out must be 3D tensors");
    }
    if (pos_ids->ndim() != 1) {
        EXCEPTION_INVALID_ARGUMENT("pos_ids must be 1D tensor");
    }
    
    // 检查数据类型
    if (in->dtype() != out->dtype()) {
        EXCEPTION_INVALID_ARGUMENT("in and out must have the same data type");
    }
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        EXCEPTION_INVALID_ARGUMENT("pos_ids must be int64");
    }
    
    // 检查设备（假设都在 CPU 上）
    if (in->deviceType() != LLAISYS_DEVICE_CPU || out->deviceType() != LLAISYS_DEVICE_CPU ||
        pos_ids->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    // 获取形状
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    
    // 检查形状匹配
    if (out->shape()[0] != seqlen || out->shape()[1] != nhead || out->shape()[2] != d) {
        EXCEPTION_INVALID_ARGUMENT("out shape must match in shape");
    }
    if (pos_ids->shape()[0] != seqlen) {
        EXCEPTION_INVALID_ARGUMENT("pos_ids shape must be (seqlen,)");
    }
    if (d % 2 != 0) {
        EXCEPTION_INVALID_ARGUMENT("d must be even");
    }

    const int64_t *pos_ids_data = reinterpret_cast<const int64_t *>(pos_ids->data());
    size_t half_d = d / 2;
    
    // 预计算频率向量 freq[j] = theta^{2j / d} for j in 0..half_d-1
    std::vector<double> freq(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        freq[j] = std::pow(static_cast<double>(theta), -2.0 * j / static_cast<double>(d));
    }
    
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *in_data = reinterpret_cast<const float *>(in->data());
        float *out_data = reinterpret_cast<float *>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_ids_data[i];
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    float phi = pos * freq[j];
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    size_t idx_a = i * nhead * d + h * d + j;
                    size_t idx_b = i * nhead * d + h * d + j + half_d;
                    float a = in_data[idx_a];
                    float b = in_data[idx_b];
                    
                    float a_prime = a * cos_phi - b * sin_phi;
                    float b_prime = b * cos_phi + a * sin_phi;
                    
                    out_data[idx_a] = a_prime;
                    out_data[idx_b] = b_prime;
                }
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_F16: {
        const fp16_t *in_data = reinterpret_cast<const fp16_t *>(in->data());
        fp16_t *out_data = reinterpret_cast<fp16_t *>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_ids_data[i];
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    float phi = pos * freq[j];
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    size_t idx_a = i * nhead * d + h * d + j;
                    size_t idx_b = i * nhead * d + h * d + j + half_d;
                    float a = utils::cast<float>(in_data[idx_a]);
                    float b = utils::cast<float>(in_data[idx_b]);
                    
                    float a_prime = a * cos_phi - b * sin_phi;
                    float b_prime = b * cos_phi + a * sin_phi;
                    
                    out_data[idx_a] = utils::cast<fp16_t>(a_prime);
                    out_data[idx_b] = utils::cast<fp16_t>(b_prime);
                }
            }
        }
        break;
    }
    
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *in_data = reinterpret_cast<const bf16_t *>(in->data());
        bf16_t *out_data = reinterpret_cast<bf16_t *>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_ids_data[i];
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    float phi = pos * freq[j];
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    size_t idx_a = i * nhead * d + h * d + j;
                    size_t idx_b = i * nhead * d + h * d + j + half_d;
                    float a = utils::cast<float>(in_data[idx_a]);
                    float b = utils::cast<float>(in_data[idx_b]);
                    
                    float a_prime = a * cos_phi - b * sin_phi;
                    float b_prime = b * cos_phi + a * sin_phi;
                    
                    out_data[idx_a] = utils::cast<bf16_t>(a_prime);
                    out_data[idx_b] = utils::cast<bf16_t>(b_prime);
                }
            }
        }
        break;
    }
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }    

}
} // namespace llaisys::ops
