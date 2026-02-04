#include "op.hpp"
#include <cmath>

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 基本检查
    if (out->ndim() != 2 || gate->ndim() != 2 || up->ndim() != 2) {
        EXCEPTION_INVALID_ARGUMENT("out, gate, up must be 2D tensors");
    }
    if (out->shape() != gate->shape() || out->shape() != up->shape()) {
        EXCEPTION_SHAPE_MISMATCH;
    }
    if (out->dtype() != gate->dtype() || out->dtype() != up->dtype()) {
        EXCEPTION_DATATYPE_MISMATCH;
    }
    // 仅 CPU 实现
    if (out->deviceType() != LLAISYS_DEVICE_CPU || gate->deviceType() != LLAISYS_DEVICE_CPU ||
        up->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    size_t elems = out->numel();

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *gate_data = reinterpret_cast<const float *>(gate->data());
        const float *up_data = reinterpret_cast<const float *>(up->data());
        float *out_data = reinterpret_cast<float *>(out->data());
        for (size_t i = 0; i < elems; ++i) {
            float g = gate_data[i];
            float u = up_data[i];
            float sig = g / (1.0f + std::exp(-g));
            out_data[i] = u * sig;
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *gate_data = reinterpret_cast<const fp16_t *>(gate->data());
        const fp16_t *up_data = reinterpret_cast<const fp16_t *>(up->data());
        fp16_t *out_data = reinterpret_cast<fp16_t *>(out->data());
        for (size_t i = 0; i < elems; ++i) {
            float g = utils::cast<float>(gate_data[i]);
            float u = utils::cast<float>(up_data[i]);
            float sig = g / (1.0f + std::exp(-g));
            out_data[i] = utils::cast<fp16_t>(u * sig);
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *gate_data = reinterpret_cast<const bf16_t *>(gate->data());
        const bf16_t *up_data = reinterpret_cast<const bf16_t *>(up->data());
        bf16_t *out_data = reinterpret_cast<bf16_t *>(out->data());
        for (size_t i = 0; i < elems; ++i) {
            float g = utils::cast<float>(gate_data[i]);
            float u = utils::cast<float>(up_data[i]);
            float sig = g / (1.0f + std::exp(-g));
            out_data[i] = utils::cast<bf16_t>(u * sig);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
