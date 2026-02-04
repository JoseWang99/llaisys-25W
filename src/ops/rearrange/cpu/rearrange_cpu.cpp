#include "rearrange_cpu.hpp"

#include <cstring>
#include <functional>

namespace llaisys::ops::cpu {
void rearrange(tensor_t out, tensor_t in) {
    ASSERT(out != nullptr && in != nullptr, "rearrange: null tensor");
    ASSERT(out->dtype() == in->dtype(), "rearrange: dtype mismatch");
    ASSERT(out->shape() == in->shape(), "rearrange: shape mismatch");
    ASSERT(out->deviceType() == LLAISYS_DEVICE_CPU && in->deviceType() == LLAISYS_DEVICE_CPU,
           "rearrange: only CPU tensors are supported for now");

    const auto &shape = in->shape();
    const auto &src_strides = in->strides();
    const auto &dst_strides = out->strides();
    const size_t elem_size = in->elementSize();

    std::function<void(const std::byte *, std::byte *, size_t)> copy_nd;
    copy_nd = [&](const std::byte *src, std::byte *dst, size_t dim) {
        if (shape.empty()) {
            return;
        }
        if (dim == shape.size() - 1) {
            for (size_t i = 0; i < shape[dim]; ++i) {
                std::memcpy(dst + i * dst_strides[dim] * elem_size,
                            src + i * src_strides[dim] * elem_size,
                            elem_size);
            }
            return;
        }
        for (size_t i = 0; i < shape[dim]; ++i) {
            copy_nd(src + i * src_strides[dim] * elem_size,
                    dst + i * dst_strides[dim] * elem_size,
                    dim + 1);
        }
    };
    copy_nd(in->data(), out->data(), 0);
}
} // namespace llaisys::ops::cpu