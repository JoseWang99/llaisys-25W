#include "op.hpp"
#include "./cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rearrange(out, in);
    }
}
} // namespace llaisys::ops