#pragma once

#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
void rearrange(tensor_t out, tensor_t in);
} // namespace llaisys::ops::cpu