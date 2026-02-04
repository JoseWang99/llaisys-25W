#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    if (weight->ndim() != 2) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    if (index->ndim() != 1) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    if (out->ndim() != 2) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    if (index->dtype() != LLAISYS_DTYPE_I64) {
        EXCEPTION_DATATYPE_MISMATCH;
    }

    if (weight->dtype() != out->dtype()) {
        EXCEPTION_DATATYPE_MISMATCH;
    }

    // 检查形状
    size_t num_indices = index->numel();  // index 的长度
    size_t embedding_dim = weight->shape()[1];  // weight 的第二维度（嵌入维度）
    // out 的形状必须是 (num_indices, embedding_dim)
    if (out->shape()[0] != num_indices || out->shape()[1] != embedding_dim) {
        EXCEPTION_INVALID_ARGUMENT("out shape must be (num_indices, embedding_dim)");
    }
    // index 的长度必须等于 out 的第一维度
    if (index->shape()[0] != num_indices) {
        EXCEPTION_INVALID_ARGUMENT("index length must match out's first dimension");
    }
    const int64_t *index_data = reinterpret_cast<const int64_t *>(index->data());
    const std::byte *weight_data = weight->data();
    std::byte *out_data = out->data();

    size_t row_size_bytes = embedding_dim * weight->elementSize();

    // 对于每个索引，从weight 中复制对应行到out
    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = index_data[i];
        if (idx < 0 || static_cast<size_t>(idx) >= weight->shape()[0]) {
            EXCEPTION_INVALID_ARGUMENT("index out of range");
        }
        const std::byte *src = weight_data + idx * row_size_bytes;
        std::byte *dst = out_data + i * row_size_bytes;
        std::memcpy(dst, src, row_size_bytes);
    }
}
} // namespace llaisys::ops
