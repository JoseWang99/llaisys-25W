#include "op.hpp"
#include <cmath>
#include <limits>


namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 检查维度
    if (q->ndim() != 3 || k->ndim() != 3 || v->ndim() != 3 || attn_val->ndim() != 3) {
        EXCEPTION_INVALID_ARGUMENT("q, k, v, attn_val must be 3D tensors");
    }
    
    // 检查数据类型
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype() || q->dtype() != attn_val->dtype()) {
        EXCEPTION_INVALID_ARGUMENT("All tensors must have the same data type");
    }
    
    // 检查设备（假设 CPU）
    if (q->deviceType() != LLAISYS_DEVICE_CPU || k->deviceType() != LLAISYS_DEVICE_CPU ||
        v->deviceType() != LLAISYS_DEVICE_CPU || attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
    // 获取形状
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    // 检查形状匹配
    if (k->shape()[2] != d || v->shape()[0] != total_len || v->shape()[1] != nkvhead) {
        EXCEPTION_INVALID_ARGUMENT("k and v shapes mismatch");
    }
    if (attn_val->shape()[0] != seqlen || attn_val->shape()[1] != nhead || attn_val->shape()[2] != dv) {
        EXCEPTION_INVALID_ARGUMENT("attn_val shape mismatch");
    }
    if (seqlen > total_len) {
        EXCEPTION_INVALID_ARGUMENT("seqlen cannot exceed total_len");
    }
    // 支持 nkvhead == 1 (MQA) 或 nkvhead == nhead
    if (nkvhead != 1 && nhead % nkvhead != 0) {
        EXCEPTION_INVALID_ARGUMENT("nkvhead must be 1 or divide nhead");
    }

    size_t group_size = (nkvhead == 1) ? 1 : (nhead / nkvhead);
    
    // mask 对角偏移：torch 中使用 diagonal = S - L
    int diag = static_cast<int>(total_len) - static_cast<int>(seqlen); // 可以为负或0或正
    // 注意：允许的 j 满足 j <= i + diag

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *q_data = reinterpret_cast<const float *>(q->data());
        const float *k_data = reinterpret_cast<const float *>(k->data());
        const float *v_data = reinterpret_cast<const float *>(v->data());
        float *attn_val_data = reinterpret_cast<float *>(attn_val->data());

        // A: [seqlen, nhead, total_len]
        std::vector<std::vector<std::vector<float>>> A(seqlen, std::vector<std::vector<float>>(nhead, std::vector<float>(total_len)));

        // Q @ K^T
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t kh = 0;
                for (size_t j = 0; j < total_len; ++j) {
                    float sum = 0.0f;
                    kh = (nkvhead == 1) ? 0 : (h / group_size);
                    for (size_t l = 0; l < d; ++l) {
                        size_t q_idx = i * nhead * d + h * d + l;
                        size_t k_idx = j * nkvhead * d + kh * d + l;
                        sum += q_data[q_idx] * k_data[k_idx];
                    }
                    A[i][h][j] = sum * scale;
                }
            }
        }

        // softmax with causal-like mask constructed as in test (diagonal = S-L)
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                float max_val = -std::numeric_limits<float>::infinity();
                int cutoff = static_cast<int>(i) + diag; // 允许的最大 j
                for (size_t j = 0; j < total_len; ++j) {
                    if (static_cast<int>(j) > cutoff) {
                        A[i][h][j] = -std::numeric_limits<float>::infinity();
                    } else {
                        max_val = std::max(max_val, A[i][h][j]);
                    }
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) {
                        continue;
                    }
                    A[i][h][j] = std::exp(A[i][h][j] - max_val);
                    sum_exp += A[i][h][j];
                }
                if (sum_exp == 0.0f) sum_exp = 1.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) {
                        A[i][h][j] = 0.0f;
                    } else {
                        A[i][h][j] /= sum_exp;
                    }
                }
            }
        }

        // Y = softmax(A) @ V
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t vh = 0;
                for (size_t l = 0; l < dv; ++l) {
                    float sum = 0.0f;
                    int cutoff = static_cast<int>(i) + diag;
                    for (int j = 0; j < static_cast<int>(total_len); ++j) {
                        if (j > cutoff) continue;
                        vh = (nkvhead == 1) ? 0 : (h / group_size);
                        size_t v_idx = static_cast<size_t>(j) * nkvhead * dv + vh * dv + l;
                        sum += A[i][h][static_cast<size_t>(j)] * v_data[v_idx];
                    }
                    size_t out_idx = i * nhead * dv + h * dv + l;
                    attn_val_data[out_idx] = sum;
                }
            }
        }
        break;
    }

    case LLAISYS_DTYPE_F16: {
        const fp16_t *q_data = reinterpret_cast<const fp16_t *>(q->data());
        const fp16_t *k_data = reinterpret_cast<const fp16_t *>(k->data());
        const fp16_t *v_data = reinterpret_cast<const fp16_t *>(v->data());
        fp16_t *attn_val_data = reinterpret_cast<fp16_t *>(attn_val->data());

        std::vector<std::vector<std::vector<float>>> A(seqlen, std::vector<std::vector<float>>(nhead, std::vector<float>(total_len)));

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < total_len; ++j) {
                    float sum = 0.0f;
                    size_t kh = (nkvhead == 1) ? 0 : (h / group_size);
                    for (size_t l = 0; l < d; ++l) {
                        size_t q_idx = i * nhead * d + h * d + l;
                        size_t k_idx = j * nkvhead * d + kh * d + l;
                        float qv = utils::cast<float>(q_data[q_idx]);
                        float kv = utils::cast<float>(k_data[k_idx]);
                        sum += qv * kv;
                    }
                    A[i][h][j] = sum * scale;
                }
            }
        }

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                float max_val = -std::numeric_limits<float>::infinity();
                int cutoff = static_cast<int>(i) + diag;
                for (size_t j = 0; j < total_len; ++j) {
                    if (static_cast<int>(j) > cutoff) {
                        A[i][h][j] = -std::numeric_limits<float>::infinity();
                    } else {
                        max_val = std::max(max_val, A[i][h][j]);
                    }
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) continue;
                    A[i][h][j] = std::exp(A[i][h][j] - max_val);
                    sum_exp += A[i][h][j];
                }
                if (sum_exp == 0.0f) sum_exp = 1.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) A[i][h][j] = 0.0f;
                    else A[i][h][j] /= sum_exp;
                }
            }
        }

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t l = 0; l < dv; ++l) {
                    float sum = 0.0f;
                    int cutoff = static_cast<int>(i) + diag;
                    for (int j = 0; j < static_cast<int>(total_len); ++j) {
                        if (j > cutoff) continue;
                        size_t vh = (nkvhead == 1) ? 0 : (h / group_size);
                        size_t v_idx = static_cast<size_t>(j) * nkvhead * dv + vh * dv + l;
                        float vv = utils::cast<float>(v_data[v_idx]);
                        sum += A[i][h][static_cast<size_t>(j)] * vv;
                    }
                    size_t out_idx = i * nhead * dv + h * dv + l;
                    attn_val_data[out_idx] = utils::cast<fp16_t>(sum);
                }
            }
        }
        break;
    }

    case LLAISYS_DTYPE_BF16: {
        const bf16_t *q_data = reinterpret_cast<const bf16_t *>(q->data());
        const bf16_t *k_data = reinterpret_cast<const bf16_t *>(k->data());
        const bf16_t *v_data = reinterpret_cast<const bf16_t *>(v->data());
        bf16_t *attn_val_data = reinterpret_cast<bf16_t *>(attn_val->data());

        std::vector<std::vector<std::vector<float>>> A(seqlen, std::vector<std::vector<float>>(nhead, std::vector<float>(total_len)));

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < total_len; ++j) {
                    float sum = 0.0f;
                    size_t kh = (nkvhead == 1) ? 0 : (h / group_size);
                    for (size_t l = 0; l < d; ++l) {
                        size_t q_idx = i * nhead * d + h * d + l;
                        size_t k_idx = j * nkvhead * d + kh * d + l;
                        float qv = utils::cast<float>(q_data[q_idx]);
                        float kv = utils::cast<float>(k_data[k_idx]);
                        sum += qv * kv;
                    }
                    A[i][h][j] = sum * scale;
                }
            }
        }

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                float max_val = -std::numeric_limits<float>::infinity();
                int cutoff = static_cast<int>(i) + diag;
                for (size_t j = 0; j < total_len; ++j) {
                    if (static_cast<int>(j) > cutoff) {
                        A[i][h][j] = -std::numeric_limits<float>::infinity();
                    } else {
                        max_val = std::max(max_val, A[i][h][j]);
                    }
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) continue;
                    A[i][h][j] = std::exp(A[i][h][j] - max_val);
                    sum_exp += A[i][h][j];
                }
                if (sum_exp == 0.0f) sum_exp = 1.0f;
                for (int j = 0; j < static_cast<int>(total_len); ++j) {
                    if (j > cutoff) A[i][h][j] = 0.0f;
                    else A[i][h][j] /= sum_exp;
                }
            }
        }

        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t l = 0; l < dv; ++l) {
                    float sum = 0.0f;
                    int cutoff = static_cast<int>(i) + diag;
                    for (int j = 0; j < static_cast<int>(total_len); ++j) {
                        if (j > cutoff) continue;
                        size_t vh = (nkvhead == 1) ? 0 : (h / group_size);
                        size_t v_idx = static_cast<size_t>(j) * nkvhead * dv + vh * dv + l;
                        float vv = utils::cast<float>(v_data[v_idx]);
                        sum += A[i][h][static_cast<size_t>(j)] * vv;
                    }
                    size_t out_idx = i * nhead * dv + h * dv + l;
                    attn_val_data[out_idx] = utils::cast<bf16_t>(sum);
                }
            }
        }
        break;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops
