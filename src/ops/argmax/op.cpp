#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t n = vals->numel();

    if (n == 0) {
        EXCEPTION_INVALID_ARGUMENT("argmax on empty tensor");
    }
    switch(vals->dtype()) {

        case LLAISYS_DTYPE_I64: {
            const int64_t *v = reinterpret_cast<const int64_t *>(vals->data());

            int64_t best_val = v[0];
            int64_t best_idx = 0;

            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {
                    best_val = v[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }
            *reinterpret_cast<int64_t *>(max_val->data()) = best_val;
            *reinterpret_cast<int64_t *>(max_idx->data()) = best_idx;
            break;  
        }

        case LLAISYS_DTYPE_I32: {
            // 类似处理 int32 类型
            const int32_t *v = reinterpret_cast<const int32_t *>(vals->data());
            int32_t best_val = v[0];
            int32_t best_idx = 0;
            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {
                    best_val = v[i];
                    best_idx = static_cast<int32_t>(i);
                }
            }
            *reinterpret_cast<int32_t *>(max_val->data()) = best_val;
            *reinterpret_cast<int32_t *>(max_idx->data()) = best_idx;
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            // 假设 float16 用 uint16_t 表示（项目可能有自定义 half 类型）
            // 注意：float16 比较应转换到 float，但这里为简单按 uint16_t 比较（可能不准确）
            const uint16_t *v = reinterpret_cast<const uint16_t *>(vals->data());
            uint16_t best_val = v[0];
            int64_t best_idx = 0;
            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {  // 按位比较，可能需要转换到 float
                    best_val = v[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }
            *reinterpret_cast<uint16_t *>(max_val->data()) = best_val;
            *reinterpret_cast<int64_t *>(max_idx->data()) = best_idx;
            break;
        }

        case LLAISYS_DTYPE_F32: {
            // 类似处理 float32 类型
            const float *v = reinterpret_cast<const float *>(vals->data());
            float best_val = v[0];
            int64_t best_idx = 0;  // 索引始终用 int64 存储
            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {
                    best_val = v[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }
            *reinterpret_cast<float *>(max_val->data()) = best_val;
            *reinterpret_cast<int64_t *>(max_idx->data()) = best_idx;
            break;
        }
        
        case LLAISYS_DTYPE_F64: {
            // 类似处理 float64 类型
            const double *v = reinterpret_cast<const double *>(vals->data());
            double best_val = v[0];
            int64_t best_idx = 0;
            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {
                    best_val = v[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }
            *reinterpret_cast<double *>(max_val->data()) = best_val;
            *reinterpret_cast<int64_t *>(max_idx->data()) = best_idx;
            break;
        }

        case LLAISYS_DTYPE_BF16: {
            const uint16_t *v = reinterpret_cast<const uint16_t *>(vals->data());
            uint16_t best_val = v[0];
            int64_t best_idx = 0;
            for (size_t i = 1; i < n; ++i) {
                if (v[i] > best_val) {  // 按位比较，可能需要转换到 float
                    best_val = v[i];
                    best_idx = static_cast<int64_t>(i);
                }
            }
            *reinterpret_cast<uint16_t *>(max_val->data()) = best_val;
            *reinterpret_cast<int64_t *>(max_idx->data()) = best_idx;
            break;
        }
        
        default:
            // 如果数据类型不支持，抛出异常
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    
}
} // namespace llaisys::ops