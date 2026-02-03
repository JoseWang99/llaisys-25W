#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

/**
 * @brief 表示张量的元数据结构体。
 *
 * @struct TensorMeta
 * @var dtype 张量的数据类型（llaisysDataType_t）。
 * @var shape 张量的形状，每个元素表示对应维度的大小。
 * @var strides 张量的步长，每个元素表示在该维度上移动一个元素时需要跳过的内存单元数。
 */
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

/**
 * @class Tensor
 * @brief 表示多维张量的数据结构，支持多种数据类型和设备。
 *
 * 构造函数被私有化的原因：
 * - 通过私有化构造函数，禁止用户直接实例化 Tensor 对象，强制用户通过静态工厂方法（如 create）来创建 Tensor 实例。
 * - 这样可以更好地控制对象的创建流程，例如内存分配、元数据初始化等，保证对象始终处于有效状态。
 * - 便于实现引用计数、内存池等高级特性，提升性能和安全性。
 * - 统一管理 Tensor 的生命周期，防止资源泄漏或非法操作。
 */
class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
