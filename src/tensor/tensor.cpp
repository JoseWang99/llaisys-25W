#include "tensor.hpp"
#include "../ops/rearrange/op.hpp"
#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();
    if (ndim == 0) return true;

    size_t expected_stride = 1;
    for (size_t i = ndim; i-- > 0; ) {
        if (static_cast<size_t> (this->strides()[i]) != expected_stride) {
            return false;
        }
        expected_stride *= this->shape()[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim = this->ndim();
    if (order.size() != ndim) {
        EXCEPTION_INVALID_ARGUMENT("Permute order size must match tensor dimensions");
    }
    std::vector<bool> used(ndim, false);
    for (size_t idx : order) {
        if (idx >= ndim || used[idx]) {
            EXCEPTION_INVALID_ARGUMENT("Invalid permute order");
        }
        used[idx] = true;
    }

    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    if (!this->isContiguous()) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        EXCEPTION_SHAPE_MISMATCH;
    }

    size_t ndim_new = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_new);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_new; i++) {
        new_strides[ndim_new - i] = stride;
        stride *= shape[ndim_new - i];
    }

    TensorMeta new_meta{this->dtype(), shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if(dim >= this->ndim()) {
        EXCEPTION_INVALID_ARGUMENT("Dimension index out of range");
    }
    if (start >= end || end > this->shape()[dim]) {
        EXCEPTION_INVALID_ARGUMENT("Invalid slice range: start must be less than end, and end must not exceed dimension size");
    }
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    std::vector<ptrdiff_t> new_strides = this->strides();
    size_t new_offset = this->_offset + start * static_cast<size_t>(this->strides()[dim]) * this->elementSize();


    TensorMeta new_meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    size_t size = this->numel() * this->elementSize(); 
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, size);
    } else {
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(), src_, size, LLAISYS_MEMCPY_H2D);
    }

}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    ASSERT(this->deviceType() == LLAISYS_DEVICE_CPU,
           "contiguous: only CPU tensors are supported for now");
    auto result = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    llaisys::ops::rearrange(result, std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset)));
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
