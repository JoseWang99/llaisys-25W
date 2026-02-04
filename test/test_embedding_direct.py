"""
直接测试 Embedding 算子是否工作正常
"""
import sys
sys.path.insert(0, '/home/jose/llaisys/python')

import llaisys
import numpy as np
import ctypes

# 1. 创建一个小的 Embedding 权重表（10个词，每个维度为4）
vocab_size = 10
embed_dim = 4
weight_np = np.random.randn(vocab_size, embed_dim).astype(np.float32)

print("=== Embedding Weight (NumPy) ===")
print(weight_np[:5])  # 打印前5行

# 2. 使用 Tensor 类
weight_tensor = llaisys.Tensor((vocab_size, embed_dim), llaisys.DataType.F32, llaisys.DeviceType.CPU)
index_tensor = llaisys.Tensor((3,), llaisys.DataType.I64, llaisys.DeviceType.CPU)
out_tensor = llaisys.Tensor((3, embed_dim), llaisys.DataType.F32, llaisys.DeviceType.CPU)

# 3. 手动加载数据到 weight 和 index
from llaisys.libllaisys import LIB_LLAISYS

index_np = np.array([2, 5, 7], dtype=np.int64)

# 使用 _tensor 属性访问底层句柄
LIB_LLAISYS.tensorLoad(weight_tensor._tensor, weight_np.ctypes.data_as(ctypes.c_void_p))
LIB_LLAISYS.tensorLoad(index_tensor._tensor, index_np.ctypes.data_as(ctypes.c_void_p))

# 4. 调用 Embedding 算子（直接使用 C API）
print("\n=== Calling Embedding Op ===")

# 设置函数签名
LIB_LLAISYS.llaisysEmbedding.argtypes = [
    ctypes.c_void_p,  # out
    ctypes.c_void_p,  # index
    ctypes.c_void_p   # weight
]
LIB_LLAISYS.llaisysEmbedding.restype = None

# 调用
LIB_LLAISYS.llaisysEmbedding(out_tensor._tensor, index_tensor._tensor, weight_tensor._tensor)

# 5. 回读结果（手动从 C++ 内存读取）
LIB_LLAISYS.tensorGetData.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.tensorGetData.restype = ctypes.c_void_p

out_ptr = LIB_LLAISYS.tensorGetData(out_tensor._tensor)
result = np.ctypeslib.as_array(
    (ctypes.c_float * (3 * embed_dim)).from_address(out_ptr)
).reshape(3, embed_dim).copy()

print("\n=== Embedding Output (LLAISYS) ===")
print(result)

print("\n=== Expected Output (Ground Truth) ===")
expected = weight_np[[2, 5, 7]]
print(expected)

# 6. 验证
if np.allclose(result, expected, atol=1e-5):
    print("\n✅ Embedding 算子工作正常！")
else:
    print("\n❌ Embedding 输出不匹配！")
    print("差异：", np.abs(result - expected).max())