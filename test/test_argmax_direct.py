"""
直接测试 ArgMax 算子是否工作正常
"""
import sys
sys.path.insert(0, '/home/jose/llaisys/python')

import llaisys
import numpy as np
import ctypes

# 1. 创建一个简单的 Logits 向量（vocab_size=10）
# 最大值应该在索引 7（值为 5.0）
logits_np = np.array([0.1, -0.5, 0.3, -1.2, 0.8, -0.3, 1.5, 5.0, 0.2, -0.8], dtype=np.float32)

print("=== Input Logits ===")
print(logits_np)
print(f"Expected ArgMax: {np.argmax(logits_np)}")  # 应该是 7

# 2. 创建 LLAISYS Tensor
from llaisys.libllaisys import LIB_LLAISYS

logits_tensor = llaisys.Tensor((10,), llaisys.DataType.F32, llaisys.DeviceType.CPU)
LIB_LLAISYS.tensorLoad(logits_tensor._tensor, logits_np.ctypes.data_as(ctypes.c_void_p))

# 3. 调用 ArgMax（返回值应该是 int64）
print("\n=== Calling ArgMax Op ===")

LIB_LLAISYS.llaisysArgmax.argtypes = [ctypes.c_void_p]  # 注意是 llaisysArgmax
LIB_LLAISYS.llaisysArgmax.restype = ctypes.c_int64

result = LIB_LLAISYS.llaisysArgmax(logits_tensor._tensor)

print(f"\n=== ArgMax Output (LLAISYS) ===")
print(f"Result: {result}")

# 4. 验证
expected = np.argmax(logits_np)
if result == expected:
    print(f"\n✅ ArgMax 算子工作正常！(返回了正确的索引 {result})")
else:
    print(f"\n❌ ArgMax 输出错误！")
    print(f"期望: {expected}, 实际: {result}")