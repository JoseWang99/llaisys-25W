from ctypes import *
from . import LIB_LLAISYS
from .tensor import llaisysTensor_t

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

# 使用 POINTER(llaisysTensor_t) 是因为 C 结构中是指针数组
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

llaisysQwen2Model_t = c_void_p

# 函数原型
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), c_int, POINTER(c_int), c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2Model_t, POINTER(c_int64), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64
