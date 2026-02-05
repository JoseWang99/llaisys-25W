import ctypes
from ctypes import c_int64, c_size_t, c_float, c_int, POINTER, c_void_p
from . import LIB_LLAISYS
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t


class LlaisysQwen2Meta(ctypes.Structure):
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
        ("attention_dropout", c_float),
        ("initializer_range", c_float),
        ("max_window_layers", c_size_t),
        ("sliding_window", c_size_t),
        ("tie_word_embeddings", c_int),
        ("use_cache", c_int),
        ("use_mrope", c_int),
        ("use_sliding_window", c_int),
    ]

class LlaisysQwen2Weights(ctypes.Structure):
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

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2Model_t, ctypes.POINTER(ctypes.c_int64),]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64