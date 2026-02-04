from typing import Sequence, Optional, Dict, Any, List
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, llaisysQwen2Model_t
from ..libllaisys.tensor import llaisysDataType_t, dtype_str_to_enum
from ..tensor import Tensor
from pathlib import Path
import json
import numpy as np
import safetensors.numpy
import ctypes
import torch
import time


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        
        # 1. 读取 Config
        with open(self.model_path / "config.json", 'r') as f:
            self.config = json.load(f)
            
        # 2. 准备 Meta
        self.meta = LlaisysQwen2Meta()
        
        # [Fix] 强制指定使用 F32 精度，配合下面权重的强转
        self.meta.dtype = DataType.F32.value

        self.meta.nlayer = self.config["num_hidden_layers"]
        self.meta.hs = self.config["hidden_size"]
        self.meta.nh = self.config["num_attention_heads"]
        self.meta.nkvh = self.config["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = self.config["intermediate_size"]
        self.meta.maxseq = self.config["max_position_embeddings"]
        self.meta.voc = self.config["vocab_size"]
        self.meta.epsilon = self.config["rms_norm_eps"]
        self.meta.theta = self.config.get("rope_theta", 1000000.0)
        self.meta.end_token = self.config.get("eos_token_id", 151643)

        # === 调试打印 Meta 参数 ===
        print("=== Python Meta Params ===")
        print(f"nlayer={self.meta.nlayer}, hs={self.meta.hs}, nh={self.meta.nh}")
        print(f"vocab={self.meta.voc}, maxseq={self.meta.maxseq}")
        # ======================

        # 3. 创建 C 模型实例
        self.model_handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            self.device.value, 
            None, 
            0
        )
        
        if not self.model_handle:
            raise RuntimeError("Failed to create Qwen2 model backend")

        # [Fix] 明确声明 infer 签名，强制使用 c_int64 以匹配 C++ 行为
        try:
            LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
                llaisysQwen2Model_t,
                ctypes.POINTER(ctypes.c_int64), # 必须是 int64
                ctypes.c_size_t,
            ]
            LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64
        except Exception:
            pass
            
        # 4. 加载权重
        self._load_weights()

    @staticmethod
    def _torch_dtype_to_llaisys(dtype: torch.dtype) -> DataType:
        if dtype == torch.float16:
            return DataType.F16
        if dtype == torch.float32:
            return DataType.F32
        if dtype == torch.float64:
            return DataType.F64
        if dtype == torch.bfloat16:
            return DataType.BF16
        if dtype == torch.int64:
            return DataType.I64
        if dtype == torch.int32:
            return DataType.I32
        if dtype == torch.int16:
            return DataType.I16
        if dtype == torch.int8:
            return DataType.I8
        if dtype == torch.uint8:
            return DataType.U8
        if dtype == torch.bool:
            return DataType.BOOL
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    @staticmethod
    def _torch_to_llaisys_tensor(tensor: torch.Tensor, device: DeviceType) -> Tensor:
        cpu_tensor = tensor.detach().cpu().contiguous()
        
        # === [Fix] 强制转换所有浮点权重为 F32 ===
        # 避免 BF16 在 C++ 端可能的计算兼容性问题
        if cpu_tensor.is_floating_point():
            cpu_tensor = cpu_tensor.to(torch.float32)
        # ==================================
        
        if torch.isnan(cpu_tensor).any():
            print(f"Warning: Computed tensor contains NaN! dtype={cpu_tensor.dtype}")
        
        llaisys_dtype = Qwen2._torch_dtype_to_llaisys(cpu_tensor.dtype)
        
        # 1. 显式调用 C API 创建 Tensor 句柄
        shape = tuple(cpu_tensor.shape)
        c_shape = (ctypes.c_size_t * len(shape))(*shape)
        
        handle = LIB_LLAISYS.tensorCreate(
            c_shape, 
            len(shape), 
            llaisys_dtype.value if hasattr(llaisys_dtype, 'value') else llaisys_dtype, 
            device.value if hasattr(device, 'value') else device, 
            0
        )
        if not handle:
            raise RuntimeError("Failed to allocate tensor")

        # 2. 实例化 Tensor 对象
        llaisys_tensor = Tensor(shape) 
        llaisys_tensor.handle = handle
        llaisys_tensor.dtype = llaisys_dtype
        
        # 3. 加载数据
        # 由于上面已经强转为 F32，这里通常走 numpy 路径
        if cpu_tensor.dtype == torch.bfloat16:
            LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(cpu_tensor.data_ptr()))
        else:
            LIB_LLAISYS.tensorLoad(handle, cpu_tensor.numpy().ctypes.data_as(ctypes.c_void_p))
            
        return llaisys_tensor

    def _load_weights(self):
        weights_c = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_handle).contents
        
        # 构建索引 (key -> filename)
        key_to_file = {}
        st_files = sorted(list(self.model_path.glob("*.safetensors")))
        for f in st_files:
            with safetensors.numpy.safe_open(f, framework="numpy") as open_f:
                for k in open_f.keys():
                    key_to_file[k] = f

        def load(name):
            if name not in key_to_file:
                print(f"Warn: {name} missing")
                return None
            
            f_path = key_to_file[name]
            
            # 使用 PyTorch 加载
            with safetensors.safe_open(f_path, framework="pt") as f:
                pt_tensor = f.get_tensor(name)
                
            # 转换为 LLAISYS tensor
            t = self._torch_to_llaisys_tensor(pt_tensor, self.device)
            return t.handle

        # Assign weights
        weights_c.in_embed = load("model.embed_tokens.weight")
        weights_c.out_embed = load("lm_head.weight") if "lm_head.weight" in key_to_file else weights_c.in_embed
        weights_c.out_norm_w = load("model.norm.weight")

        for i in range(self.meta.nlayer):
            p = f"model.layers.{i}"
            weights_c.attn_norm_w[i] = load(f"{p}.input_layernorm.weight")
            weights_c.attn_q_w[i] = load(f"{p}.self_attn.q_proj.weight")
            weights_c.attn_q_b[i] = load(f"{p}.self_attn.q_proj.bias")
            weights_c.attn_k_w[i] = load(f"{p}.self_attn.k_proj.weight")
            weights_c.attn_k_b[i] = load(f"{p}.self_attn.k_proj.bias")
            weights_c.attn_v_w[i] = load(f"{p}.self_attn.v_proj.weight")
            weights_c.attn_v_b[i] = load(f"{p}.self_attn.v_proj.bias")
            weights_c.attn_o_w[i] = load(f"{p}.self_attn.o_proj.weight")
            
            weights_c.mlp_norm_w[i] = load(f"{p}.post_attention_layernorm.weight")
            weights_c.mlp_gate_w[i] = load(f"{p}.mlp.gate_proj.weight")
            weights_c.mlp_up_w[i] = load(f"{p}.mlp.up_proj.weight")
            weights_c.mlp_down_w[i] = load(f"{p}.mlp.down_proj.weight")

        # === 调试：回读验证 C++ 内存 ===
        print("=== Verifying Weights in C++ Memory ===")
        try:
            LIB_LLAISYS.tensorGetData.argtypes = [ctypes.c_void_p]
            LIB_LLAISYS.tensorGetData.restype = ctypes.c_void_p
            
            ptr = LIB_LLAISYS.tensorGetData(weights_c.in_embed)
            if not ptr:
                print("FATAL: tensorGetData returned NULL!")
            else:
                arr = (ctypes.c_float * 10).from_address(ptr)
                debug_vals = list(arr)
                print(f"Embed Weight [0:10]: {[f'{x:.4e}' for x in debug_vals]}")
                
                if all(x == 0 for x in debug_vals):
                    print("!!! CRITICAL FAILURE: Weights are ALL ZERO in C++ memory !!!")
                else:
                    print("Weights look OK.")
        except Exception as e:
            print(f"Verification failed: {e}")

        # 打印指针检查
        try:
            def _ptr_val(x):
                try: return int(x)
                except: return None
            print("weight pointers:",
                  "in_embed=", _ptr_val(weights_c.in_embed),
                  "out_norm_w=", _ptr_val(weights_c.out_norm_w))
        except:
            pass

    def __del__(self):
        if hasattr(self, 'model_handle') and self.model_handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_handle)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if not self.model_handle:
            raise RuntimeError("Model is not initialized")
            
        if max_new_tokens is None:
            max_new_tokens = 1
            
        if not isinstance(inputs, Sequence) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty sequence of token ids")

        tokens = list(int(t) for t in inputs)
        return self._infer_dialog(tokens, max_new_tokens)

    def _infer_dialog(self, tokens: Sequence[int], max_steps: int) -> List[int]:
        if max_steps is None:
            max_steps = 1
        
        # 保存完整对话历史用于返回
        full_response = list(tokens)
        
        # 下一轮要喂给模型的 tokens (初始为 Prompt)
        next_input_tokens = list(tokens)

        print(f"[infer] Starting inference. Max steps: {max_steps}", flush=True)
        print(f"[Debug] Vocab Size: {self.meta.voc}")
        
        for step in range(max_steps):
            # 准备输入数组
            in_len = len(next_input_tokens)
            
            # [Fix] 使用 c_int64，防止 C++ 读取越界/错位
            c_in_buf = (ctypes.c_int64 * in_len)(*next_input_tokens)
            buf_ptr = ctypes.cast(c_in_buf, ctypes.POINTER(ctypes.c_int64))
            
            print(f"[infer] C++ step={step} in_len={in_len} ...", end="", flush=True)
            t0 = time.time()
            
            # 推理：返回下一个 token ID
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_handle, 
                buf_ptr, 
                ctypes.c_size_t(in_len)
            )
            
            dt = time.time() - t0
            print(f" Done. next={next_token} t={dt:.3f}s", flush=True)
            
            full_response.append(next_token)
            
            if next_token == self.meta.end_token:
                print(f"[infer] Stop token reached (token={next_token}).")
                break
            
            
            
            # 【重要】下一轮只传最新生成的 1 个 token (Decode 阶段)
            # C++ 端应当内部维护 KV Cache 指针
            next_input_tokens = [next_token]
            
        return full_response