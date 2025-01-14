import json
import sys
import torch
from torch import nn
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from typing import Callable, Tuple
from utils import runner

LLAMA_3_2_1B_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.45.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
    "_commit_hash": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}

llama_cfg = LlamaConfig(**LLAMA_3_2_1B_CFG)
llama_cfg.num_hidden_layers = 1
llama_cfg.batch_size = 1
llama_cfg.seq_len = 6
configs = {}
configs[llama_cfg.name_or_path] = llama_cfg

if __name__ == "__main__":
    for name,cfg in configs.items():
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        def inputs():
            args = dict(
                        cache_positions=torch.arange(cfg.seq_len, device="cuda"),
                        input_ids=torch.tensor([[128000, 791, 1401, 311, 2324, 374]], device="cuda"),
                        #input_ids=torch.randint(0, cfg.bos_token_id, (cfg.batch_size, cfg.seq_len), device="cuda"),
                        attention_mask=torch.ones(1, cfg.seq_len, dtype=torch.int64, device="cuda"),
                        inputs_embeds=None,
                        use_cache=True,
                        return_dict=True,
                    )
            return args
 
        model = LlamaForCausalLM(cfg).cuda().bfloat16().requires_grad_(False).eval()
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, None)
