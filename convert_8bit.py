from transformers import AutoTokenizer
import torch

from finetune_dynamic_moe import import_moe_classes, _load_local_checkpoint_state_dict

src_ckpt = "/data/cyx/models/out_piqa_lowrank_entmax"   # 训练后输出目录
dst_ckpt = "/data/cyx/models/out_piqa_lowrank_entmax"        # 你希望推理时使用的目录

# 1) 用相同的模型类加载 checkpoint state_dict（避免 transformers.from_pretrained 的 meta/quantized加载问题）
model_cls, config_cls = import_moe_classes()
config = config_cls.from_pretrained(src_ckpt)

model = model_cls(config)
state = _load_local_checkpoint_state_dict(src_ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"loaded state dict: missing={len(missing)} unexpected={len(unexpected)}")

# 2) 转为 float16
model = model.half()

# 3) 保存为 float16 checkpoint（直接写 state_dict，避免 `save_pretrained` 导致 deepspeed/accelerate import 问题）
from pathlib import Path
from safetensors.torch import save_file as safetensors_save

out_dir = Path(dst_ckpt)
out_dir.mkdir(parents=True, exist_ok=True)

# Save config (config.json)
config.save_pretrained(out_dir)

# Save weights as safetensors
state_dict = model.state_dict()
# ensure dtype is float16 for all params
for k, v in state_dict.items():
    if v.dtype != torch.float16:
        state_dict[k] = v.to(torch.float16)

safetensors_save(state_dict, out_dir / "model.safetensors")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(src_ckpt, trust_remote_code=True)
tokenizer.save_pretrained(out_dir)

print("✅ Saved fp16 checkpoint to:", out_dir)
