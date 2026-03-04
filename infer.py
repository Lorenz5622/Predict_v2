import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from qwen_moe.modeling.modeling_moe import Qwen2MoeForCausalLM
from qwen_moe.modeling.configuration_moe import Qwen2MoeConfig
# 配置4bit量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载模型和tokenizer
model_path = "/data/cyx/models/Qwen1.5-MoE-A2.7B"
model_config = Qwen2MoeConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2MoeForCausalLM.from_pretrained(
    model_path,
    config=model_config,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# 准备输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about the highest mountain in the world in English."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
