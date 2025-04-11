# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-30b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))