!pip install -q transformers einops accelerate langchain bitsandbytes

!nvidia-smi

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)