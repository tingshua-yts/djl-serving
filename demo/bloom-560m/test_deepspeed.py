from transformers import pipeline
import deepspeed
import torch
generator = pipeline("text-generation", model="bigscience/bloom-560m")
result = generator("deepspeed is")
print(f"pipeline result: {result}")

generator.model = deepspeed.init_inference(
    generator.model,
    mp_size=2,
    dtype=torch.float,
    replace_method='auto',
    replace_with_kernel_inject=True
    #injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
result = generator("deepspeed is ")
print(f"deepspeed result:{result}")