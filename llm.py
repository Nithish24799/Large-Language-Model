import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

input_text = "Once upon a time, "
input_ids = tokenizer.encode(input_text, return_tensors="tf")

output = model.generate(input_ids, max_length=100, num_return_sequences=1)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
