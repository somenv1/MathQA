import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
import pandas as pd
import random
import transformers
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from huggingface_hub import login

with open('hg.token','r') as ft:
	token = ft.read().strip()

login(token=token) #Log into huggingface

model_path = "google/gemma-2-2b-it"   # Specify the path to the model
adapter_path = "mathwell/checkpoint-4250"   # Specify the path to the adapter weights

tokenizer_default = AutoTokenizer.from_pretrained(model_path) # Load tokenizer
tokenizer_new = AutoTokenizer.from_pretrained(adapter_path) # Load tokenizer

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
) # Set up bitsandbytes config to load model in 4 bit

def generate(tokenizer,model):
	prompt = " If I have to catch a bus at 5 in the morning, what time will I have to wake up if: I need 20 min to take a bath, 5 min to brush my teeth, 15 min to have breakfast, 10 min to reach the bus stop. Generate 3 grade school math word problems in the context of sports , no intruction , no explanation, only questions " #TODO
	#Query the model 
	inputs = tokenizer.encode(prompt, return_tensors="pt")
	attention_mask = torch.ones_like(inputs)
	attention_mask = attention_mask.to('cuda') #Line added
	inputs = inputs.to('cuda')
	output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 512, do_sample = False)
	#output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = False)
	generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
	print(generated_text)


model_default = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto",
    torch_dtype=torch.bfloat16, use_auth_token=True) # Load model in 4 bit

print("###################DEFAULT MODEL###################")
#for name, param in model_default.named_parameters():
#	print(name)
generate(tokenizer_default,model_default)

model_new = PeftModel.from_pretrained(model_default, adapter_path) # Create PEFT model 

print("###################TRAINED MODEL###################")
#print(model_new.peft_config)
#for name, param in model_new.named_parameters():
#	print(name)
generate(tokenizer_new,model_new)
